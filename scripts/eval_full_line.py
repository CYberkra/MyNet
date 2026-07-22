from pathlib import Path
import csv,sys,argparse,json
import numpy as np, torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pgdacsnet.model_raw_unet import build_model, compress_raw
from pgdacsnet.model_interfaces import unpack_pgda_output
from pgdacsnet.font_utils import get_chinese_font
from pgdacsnet.spatial_orientation import (
    align_array_for_display, get_line_orientation, profile_index_order,
)
FONT=get_chinese_font()


def profile_display_flip_or_false(line_name):
    """Return the survey display flip, with acquisition-order fallback.

    Canonical YingShan lines must be registered.  Synthetic, smoke, and future
    sites may legitimately have no survey-profile contract yet; evaluation of
    their numeric outputs must not fail merely because a plot orientation is
    unavailable.
    """
    try:
        return bool(get_line_orientation(line_name).profile_display_flip)
    except KeyError:
        return False

def add_terrain_channels(x, line_name, start, end, cfg, data_root):
    feature_names=cfg.get('terrain_feature_names', [])
    if not cfg.get('use_terrain_features', False) or not feature_names:
        return x
    feature_dir=cfg.get('terrain_feature_dir','terrain_features')
    fpath=data_root/feature_dir/f'{line_name}_terrain_features.npz'
    z=np.load(fpath,allow_pickle=False)
    names=[str(v) for v in z['feature_names']]
    idx=[names.index(name) for name in feature_names]
    feat=torch.from_numpy(z['features'][idx,start:end]).float().to(x.device)
    H,W=x.shape[-2],x.shape[-1]
    feat=F.interpolate(feat[None,:,None,:],(H,W),mode='bilinear',align_corners=False)
    return torch.cat([x,feat],dim=1)

def normalize_raw_channel_4d(x, cfg):
    if not cfg.get('per_trace_robust_norm', False):
        return x
    clip=float(cfg.get('per_trace_robust_clip',6.0))
    eps=float(cfg.get('per_trace_robust_eps',1e-4))
    raw=x[:,:1]
    med=raw.median(dim=2,keepdim=True).values
    mad=(raw-med).abs().median(dim=2,keepdim=True).values
    norm=torch.clamp((raw-med)/(1.4826*mad+eps),-clip,clip)/clip
    x=x.clone()
    x[:,:1]=norm
    return x

def unpack_model_output(out):
    mask_logits, presence_logits, center_logits = unpack_pgda_output(out)
    if mask_logits is None or presence_logits is None:
        raise ValueError('model output must include mask and presence logits')
    return mask_logits, presence_logits, center_logits

def resolve_data_root(data_root=None, cfg=None):
    value=data_root or (cfg or {}).get('data_root','data')
    p=Path(value)
    return p if p.is_absolute() else ROOT/p


def real_nopick_reporting_policy(configs):
    """Return whether measured no-pick artifacts may be emitted for an ensemble."""
    policies={str(item.get('real_nopick_metric_reporting','allowed')).strip().lower() for item in configs}
    if len(policies)>1:
        raise ValueError('All run dirs must agree on real_nopick_metric_reporting for one evaluation.')
    policy=next(iter(policies), 'allowed')
    if policy not in {'allowed','forbidden'}:
        raise ValueError(f'Unsupported real_nopick_metric_reporting={policy!r}')
    return policy

def centerline(arr,min_sum=1e-4):
    H,W=arr.shape; ys=np.arange(H,dtype=np.float32)[:,None]; s=arr.sum(axis=0); c=(arr*ys).sum(axis=0)/np.maximum(s,1e-6); valid=s>min_sum; c[~valid]=np.nan; return c,valid


def dp_ridge_centerline(prob, max_jump=8, smooth_weight=0.08, min_presence=None, search_min_sample=None, search_max_sample=None):
    """Extract one smooth ridge from a probability image using vectorized dynamic programming.
    Returns center sample per trace and a validity mask. This is post-processing, not network input.
    """
    H,W=prob.shape
    p=np.clip(prob.astype(np.float32),1e-6,1.0)
    unary=-np.log(p)
    if search_min_sample is not None or search_max_sample is not None:
        lo=0 if search_min_sample is None else max(0,int(search_min_sample))
        hi=H-1 if search_max_sample is None else min(H-1,int(search_max_sample))
        mask=np.ones(H,dtype=bool); mask[lo:hi+1]=False
        unary[mask,:]+=20.0
    dp=np.empty((H,W),np.float32)
    back=np.zeros((H,W),np.int16)
    dp[:,0]=unary[:,0]
    offsets=np.arange(-max_jump,max_jump+1,dtype=np.int16)
    big=np.float32(1e6)
    for x in range(1,W):
        prev=dp[:,x-1]
        cand=np.full((len(offsets),H), big, dtype=np.float32)
        for oi,off in enumerate(offsets):
            penalty=np.float32(smooth_weight*(int(off)**2))
            # current y came from previous y+off
            if off<0:
                cand[oi,-off:]=prev[:off]+penalty
            elif off>0:
                cand[oi,:-off]=prev[off:]+penalty
            else:
                cand[oi,:]=prev+penalty
        arg=np.argmin(cand,axis=0).astype(np.int16)
        best=cand[arg,np.arange(H)]
        dp[:,x]=unary[:,x]+best
        predecessor=np.arange(H,dtype=np.int32)+offsets[arg].astype(np.int32)
        back[:,x]=np.clip(predecessor,0,H-1).astype(np.int16)
    path=np.zeros(W,np.float32)
    y=int(np.argmin(dp[:,W-1])); path[W-1]=y
    for x in range(W-1,0,-1):
        y=int(back[y,x]); path[x-1]=y
    valid=np.ones(W,dtype=bool)
    if min_presence is not None:
        valid=min_presence.astype(bool)
        path=path.copy(); path[~valid]=np.nan
    return path, valid


def breakable_dp_ridge_centerline(prob, pres_pred, presence_thr=0.45, path_prob_thr=0.20, min_segment=16, max_jump=8, smooth_weight=0.08, search_min_sample=None, search_max_sample=None):
    """Run DP only inside contiguous high-confidence trace segments.
    This avoids forcing one ridge through low-confidence gaps in real B-scans.
    """
    H,W=prob.shape
    lo=0 if search_min_sample is None else max(0,int(search_min_sample))
    hi=H-1 if search_max_sample is None else min(H-1,int(search_max_sample))
    local_peak=np.nanmax(prob[lo:hi+1,:],axis=0)
    gate=(pres_pred>=presence_thr)&(local_peak>=path_prob_thr)
    path=np.full(W,np.nan,np.float32)
    valid=np.zeros(W,dtype=bool)
    start=None
    for i,ok in enumerate(np.r_[gate,False]):
        if ok and start is None:
            start=i
        if (not ok) and start is not None:
            end=i
            if end-start>=int(min_segment):
                sub_path,sub_valid=dp_ridge_centerline(
                    prob[:,start:end],
                    max_jump=max_jump,
                    smooth_weight=smooth_weight,
                    min_presence=None,
                    search_min_sample=search_min_sample,
                    search_max_sample=search_max_sample,
                )
                path[start:end]=sub_path
                valid[start:end]=sub_valid
            start=None
    return path, valid

def soft_dice(pred,gt,weight=None,eps=1e-6):
    if weight is None: weight=np.ones_like(gt,dtype=np.float32)
    return float(2*(pred*gt*weight).sum()/(((pred+gt)*weight).sum()+eps))

def wbce(pred,gt,weight=None,eps=1e-6):
    pred=np.clip(pred,eps,1-eps); b=-(gt*np.log(pred)+(1-gt)*np.log(1-pred))
    if weight is None: return float(b.mean())
    return float((b*weight).sum()/(weight.sum()+eps))


def normalise_trace_distribution(arr, eps=1e-8):
    """Return per-trace P(t|trace) and a validity mask for non-zero traces."""
    arr=np.clip(np.asarray(arr,dtype=np.float32),0.0,None)
    mass=arr.sum(axis=0,keepdims=True)
    valid=mass[0]>eps
    out=np.zeros_like(arr,dtype=np.float32)
    if valid.any():
        out[:,valid]=arr[:,valid]/mass[:,valid]
    return out,valid


def normalise_time_distribution(arr, eps=1e-8):
    """Backward-compatible distribution normaliser returning only probabilities."""
    return normalise_trace_distribution(arr,eps)[0]


def distribution_path_metrics(path_prob, gt, dt_ns, label_w=None, eps=1e-8):
    """Metrics for a temporal probability distribution, never a segmentation mask."""
    pred,pred_valid=normalise_trace_distribution(path_prob,eps)
    target,target_valid=normalise_trace_distribution(gt,eps)
    valid=target_valid & pred_valid
    result={
        'path_target_valid_trace_count':int(target_valid.sum()),
        'path_distribution_valid_trace_count':int(valid.sum()),
        'path_distribution_coverage':float(valid.sum()/max(int(target_valid.sum()),1)),
        'path_nll':float('nan'),
        'path_emd':float('nan'),
        'path_expected_mae_sample':float('nan'),
        'path_expected_mae_ns':float('nan'),
        'path_expected_median_ae_ns':float('nan'),
        'path_expected_p90_ae_ns':float('nan'),
        'path_expected_p95_ae_ns':float('nan'),
    }
    for tol in (5,10,20):
        result[f'path_hit_rate_le_{tol}ns']=float('nan')
    if not valid.any():
        return result
    if label_w is None:
        weights=np.ones(int(valid.sum()),dtype=np.float64)
    else:
        weights=(0.25+np.asarray(label_w,dtype=np.float64)[valid])
    weights=weights/np.maximum(weights.sum(),eps)
    p=np.clip(pred[:,valid].astype(np.float64),eps,1.0)
    q=target[:,valid].astype(np.float64)
    nll=-(q*np.log(p)).sum(axis=0)
    emd=np.abs(np.cumsum(pred[:,valid],axis=0)-np.cumsum(target[:,valid],axis=0)).sum(axis=0)
    ys=np.arange(pred.shape[0],dtype=np.float64)[:,None]
    pred_center=(pred[:,valid]*ys).sum(axis=0)
    target_center=(target[:,valid]*ys).sum(axis=0)
    abs_err=np.abs(pred_center-target_center)
    abs_err_ns=abs_err*float(dt_ns)
    result.update({
        'path_nll':float((nll*weights).sum()),
        'path_emd':float((emd*weights).sum()),
        'path_expected_mae_sample':float((abs_err*weights).sum()),
        'path_expected_mae_ns':float((abs_err_ns*weights).sum()),
        'path_expected_median_ae_ns':float(np.median(abs_err_ns)),
        'path_expected_p90_ae_ns':float(np.percentile(abs_err_ns,90)),
        'path_expected_p95_ae_ns':float(np.percentile(abs_err_ns,95)),
    })
    for tol in (5,10,20):
        result[f'path_hit_rate_le_{tol}ns']=float((abs_err_ns<=tol).mean())
    return result


def curve_distribution_metrics(curve_prob, gt, label_w, dt_ns=1.0, eps=1e-8):
    """Curve-head metrics with an explicit curve prefix."""
    base=distribution_path_metrics(curve_prob,gt,dt_ns,label_w,eps)
    mapped={}
    for key,value in base.items():
        suffix=key[len('path_'):] if key.startswith('path_') else key
        mapped[f'curve_{suffix}']=value
    # Stable aliases used by earlier reports, still distribution-only.
    mapped['curve_valid_trace_count']=mapped['curve_distribution_valid_trace_count']
    mapped['curve_emd_sample']=mapped['curve_emd']
    mapped['curve_expected_center_mae_sample']=mapped['curve_expected_mae_sample']
    mapped['curve_expected_center_mae_ns']=mapped['curve_expected_mae_ns']
    return mapped

def stitch_one(run_dir,line_name,checkpoint,device,data_root_arg='',override_cfg_json='',prefer_curve_logits=True,return_details=False):
    run_dir=ROOT/run_dir

    if checkpoint=='final':
        ckpt_path=run_dir/'checkpoint_final.pt'
    elif checkpoint=='best':
        ckpt_path=run_dir/'checkpoint_best.pt'
    else:
        ckpt_path=run_dir/'checkpoint_last.pt'
    if not ckpt_path.exists(): ckpt_path=run_dir/'checkpoint_last.pt'
    ckpt=torch.load(ckpt_path,map_location=device,weights_only=False); cfg=ckpt['cfg']
    if override_cfg_json:
        cfg=dict(cfg)
        cfg.update(json.loads(override_cfg_json))
    data_root=resolve_data_root(data_root_arg,cfg)
    model=build_model(cfg).to(device); model.load_state_dict(ckpt['model']); model.eval()
    line=np.load(data_root/'lines'/f'{line_name}.npz')
    raw=line['raw_full_normalized'].astype(np.float32); H0,W0=raw.shape
    pred_sum=np.zeros((H0,W0),np.float32); weight_sum=np.zeros((H0,W0),np.float32)
    curve_sum=np.zeros((H0,W0),np.float32); curve_wsum=np.zeros((H0,W0),np.float32)
    center_sum=np.zeros((H0,W0),np.float32); center_wsum=np.zeros((H0,W0),np.float32)
    structured_sum=np.zeros((H0,W0),np.float32); structured_wsum=np.zeros((H0,W0),np.float32)
    uncertainty_sum=np.zeros((H0,W0),np.float32); uncertainty_wsum=np.zeros((H0,W0),np.float32)
    pres_sum=np.zeros((W0,),np.float32); pres_wsum=np.zeros((W0,),np.float32)
    no_pick_sum=np.zeros((W0,),np.float32); no_pick_wsum=np.zeros((W0,),np.float32)
    H,W=cfg['height_resize'],cfg['width_resize']
    rows=[r for r in csv.DictReader(open(data_root/'window_index.csv',encoding='utf-8')) if r['line']==line_name]
    for r in rows:
        s=int(r['start']); e=int(r['end'])+1
        x=torch.from_numpy(raw[:,s:e][None,None]).float().to(device)
        xrs=F.interpolate(x,(H,W),mode='bilinear',align_corners=False)
        xrs=compress_raw(xrs, cfg.get('input_log_scale',1e-3))
        xrs=normalize_raw_channel_4d(xrs,cfg)
        xrs=add_terrain_channels(xrs,line_name,s,e,cfg,data_root)
        altitude=None
        if bool(getattr(model, 'accepts_altitude', False)) and 'flight_height_agl_m' in line.files:
            values=np.asarray(line['flight_height_agl_m'][s:e],dtype=np.float32)
            if values.size==e-s and np.isfinite(values).all() and np.all(values>0):
                altitude=torch.from_numpy(values[None]).to(device=device,dtype=xrs.dtype)
                altitude=F.interpolate(altitude[:,None],size=W,mode='linear',align_corners=False)[:,0]
        chainage=None
        if bool(getattr(model, 'accepts_altitude', False)) and 'gnss_cumulative_distance_m' in line.files:
            values=np.asarray(line['gnss_cumulative_distance_m'][s:e],dtype=np.float32)
            if values.size==e-s and np.isfinite(values).all() and np.all(np.diff(values)>=0):
                chainage=torch.from_numpy(values[None]).to(device=device,dtype=xrs.dtype)
                chainage=F.interpolate(chainage[:,None],size=W,mode='linear',align_corners=False)[:,0]
        with torch.no_grad():
            out_obj=model(xrs,altitude=altitude,chainage_m=chainage) if bool(getattr(model, 'accepts_altitude', False)) else model(xrs)
            logits,pres_logits,center_logits=unpack_model_output(out_obj)
            curve_logits=getattr(out_obj,'curve_logits',None)
            path_marginals=getattr(out_obj,'path_marginals',None)
            uncertainty_logits=getattr(out_obj,'uncertainty_logits',None)
            no_pick_logits=getattr(out_obj,'no_pick_logits',None)
            mask_prob=torch.sigmoid(logits)
            curve_prob=torch.softmax(curve_logits,dim=2) if curve_logits is not None else None
            structured_prob=path_marginals if path_marginals is not None else None
            pp=torch.sigmoid(pres_logits)
            cp=torch.sigmoid(center_logits) if center_logits is not None else None
            no_pick_prob=torch.sigmoid(no_pick_logits).reshape(-1) if no_pick_logits is not None else None
        p0=F.interpolate(mask_prob,(H0,e-s),mode='bilinear',align_corners=False)[0,0].detach().cpu().numpy()
        pp0=F.interpolate(pp, size=e-s, mode='linear', align_corners=False)[0,0].detach().cpu().numpy()
        cp0=F.interpolate(cp,(H0,e-s),mode='bilinear',align_corners=False)[0,0].detach().cpu().numpy() if cp is not None else None
        curve0=F.interpolate(curve_prob,(H0,e-s),mode='bilinear',align_corners=False)[0,0].detach().cpu().numpy() if curve_prob is not None else None
        structured0=F.interpolate(structured_prob,(H0,e-s),mode='bilinear',align_corners=False)[0,0].detach().cpu().numpy() if structured_prob is not None else None
        uncertainty0=F.interpolate(uncertainty_logits,(H0,e-s),mode='bilinear',align_corners=False)[0,0].detach().cpu().numpy() if uncertainty_logits is not None else None
        ww=np.hanning(e-s).astype(np.float32)
        if ww.max()>0: ww=ww/ww.max()
        ww=0.15+0.85*ww
        w2=np.broadcast_to(ww[None,:],p0.shape).astype(np.float32)
        pred_sum[:,s:e]+=p0*w2; weight_sum[:,s:e]+=w2
        if curve0 is not None:
            curve_sum[:,s:e]+=curve0*w2; curve_wsum[:,s:e]+=w2
        if cp0 is not None:
            center_sum[:,s:e]+=cp0*w2; center_wsum[:,s:e]+=w2
        if structured0 is not None:
            structured_sum[:,s:e]+=structured0*w2; structured_wsum[:,s:e]+=w2
        if uncertainty0 is not None:
            uncertainty_sum[:,s:e]+=uncertainty0*w2; uncertainty_wsum[:,s:e]+=w2
        pres_sum[s:e]+=pp0*ww; pres_wsum[s:e]+=ww
        if no_pick_prob is not None:
            no_pick_sum[s:e]+=float(no_pick_prob[0])*ww; no_pick_wsum[s:e]+=ww
    pred=pred_sum/np.maximum(weight_sum,1e-6)
    curve_pred=curve_sum/np.maximum(curve_wsum,1e-6) if curve_wsum.max()>0 else None
    center_pred=center_sum/np.maximum(center_wsum,1e-6) if center_wsum.max()>0 else None
    details={
        'structured_path_prob': structured_sum/np.maximum(structured_wsum,1e-6) if structured_wsum.max()>0 else None,
        'uncertainty_log_variance': uncertainty_sum/np.maximum(uncertainty_wsum,1e-6) if uncertainty_wsum.max()>0 else None,
        'no_pick_prob': no_pick_sum/np.maximum(no_pick_wsum,1e-6) if no_pick_wsum.max()>0 else None,
        'altitude_conditioning_used': bool(getattr(model, 'accepts_altitude', False) and 'flight_height_agl_m' in line.files),
        'chainage_conditioning_used': bool(getattr(model, 'accepts_altitude', False) and 'gnss_cumulative_distance_m' in line.files),
    }
    result=(pred, pres_sum/np.maximum(pres_wsum,1e-6), center_pred, curve_pred, cfg, data_root)
    return (*result,details) if return_details else result


def write_centerline_csv(out,line_name,pred,pres_pred,gt,dt_ns, search_min_ns=320.0, search_max_ns=560.0, presence_thr=0.45, path_prob_thr=0.20, trace_offset=0, dp_max_jump=8, dp_smooth_weight=0.08, dp_breakable=False, dp_min_segment=16, distance_m=None, no_pick_prob=None, no_pick_thr=0.5, path_uncertainty=None):
    cgt,vgt=centerline(gt,1e-3)
    cmean,vmean=centerline(pred*(pred>0.15),1e-3)
    search_min=int(round(float(search_min_ns)/dt_ns)); search_max=int(round(float(search_max_ns)/dt_ns))
    if dp_breakable:
        cdp,vdp=breakable_dp_ridge_centerline(pred,pres_pred,presence_thr=presence_thr,path_prob_thr=path_prob_thr,min_segment=dp_min_segment,max_jump=int(dp_max_jump),smooth_weight=float(dp_smooth_weight),search_min_sample=search_min,search_max_sample=search_max)
    else:
        cdp,vdp=dp_ridge_centerline(pred, max_jump=int(dp_max_jump), smooth_weight=float(dp_smooth_weight), min_presence=(pres_pred>=presence_thr), search_min_sample=search_min, search_max_sample=search_max)
    H,W=pred.shape
    no_pick_prob=np.asarray(no_pick_prob,dtype=np.float32).reshape(-1) if no_pick_prob is not None else None
    path_uncertainty=np.asarray(path_uncertainty,dtype=np.float32).reshape(-1) if path_uncertainty is not None else None
    if no_pick_prob is not None and no_pick_prob.size!=W: raise ValueError('no_pick_prob must match prediction width')
    if path_uncertainty is not None and path_uncertainty.size!=W: raise ValueError('path_uncertainty must match prediction width')
    path_prob=np.full(W,np.nan,np.float32)
    final_valid=np.zeros(W,dtype=bool)
    pick_status=[]
    for i in range(W):
        if bool(vdp[i]) and np.isfinite(cdp[i]):
            yi=int(np.clip(round(float(cdp[i])),0,H-1)); path_prob[i]=pred[yi,i]
            final_valid[i]=(pres_pred[i]>=presence_thr) and (path_prob[i]>=path_prob_thr) and (no_pick_prob is None or no_pick_prob[i]<no_pick_thr)
        if final_valid[i]: pick_status.append('pick')
        elif pres_pred[i] < presence_thr: pick_status.append('reject_presence')
        elif no_pick_prob is not None and no_pick_prob[i]>=no_pick_thr: pick_status.append('reject_no_pick')
        else: pick_status.append('reject_low_path_prob')
    cdp_out=cdp.copy(); cdp_out[~final_valid]=np.nan
    if distance_m is None:
        distance_m=np.arange(trace_offset,trace_offset+W,dtype=np.float64)
    distance_m=np.asarray(distance_m,dtype=np.float64).reshape(-1)
    if distance_m.size!=W:
        raise ValueError(f'distance_m has {distance_m.size} values for prediction width {W}')
    with open(out/f'{line_name}_pred_centerline.csv','w',encoding='utf-8') as f:
        f.write('trace_idx,distance_m,mean_valid,mean_center_sample,mean_time_ns,dp_valid,dp_center_sample,dp_time_ns,dp_path_prob,path_uncertainty,no_pick_prob,pick_status,gt_valid,gt_center_sample,gt_time_ns,presence_prob\n')
        for i in range(W):
            mv=bool(vmean[i]); dv=bool(final_valid[i]); gv=bool(vgt[i])
            mcs='' if not mv else f'{float(cmean[i]):.4f}'
            mts='' if not mv else f'{float(cmean[i])*dt_ns:.4f}'
            dcs='' if not dv or not np.isfinite(cdp_out[i]) else f'{float(cdp_out[i]):.4f}'
            dts='' if not dv or not np.isfinite(cdp_out[i]) else f'{float(cdp_out[i])*dt_ns:.4f}'
            dpp='' if not np.isfinite(path_prob[i]) else f'{float(path_prob[i]):.6f}'
            pu='' if path_uncertainty is None or not np.isfinite(path_uncertainty[i]) else f'{float(path_uncertainty[i]):.6f}'
            npp='' if no_pick_prob is None or not np.isfinite(no_pick_prob[i]) else f'{float(no_pick_prob[i]):.6f}'
            gcs='' if not gv else f'{float(cgt[i]):.4f}'
            gts='' if not gv else f'{float(cgt[i])*dt_ns:.4f}'
            f.write(f'{i+trace_offset},{float(distance_m[i]):.6f},{int(mv)},{mcs},{mts},{int(dv)},{dcs},{dts},{dpp},{pu},{npp},{pick_status[i]},{int(gv)},{gcs},{gts},{float(pres_pred[i]):.6f}\n')
    return cmean,vmean,cdp_out,final_valid,cgt,vgt,path_prob

def _uncertainty_metrics(path_log_variance, cdp, vdp, cgt, vgt, dt_ns):
    result={'uncertainty_available':False}
    if path_log_variance is None or cdp is None or vdp is None or cgt is None or vgt is None:
        return result
    score=np.asarray(path_log_variance,dtype=np.float64).reshape(-1)
    valid=np.asarray(vdp,dtype=bool)&np.asarray(vgt,dtype=bool)&np.isfinite(cdp)&np.isfinite(cgt)&np.isfinite(score)
    if valid.sum()<3:
        return result
    # The model exports log variance. Exponentiating gives a monotonic,
    # positive uncertainty score while preserving a stable calibration scale.
    score=np.exp(np.clip(score,-8.0,5.0))
    error=np.abs(np.asarray(cdp,dtype=np.float64)-np.asarray(cgt,dtype=np.float64))*float(dt_ns)
    result['uncertainty_available']=True
    result['uncertainty_valid_trace_count']=int(valid.sum())
    # Avoid np.corrcoef here: on the project Windows NumPy build this tiny
    # operation can raise a native exception during otherwise pure-Python tests.
    # The explicit Pearson correlation of ordinal ranks is the same Spearman
    # proxy used previously, with a defined outcome for constant series.
    score_rank=np.argsort(np.argsort(score[valid])).astype(np.float64)
    error_rank=np.argsort(np.argsort(error[valid])).astype(np.float64)
    score_centered=score_rank-score_rank.mean()
    error_centered=error_rank-error_rank.mean()
    rank_denom=np.sqrt(np.dot(score_centered,score_centered)*np.dot(error_centered,error_centered))
    result['uncertainty_error_spearman_proxy']=float(np.dot(score_centered,error_centered)/rank_denom) if rank_denom>0 else float('nan')
    for coverage in (0.50,0.80,0.90):
        n=max(1,int(np.floor(valid.sum()*coverage)))
        keep=np.argsort(score[valid])[:n]
        result[f'uncertainty_risk_at_coverage_{int(coverage*100)}']=float(error[valid][keep].mean())
    return result


def write_metrics(out,line_name,mask_pred,path_pred,pres_pred,gt,status,label_w,dt_ns, curve_prob=None, cmean=None, vmean=None, cdp=None, vdp=None, cgt=None, vgt=None, path_prob=None, presence_thr=0.45, path_prob_thr=0.20, trace_start=0, trace_end=None, dp_max_jump=8, dp_smooth_weight=0.08, curve_source='mask_dp', path_source=None, dp_breakable=False, dp_min_segment=16, path_log_variance=None, no_pick_prob=None, no_pick_thr=0.5):
    """Write semantically separated mask, temporal-path, and presence metrics."""
    mask_pred=np.asarray(mask_pred,dtype=np.float32)
    path_pred=np.asarray(path_pred,dtype=np.float32)
    gt=np.asarray(gt,dtype=np.float32)
    status=np.asarray(status).reshape(-1)
    label_w=np.asarray(label_w,dtype=np.float32).reshape(-1)
    pres_pred=np.asarray(pres_pred,dtype=np.float32).reshape(-1)
    if mask_pred.shape!=gt.shape or path_pred.shape!=gt.shape:
        raise ValueError('mask_pred, path_pred, and gt must share HxW shape')
    if gt.shape[1]!=status.size or status.size!=label_w.size or status.size!=pres_pred.size:
        raise ValueError('trace-level arrays must match prediction width')
    source=path_source or curve_source
    w=0.10+np.broadcast_to(label_w[None,:],gt.shape)
    metrics={
        'trace_start':int(trace_start),
        'trace_end':int(trace_end if trace_end is not None else trace_start+mask_pred.shape[1]-1),
        'path_source':source,
        'mask_soft_dice_weighted':soft_dice(mask_pred,gt,w),
        'mask_weighted_bce':wbce(mask_pred,gt,w),
    }
    gb=gt>=0.1
    metrics['mask_gt_area_gt0p1']=float(gb.mean())
    for thr in (0.2,0.3,0.5):
        pb=mask_pred>=thr
        inter=np.logical_and(pb,gb).sum(); union=np.logical_or(pb,gb).sum()
        metrics[f'mask_iou_thr_{thr}']=float(inter/max(union,1))
        metrics[f'mask_pred_area_thr_{thr}']=float(pb.mean())
        metrics[f'mask_false_positive_area_thr_{thr}']=float(np.logical_and(pb,gt<0.05).mean())

    metrics.update(distribution_path_metrics(path_pred,gt,dt_ns,label_w))
    if curve_prob is not None:
        metrics.update(curve_distribution_metrics(curve_prob,gt,label_w,dt_ns))

    if cmean is None or vmean is None or cgt is None or vgt is None:
        cgt,vgt=centerline(gt,1e-3)
        path_dist,path_valid=normalise_trace_distribution(path_pred)
        cmean,vmean=centerline(path_dist,1e-8)
        vmean=vmean & path_valid
    both=vgt&vmean
    metrics['path_mean_center_valid_trace_count']=int(both.sum())
    metrics['path_mean_center_mae_sample']=float(np.nanmean(np.abs(cmean[both]-cgt[both]))) if both.any() else float('nan')
    metrics['path_mean_center_mae_ns']=metrics['path_mean_center_mae_sample']*float(dt_ns) if np.isfinite(metrics['path_mean_center_mae_sample']) else float('nan')
    metrics['path_mean_center_valid_ratio']=float(vmean.mean())
    if cdp is not None and vdp is not None:
        both2=vgt&vdp&np.isfinite(cdp)
        abs_err=np.abs(cdp[both2]-cgt[both2]) if both2.any() else np.asarray([],dtype=np.float32)
        metrics['path_dp_valid_trace_count']=int(both2.sum())
        metrics['path_dp_mae_sample']=float(np.nanmean(abs_err)) if abs_err.size else float('nan')
        metrics['path_dp_mae_ns']=metrics['path_dp_mae_sample']*float(dt_ns) if np.isfinite(metrics['path_dp_mae_sample']) else float('nan')
        metrics['path_dp_median_ae_ns']=float(np.nanmedian(abs_err)*dt_ns) if abs_err.size else float('nan')
        metrics['path_dp_p90_ae_ns']=float(np.nanpercentile(abs_err,90)*dt_ns) if abs_err.size else float('nan')
        metrics['path_dp_p95_ae_ns']=float(np.nanpercentile(abs_err,95)*dt_ns) if abs_err.size else float('nan')
        for tol_ns in (5.0,10.0,20.0):
            metrics[f'path_dp_hit_rate_le_{int(tol_ns)}ns']=float((abs_err*dt_ns<=tol_ns).mean()) if abs_err.size else float('nan')
        metrics['path_dp_valid_ratio']=float(vdp.mean())
        metrics['final_pick_rate']=float(vdp.mean())
        metrics['final_reject_rate']=float(1.0-vdp.mean())
        if path_prob is not None:
            metrics['path_dp_probability_mean_picked']=float(np.nanmean(path_prob[vdp])) if vdp.any() else float('nan')
        metrics['presence_threshold_for_pick']=float(presence_thr)
        metrics['path_probability_threshold_for_pick']=float(path_prob_thr)
        metrics['dp_max_jump']=int(dp_max_jump)
        metrics['dp_smooth_weight']=float(dp_smooth_weight)
        metrics['dp_breakable']=int(bool(dp_breakable))
        metrics['dp_min_segment']=int(dp_min_segment)
        metrics.update(_uncertainty_metrics(path_log_variance, cdp, vdp, cgt, vgt, dt_ns))
    else:
        metrics['uncertainty_available']=False
    if no_pick_prob is not None:
        no_pick_prob=np.asarray(no_pick_prob,dtype=np.float32).reshape(-1)
        if no_pick_prob.size != status.size:
            raise ValueError('no_pick_prob must match trace width')
        metrics['no_pick_probability_mean']=float(np.nanmean(no_pick_prob))
        metrics['no_pick_threshold']=float(no_pick_thr)
        metrics['no_pick_reject_rate']=float((no_pick_prob>=float(no_pick_thr)).mean())

    # status 0/1 are confirmed negatives/positives. status 2 is weak/unknown
    # and must not enter hard rejection metrics.
    known=(status==0)|(status==1)
    negative=status==0
    positive=status==1
    hard_pred=pres_pred>=float(presence_thr)
    metrics['presence_known_trace_count']=int(known.sum())
    metrics['presence_weak_or_unknown_trace_count']=int((~known).sum())
    metrics['presence_true_negative_trace_count']=int(negative.sum())
    metrics['presence_true_positive_trace_count']=int(positive.sum())
    if known.any():
        known_target=positive[known].astype(np.float32)
        metrics['presence_bce_confirmed']=wbce(pres_pred[known],known_target,0.25+label_w[known])
        metrics['presence_accuracy_confirmed']=float((hard_pred[known]==positive[known]).mean())
    else:
        metrics['presence_bce_confirmed']=float('nan')
        metrics['presence_accuracy_confirmed']=float('nan')
    metrics['presence_recall_confirmed_positive']=float(hard_pred[positive].mean()) if positive.any() else float('nan')
    metrics['presence_false_pick_rate_confirmed_negative']=float(hard_pred[negative].mean()) if negative.any() else float('nan')

    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    with (out/f'{line_name}_full_metrics.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f); writer.writerow(['metric','value'])
        for key,value in metrics.items(): writer.writerow([key,value])
    return metrics


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--line',default='Line9')
    ap.add_argument('--run-dirs',nargs='+',required=True)
    ap.add_argument('--out-dir',default='outputs/eval_full_line')
    ap.add_argument('--checkpoint',choices=['best','last','final'],default='best')
    ap.add_argument('--search-min-ns',type=float,default=320.0)
    ap.add_argument('--search-max-ns',type=float,default=560.0)
    ap.add_argument('--presence-thr',type=float,default=0.45)
    ap.add_argument('--path-prob-thr',type=float,default=0.20)
    ap.add_argument('--no-pick-thr',type=float,default=0.50,help='Reject structured-path picks when the no-pick probability reaches this threshold.')
    ap.add_argument('--dp-max-jump',type=int,default=8)
    ap.add_argument('--dp-smooth-weight',type=float,default=0.08)
    ap.add_argument('--dp-breakable',action='store_true',help='Run DP independently inside high-confidence segments instead of forcing one global ridge.')
    ap.add_argument('--dp-min-segment',type=int,default=16,help='Minimum contiguous trace count for breakable DP segments.')
    ap.add_argument('--threshold-json',default='')
    ap.add_argument('--data-root',default='',help='Override dataset root; defaults to checkpoint cfg data_root or data')
    ap.add_argument('--force-cpu',action='store_true',help='Run evaluation on CPU even when CUDA is available.')
    ap.add_argument('--no-plot',action='store_true',help='Write arrays, centerline CSV, and metrics without rendering PNG previews.')
    ap.add_argument('--trace-start',type=int,default=0)
    ap.add_argument('--trace-end',type=int,default=-1,help='Inclusive; -1 evaluates through the final trace')
    ap.add_argument('--center-fusion-weight',type=float,default=0.0,help='0 disables fusion; >0 fuses a normalised center response into the selected path distribution.')
    ap.add_argument('--allow-uncalibrated-center-fusion',action='store_true',help='Required when center-fusion-weight>0; records that the fusion is an explicit uncalibrated ablation.')
    ap.add_argument('--disable-curve-logits',action='store_true',help='For models with curve_logits, ignore them and evaluate the mask-derived path instead.')
    ap.add_argument('--write-legacy-aliases',action='store_true',help='Also emit deprecated *_softmask.npy aliases for old plotting scripts.')
    ap.add_argument('--override-cfg-json',default='',help='JSON object with evaluation-time cfg overrides, e.g. {"per_trace_robust_norm": true}.')
    ap.add_argument('--display-orientation',choices=['acquisition','profile'],default='profile',help='Plot/export view only. Metrics and canonical arrays always remain in CSV acquisition order.')
    ap.add_argument('--distance-axis',choices=['gnss','profile'],default='profile',help='Horizontal axis for plots. profile uses nominal engineering-profile chainage; gnss uses cumulative trajectory distance.')
    args=ap.parse_args()
    if args.threshold_json:
        tj=json.load(open(ROOT/args.threshold_json if not Path(args.threshold_json).is_absolute() else args.threshold_json,encoding='utf-8'))
        args.presence_thr=float(tj.get('presence_thr',args.presence_thr))
        args.path_prob_thr=float(tj.get('path_prob_thr',args.path_prob_thr))
        args.no_pick_thr=float(tj.get('no_pick_thr',args.no_pick_thr))
        args.search_min_ns=float(tj.get('search_min_ns',args.search_min_ns))
        args.search_max_ns=float(tj.get('search_max_ns',args.search_max_ns))
        args.dp_max_jump=int(tj.get('dp_max_jump',args.dp_max_jump))
        args.dp_smooth_weight=float(tj.get('dp_smooth_weight',args.dp_smooth_weight))
    torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    device=torch.device('cpu' if args.force_cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    preds=[]; presses=[]; centers=[]; curves=[]; structured_paths=[]; uncertainties=[]; no_picks=[]; data_roots=[]; altitude_conditioning=[]; run_cfgs=[]
    for rd in args.run_dirs:
        print(f'评估 {args.line}: {rd}',flush=True)
        p,pp,cp,curve,cfg,data_root,details=stitch_one(Path(rd),args.line,args.checkpoint,device,args.data_root,args.override_cfg_json,prefer_curve_logits=not args.disable_curve_logits,return_details=True); preds.append(p); presses.append(pp); centers.append(cp); curves.append(curve); structured_paths.append(details['structured_path_prob']); uncertainties.append(details['uncertainty_log_variance']); no_picks.append(details['no_pick_prob']); altitude_conditioning.append(details['altitude_conditioning_used']); data_roots.append(data_root); run_cfgs.append(cfg)
    mask_pred=np.mean(preds,axis=0).astype(np.float32); pres_pred=np.mean(presses,axis=0).astype(np.float32)
    center_pred=np.mean([cp for cp in centers if cp is not None],axis=0).astype(np.float32) if any(cp is not None for cp in centers) else None
    curve_pred=np.mean([cv for cv in curves if cv is not None],axis=0).astype(np.float32) if any(cv is not None for cv in curves) else None
    structured_path_pred=np.mean([sp for sp in structured_paths if sp is not None],axis=0).astype(np.float32) if any(sp is not None for sp in structured_paths) else None
    uncertainty_pred=np.mean([up for up in uncertainties if up is not None],axis=0).astype(np.float32) if any(up is not None for up in uncertainties) else None
    no_pick_pred=np.mean([npred for npred in no_picks if npred is not None],axis=0).astype(np.float32) if any(npred is not None for npred in no_picks) else None
    real_nopick_reporting=real_nopick_reporting_policy(run_cfgs)
    # V15 is an interface-following survey, not a measured rejection set. A
    # formal conditional-path run may retain the head for controlled-simulation
    # training, but cannot apply it or report it on measured full lines.
    if real_nopick_reporting=='forbidden':
        no_pick_pred=None
    if curve_pred is not None:
        curve_pred=normalise_time_distribution(curve_pred)
    used_structured_path = structured_path_pred is not None and (not args.disable_curve_logits)
    used_curve_logits = (curve_pred is not None) and (not args.disable_curve_logits) and not used_structured_path
    data_root=data_roots[0] if data_roots else resolve_data_root(args.data_root)
    if any(dr!=data_root for dr in data_roots):
        raise ValueError('All run dirs must resolve to the same data root for one evaluation.')
    print('DATA_ROOT',str(data_root),flush=True)
    line=np.load(data_root/'lines'/f'{args.line}.npz')
    raw=line['raw_full_normalized'].astype(np.float32); gt=line['soft_mask_train'].astype(np.float32); label_w=line['label_weight'].astype(np.float32); status=line['status_code'].astype(np.int16)
    gnss_distance_full=(line['gnss_cumulative_distance_m'].astype(np.float64) if 'gnss_cumulative_distance_m' in line.files else np.arange(raw.shape[1],dtype=np.float64))
    profile_distance_full=(line['profile_chainage_m'].astype(np.float64) if 'profile_chainage_m' in line.files else (line['declared_trace_distance_m'].astype(np.float64) if 'declared_trace_distance_m' in line.files else np.arange(raw.shape[1],dtype=np.float64)))
    distance_full=profile_distance_full if args.distance_axis=='profile' else gnss_distance_full
    trace_start=max(0,args.trace_start); trace_end=raw.shape[1]-1 if args.trace_end<0 else min(args.trace_end,raw.shape[1]-1)
    if trace_end<trace_start: raise ValueError('trace-end must be >= trace-start')
    sl=slice(trace_start,trace_end+1)
    raw=raw[:,sl]; gt=gt[:,sl]; label_w=label_w[sl]; status=status[sl]; mask_pred=mask_pred[:,sl]; pres_pred=pres_pred[sl]; distance_m=distance_full[sl]
    if curve_pred is not None:
        curve_pred=curve_pred[:,sl]
    if structured_path_pred is not None:
        structured_path_pred=structured_path_pred[:,sl]
    if uncertainty_pred is not None:
        uncertainty_pred=uncertainty_pred[:,sl]
    if no_pick_pred is not None:
        no_pick_pred=no_pick_pred[sl]
    if center_pred is not None:
        center_pred=center_pred[:,sl]
    fusion_w=max(0.0,min(1.0,float(args.center_fusion_weight)))
    if fusion_w>0 and not args.allow_uncalibrated_center_fusion:
        raise ValueError('center fusion is uncalibrated; pass --allow-uncalibrated-center-fusion for an explicit ablation')
    path_pred=structured_path_pred.copy() if used_structured_path else (curve_pred.copy() if used_curve_logits else mask_pred.copy())
    curve_source='aeropath_soft_dp_marginals' if used_structured_path else ('curve_distribution_dp' if used_curve_logits else 'mask_dp')
    if center_pred is not None and fusion_w>0:
        center_dist=normalise_time_distribution(center_pred)
        path_pred=normalise_time_distribution((1.0-fusion_w)*path_pred+fusion_w*center_dist)
        curve_source=f'{curve_source[:-3]}_center_fusion_{fusion_w:.2f}_dp'
    eval_name=args.line if trace_start==0 and trace_end==line['raw_full_normalized'].shape[1]-1 else f'{args.line}_holdout_tr{trace_start}_{trace_end}'
    out=ROOT/args.out_dir; out.mkdir(parents=True,exist_ok=True)
    np.save(out/f'{eval_name}_mask_prob.npy',mask_pred)
    np.save(out/f'{eval_name}_presence_prob.npy',pres_pred)
    if curve_pred is not None:
        np.save(out/f'{eval_name}_curve_prob.npy',curve_pred)
    if structured_path_pred is not None:
        np.save(out/f'{eval_name}_structured_path_prob.npy',structured_path_pred)
    if uncertainty_pred is not None:
        np.save(out/f'{eval_name}_path_log_variance.npy',uncertainty_pred)
    if no_pick_pred is not None:
        np.save(out/f'{eval_name}_no_pick_prob.npy',no_pick_pred)
    if center_pred is not None:
        np.save(out/f'{eval_name}_center_response_prob.npy',center_pred)
    np.save(out/f'{eval_name}_path_prob_image.npy',path_pred)
    np.save(out/f'{eval_name}_gnss_distance_m.npy',gnss_distance_full[sl].astype(np.float64))
    np.save(out/f'{eval_name}_profile_chainage_m.npy',profile_distance_full[sl].astype(np.float64))
    if args.write_legacy_aliases:
        np.save(out/f'{eval_name}_pred_softmask.npy',mask_pred)
        if center_pred is not None:
            np.save(out/f'{eval_name}_center_softmask.npy',center_pred)
        np.save(out/f'{eval_name}_path_softmask.npy',path_pred)
    artifact_contract={
        'mask_prob_semantics':'sigmoid pixel-mask probability; only this artifact is used for Dice/IoU/BCE',
        'curve_prob_semantics':'time-normalised P(t|trace)' if curve_pred is not None else 'not available',
        'path_prob_image_semantics':curve_source,
        'structured_path_prob_semantics':'soft-DP path marginal P(t|trace)' if structured_path_pred is not None else 'not available',
        'path_log_variance_semantics':'pixelwise AeroPath log variance; per-trace summaries are weighted by structured path marginals' if uncertainty_pred is not None else 'not available',
        'no_pick_prob_semantics':'stitched local no-interface probability' if no_pick_pred is not None else 'not available',
        'real_nopick_metric_reporting':real_nopick_reporting,
        'altitude_conditioning_used':bool(any(altitude_conditioning)),
        'center_response_prob_semantics':'sigmoid center response (not a segmentation mask)' if center_pred is not None else 'not available',
        'legacy_aliases_written':bool(args.write_legacy_aliases),
        'center_fusion_weight':fusion_w,
        'uncalibrated_center_fusion':bool(fusion_w>0),
        'canonical_prediction_order':'acquisition_csv',
        'display_orientation':args.display_orientation,
        'display_distance_axis':args.distance_axis,
        'profile_display_flip':profile_display_flip_or_false(args.line),
        'spatial_axis':'profile_chainage_m' if args.distance_axis=='profile' else ('gnss_cumulative_distance_m' if 'gnss_cumulative_distance_m' in line.files else 'trace_index_fallback'),
    }
    json.dump(artifact_contract,open(out/f'{eval_name}_artifact_contract.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
    path_log_variance=(uncertainty_pred*path_pred).sum(axis=0) if uncertainty_pred is not None else None
    path_uncertainty=np.exp(np.clip(path_log_variance,-8.0,5.0)) if path_log_variance is not None else None
    cmean,vmean,cdp,vdp,cgt,vgt,path_prob=write_centerline_csv(out,eval_name,path_pred,pres_pred,gt,float(line['dt_ns']),args.search_min_ns,args.search_max_ns,args.presence_thr,args.path_prob_thr,trace_start,args.dp_max_jump,args.dp_smooth_weight,args.dp_breakable,args.dp_min_segment,distance_m=distance_m,no_pick_prob=no_pick_pred,no_pick_thr=args.no_pick_thr,path_uncertainty=path_uncertainty)
    write_metrics(out,eval_name,mask_pred,path_pred,pres_pred,gt,status,label_w,float(line['dt_ns']),curve_prob=curve_pred,cmean=cmean,vmean=vmean,cdp=cdp,vdp=vdp,cgt=cgt,vgt=vgt,path_prob=path_prob,presence_thr=args.presence_thr,path_prob_thr=args.path_prob_thr,trace_start=trace_start,trace_end=trace_end,dp_max_jump=args.dp_max_jump,dp_smooth_weight=args.dp_smooth_weight,curve_source=curve_source,dp_breakable=args.dp_breakable,dp_min_segment=args.dp_min_segment,path_log_variance=path_log_variance,no_pick_prob=no_pick_pred,no_pick_thr=args.no_pick_thr)
    display_order=(profile_index_order(mask_pred.shape[1],args.line) if profile_display_flip_or_false(args.line) else np.arange(mask_pred.shape[1],dtype=np.int64)) if args.display_orientation=='profile' else np.arange(mask_pred.shape[1],dtype=np.int64)
    display_base_full=profile_distance_full if args.distance_axis=='profile' else gnss_distance_full
    display_base_subset=display_base_full[sl]
    if args.display_orientation=='profile' and profile_display_flip_or_false(args.line):
        display_distance=display_base_full[-1]-display_base_subset[::-1]
    else:
        display_distance=display_base_subset.copy()
    with open(out/f'{eval_name}_{args.display_orientation}_display_centerline.csv','w',encoding='utf-8') as f:
        f.write('display_idx,source_trace_idx,distance_m,dp_valid,dp_center_sample,dp_time_ns,presence_prob\n')
        for display_idx,local_idx in enumerate(display_order):
            source_idx=trace_start+int(local_idx)
            valid=bool(vdp[local_idx]) and np.isfinite(cdp[local_idx])
            center='' if not valid else f'{float(cdp[local_idx]):.4f}'
            time='' if not valid else f'{float(cdp[local_idx])*float(line["dt_ns"]):.4f}'
            f.write(f'{display_idx},{source_idx},{float(display_distance[display_idx]):.6f},{int(valid)},{center},{time},{float(pres_pred[local_idx]):.6f}\n')
    if args.no_plot:
        print(out/f'{eval_name}_full_metrics.csv')
        return
    fig,ax=plt.subplots(1,5,figsize=(20,4.5))
    raw_view=align_array_for_display(raw,args.line,axis=-1,orientation=args.display_orientation)
    gt_view=align_array_for_display(gt,args.line,axis=-1,orientation=args.display_orientation)
    path_view=align_array_for_display(path_pred,args.line,axis=-1,orientation=args.display_orientation)
    pres_view=align_array_for_display(pres_pred,args.line,axis=-1,orientation=args.display_orientation)
    cdp_view=align_array_for_display(cdp,args.line,axis=-1,orientation=args.display_orientation)
    v=np.nanpercentile(np.abs(raw_view),98)
    extent=(float(display_distance[0]),float(display_distance[-1]),mask_pred.shape[0],0); xcoords=display_distance
    x_label='剖面里程 / m' if args.distance_axis=='profile' else 'GNSS累计距离 / m'
    base_title=args.line if eval_name==args.line else f'{args.line} holdout {trace_start}-{trace_end}'
    view_title=f'{base_title}（{args.display_orientation}视图）'
    ax[0].imshow(raw_view,aspect='auto',origin='upper',extent=extent,cmap='gray',vmin=-v,vmax=v); ax[0].set_title(f'{view_title} 输入：原始 raw',fontproperties=FONT)
    ax[1].imshow(gt_view,aspect='auto',origin='upper',extent=extent,cmap='viridis',vmin=0,vmax=max(0.6,float(gt_view.max()))); ax[1].set_title(f'{view_title} 标签：响应带',fontproperties=FONT)
    ax[2].imshow(path_view,aspect='auto',origin='upper',extent=extent,cmap='viridis',vmin=0,vmax=max(0.6,float(path_view.max()))); ax[2].set_title(f'{view_title} 路径概率',fontproperties=FONT)
    ax[3].plot(xcoords,pres_view); ax[3].set_ylim(-0.05,1.05); ax[3].set_title('presence：每道可拾取概率',fontproperties=FONT); ax[3].set_xlabel(x_label,fontproperties=FONT); ax[3].set_ylabel('概率',fontproperties=FONT)
    ax[4].imshow(raw_view,aspect='auto',origin='upper',extent=extent,cmap='gray',vmin=-v,vmax=v); ax[4].imshow(path_view,aspect='auto',origin='upper',extent=extent,cmap='magma',alpha=np.clip(path_view*0.85,0,0.65)); ax[4].plot(xcoords, cdp_view, linewidth=1.0); ax[4].set_title(f'{view_title} 叠加图 + DP中心线',fontproperties=FONT)
    for a in [ax[0],ax[1],ax[2],ax[4]]: a.set_xlabel(x_label,fontproperties=FONT); a.set_ylabel('采样点 / sample',fontproperties=FONT)
    fig.subplots_adjust(left=0.035, right=0.995, bottom=0.12, top=0.86, wspace=0.24)
    fig.savefig(out/f'{eval_name}_stitched_prediction_cn.png',dpi=160); plt.close(fig)
    print(out/f'{eval_name}_stitched_prediction_cn.png'); print(out/f'{eval_name}_full_metrics.csv')
if __name__=='__main__': main()
