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
FONT=get_chinese_font()

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

def stitch_one(run_dir,line_name,checkpoint,device,data_root_arg='',override_cfg_json=''):
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
    center_sum=np.zeros((H0,W0),np.float32); center_wsum=np.zeros((H0,W0),np.float32)
    pres_sum=np.zeros((W0,),np.float32); pres_wsum=np.zeros((W0,),np.float32)
    H,W=cfg['height_resize'],cfg['width_resize']
    rows=[r for r in csv.DictReader(open(data_root/'window_index.csv',encoding='utf-8')) if r['line']==line_name]
    for r in rows:
        s=int(r['start']); e=int(r['end'])+1
        x=torch.from_numpy(raw[:,s:e][None,None]).float().to(device)
        xrs=F.interpolate(x,(H,W),mode='bilinear',align_corners=False)
        xrs=compress_raw(xrs, cfg.get('input_log_scale',1e-3))
        xrs=normalize_raw_channel_4d(xrs,cfg)
        xrs=add_terrain_channels(xrs,line_name,s,e,cfg,data_root)
        with torch.no_grad():
            logits,pres_logits,center_logits=unpack_model_output(model(xrs)); p=torch.sigmoid(logits); pp=torch.sigmoid(pres_logits)
            cp=torch.sigmoid(center_logits) if center_logits is not None else None
        p0=F.interpolate(p,(H0,e-s),mode='bilinear',align_corners=False)[0,0].detach().cpu().numpy()
        pp0=F.interpolate(pp, size=e-s, mode='linear', align_corners=False)[0,0].detach().cpu().numpy()
        cp0=F.interpolate(cp,(H0,e-s),mode='bilinear',align_corners=False)[0,0].detach().cpu().numpy() if cp is not None else None
        ww=np.hanning(e-s).astype(np.float32)
        if ww.max()>0: ww=ww/ww.max()
        ww=0.15+0.85*ww
        w2=np.broadcast_to(ww[None,:],p0.shape).astype(np.float32)
        pred_sum[:,s:e]+=p0*w2; weight_sum[:,s:e]+=w2
        if cp0 is not None:
            center_sum[:,s:e]+=cp0*w2; center_wsum[:,s:e]+=w2
        pres_sum[s:e]+=pp0*ww; pres_wsum[s:e]+=ww
    pred=pred_sum/np.maximum(weight_sum,1e-6)
    center_pred=center_sum/np.maximum(center_wsum,1e-6) if center_wsum.max()>0 else None
    return pred, pres_sum/np.maximum(pres_wsum,1e-6), center_pred, cfg, data_root


def write_centerline_csv(out,line_name,pred,pres_pred,gt,dt_ns, search_min_ns=320.0, search_max_ns=560.0, presence_thr=0.45, path_prob_thr=0.20, trace_offset=0, dp_max_jump=8, dp_smooth_weight=0.08, dp_breakable=False, dp_min_segment=16):
    cgt,vgt=centerline(gt,1e-3)
    cmean,vmean=centerline(pred*(pred>0.15),1e-3)
    search_min=int(round(float(search_min_ns)/dt_ns)); search_max=int(round(float(search_max_ns)/dt_ns))
    if dp_breakable:
        cdp,vdp=breakable_dp_ridge_centerline(pred,pres_pred,presence_thr=presence_thr,path_prob_thr=path_prob_thr,min_segment=dp_min_segment,max_jump=int(dp_max_jump),smooth_weight=float(dp_smooth_weight),search_min_sample=search_min,search_max_sample=search_max)
    else:
        cdp,vdp=dp_ridge_centerline(pred, max_jump=int(dp_max_jump), smooth_weight=float(dp_smooth_weight), min_presence=(pres_pred>=presence_thr), search_min_sample=search_min, search_max_sample=search_max)
    H,W=pred.shape
    path_prob=np.full(W,np.nan,np.float32)
    final_valid=np.zeros(W,dtype=bool)
    pick_status=[]
    for i in range(W):
        if bool(vdp[i]) and np.isfinite(cdp[i]):
            yi=int(np.clip(round(float(cdp[i])),0,H-1)); path_prob[i]=pred[yi,i]
            final_valid[i]=(pres_pred[i]>=presence_thr) and (path_prob[i]>=path_prob_thr)
        if final_valid[i]: pick_status.append('pick')
        elif pres_pred[i] < presence_thr: pick_status.append('reject_presence')
        else: pick_status.append('reject_low_path_prob')
    cdp_out=cdp.copy(); cdp_out[~final_valid]=np.nan
    with open(out/f'{line_name}_pred_centerline.csv','w',encoding='utf-8') as f:
        f.write('trace_idx,mean_valid,mean_center_sample,mean_time_ns,dp_valid,dp_center_sample,dp_time_ns,dp_path_prob,pick_status,gt_valid,gt_center_sample,gt_time_ns,presence_prob\n')
        for i in range(W):
            mv=bool(vmean[i]); dv=bool(final_valid[i]); gv=bool(vgt[i])
            mcs='' if not mv else f'{float(cmean[i]):.4f}'
            mts='' if not mv else f'{float(cmean[i])*dt_ns:.4f}'
            dcs='' if not dv or not np.isfinite(cdp_out[i]) else f'{float(cdp_out[i]):.4f}'
            dts='' if not dv or not np.isfinite(cdp_out[i]) else f'{float(cdp_out[i])*dt_ns:.4f}'
            dpp='' if not np.isfinite(path_prob[i]) else f'{float(path_prob[i]):.6f}'
            gcs='' if not gv else f'{float(cgt[i]):.4f}'
            gts='' if not gv else f'{float(cgt[i])*dt_ns:.4f}'
            f.write(f'{i+trace_offset},{int(mv)},{mcs},{mts},{int(dv)},{dcs},{dts},{dpp},{pick_status[i]},{int(gv)},{gcs},{gts},{float(pres_pred[i]):.6f}\n')
    return cmean,vmean,cdp_out,final_valid,cgt,vgt,path_prob

def write_metrics(out,line_name,pred,pres_pred,gt,status,label_w,dt_ns, cmean=None, vmean=None, cdp=None, vdp=None, cgt=None, vgt=None, path_prob=None, presence_thr=0.45, path_prob_thr=0.20, trace_start=0, trace_end=None, dp_max_jump=8, dp_smooth_weight=0.08, curve_source='mask_dp', dp_breakable=False, dp_min_segment=16):
    w=0.10+np.broadcast_to(label_w[None,:],gt.shape)
    metrics={'trace_start':int(trace_start),'trace_end':int(trace_end if trace_end is not None else trace_start+pred.shape[1]-1),'curve_source':curve_source,'soft_dice_weighted':soft_dice(pred,gt,w),'weighted_bce':wbce(pred,gt,w)}
    gb=gt>=0.1; metrics['gt_area_gt0p1']=float(gb.mean())
    for thr in [0.2,0.3,0.5]:
        pb=pred>=thr; inter=np.logical_and(pb,gb).sum(); union=np.logical_or(pb,gb).sum()
        metrics[f'iou_thr_{thr}']=float(inter/max(union,1)); metrics[f'pred_area_thr_{thr}']=float(pb.mean()); metrics[f'false_positive_area_thr_{thr}']=float(np.logical_and(pb,gt<0.05).mean())
    if cmean is None or vmean is None or cgt is None or vgt is None:
        cgt,vgt=centerline(gt,1e-3); cmean,vmean=centerline(pred*(pred>0.15),1e-3)
    both=vgt&vmean
    metrics['mean_center_valid_trace_count']=int(both.sum())
    metrics['mean_center_mae_sample']=float(np.nanmean(np.abs(cmean[both]-cgt[both]))) if both.any() else float('nan')
    metrics['mean_center_mae_ns']=metrics['mean_center_mae_sample']*float(dt_ns) if np.isfinite(metrics['mean_center_mae_sample']) else float('nan')
    metrics['mean_center_valid_ratio']=float(vmean.mean())
    if cdp is not None and vdp is not None:
        both2=vgt&vdp&np.isfinite(cdp)
        metrics['dp_center_valid_trace_count']=int(both2.sum())
        metrics['dp_center_mae_sample']=float(np.nanmean(np.abs(cdp[both2]-cgt[both2]))) if both2.any() else float('nan')
        metrics['dp_center_mae_ns']=metrics['dp_center_mae_sample']*float(dt_ns) if np.isfinite(metrics['dp_center_mae_sample']) else float('nan')
        metrics['dp_center_valid_ratio']=float(vdp.mean())
        metrics['final_pick_rate']=float(vdp.mean())
        metrics['final_reject_rate']=float(1.0-vdp.mean())
        if path_prob is not None:
            metrics['dp_path_prob_mean_picked']=float(np.nanmean(path_prob[vdp])) if vdp.any() else float('nan')
        metrics['presence_threshold_for_pick']=float(presence_thr)
        metrics['path_probability_threshold_for_pick']=float(path_prob_thr)
        metrics['dp_max_jump']=int(dp_max_jump)
        metrics['dp_smooth_weight']=float(dp_smooth_weight)
        metrics['dp_breakable']=int(bool(dp_breakable))
        metrics['dp_min_segment']=int(dp_min_segment)
    # Presence metrics: status 1=present, 2=weak_visible, 0=no_pick. Weak target = 0.5.
    pres_target=(status==1).astype(np.float32)+(status==2).astype(np.float32)*0.5
    pres_w=0.25+label_w
    metrics['presence_soft_bce']=wbce(pres_pred,pres_target,pres_w)
    hard_target=(status>0).astype(np.float32); hard_pred=(pres_pred>=0.5).astype(np.float32)
    if hard_target.size:
        metrics['presence_hard_accuracy']=float((hard_pred==hard_target).mean())
        pos=hard_target>0.5; neg=~pos
        metrics['presence_recall_pickable']=float((hard_pred[pos]==1).mean()) if pos.any() else float('nan')
        metrics['presence_false_pick_rate_no_pick']=float((hard_pred[neg]==1).mean()) if neg.any() else float('nan')
    with open(out/f'{line_name}_full_metrics.csv','w',encoding='utf-8') as f:
        f.write('metric,value\n')
        for k,v in metrics.items(): f.write(f'{k},{v}\n')

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
    ap.add_argument('--center-fusion-weight',type=float,default=0.0,help='0 keeps legacy mask-DP; >0 fuses center head probability into the DP path image.')
    ap.add_argument('--override-cfg-json',default='',help='JSON object with evaluation-time cfg overrides, e.g. {"per_trace_robust_norm": true}.')
    args=ap.parse_args()
    if args.threshold_json:
        tj=json.load(open(ROOT/args.threshold_json if not Path(args.threshold_json).is_absolute() else args.threshold_json,encoding='utf-8'))
        args.presence_thr=float(tj.get('presence_thr',args.presence_thr))
        args.path_prob_thr=float(tj.get('path_prob_thr',args.path_prob_thr))
        args.search_min_ns=float(tj.get('search_min_ns',args.search_min_ns))
        args.search_max_ns=float(tj.get('search_max_ns',args.search_max_ns))
        args.dp_max_jump=int(tj.get('dp_max_jump',args.dp_max_jump))
        args.dp_smooth_weight=float(tj.get('dp_smooth_weight',args.dp_smooth_weight))
    torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    device=torch.device('cpu' if args.force_cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    preds=[]; presses=[]; centers=[]; data_roots=[]
    for rd in args.run_dirs:
        print(f'评估 {args.line}: {rd}',flush=True)
        p,pp,cp,cfg,data_root=stitch_one(Path(rd),args.line,args.checkpoint,device,args.data_root,args.override_cfg_json); preds.append(p); presses.append(pp); centers.append(cp); data_roots.append(data_root)
    pred=np.mean(preds,axis=0).astype(np.float32); pres_pred=np.mean(presses,axis=0).astype(np.float32)
    center_pred=np.mean([cp for cp in centers if cp is not None],axis=0).astype(np.float32) if any(cp is not None for cp in centers) else None
    data_root=data_roots[0] if data_roots else resolve_data_root(args.data_root)
    if any(dr!=data_root for dr in data_roots):
        raise ValueError('All run dirs must resolve to the same data root for one evaluation.')
    print('DATA_ROOT',str(data_root),flush=True)
    line=np.load(data_root/'lines'/f'{args.line}.npz')
    raw=line['raw_full_normalized'].astype(np.float32); gt=line['soft_mask_train'].astype(np.float32); label_w=line['label_weight'].astype(np.float32); status=line['status_code'].astype(np.int16)
    trace_start=max(0,args.trace_start); trace_end=raw.shape[1]-1 if args.trace_end<0 else min(args.trace_end,raw.shape[1]-1)
    if trace_end<trace_start: raise ValueError('trace-end must be >= trace-start')
    sl=slice(trace_start,trace_end+1)
    raw=raw[:,sl]; gt=gt[:,sl]; label_w=label_w[sl]; status=status[sl]; pred=pred[:,sl]; pres_pred=pres_pred[sl]
    if center_pred is not None:
        center_pred=center_pred[:,sl]
    fusion_w=max(0.0,min(1.0,float(args.center_fusion_weight)))
    path_pred=pred
    curve_source='mask_dp'
    if center_pred is not None and fusion_w>0:
        path_pred=((1.0-fusion_w)*pred+fusion_w*center_pred).astype(np.float32)
        curve_source=f'mask_center_fusion_{fusion_w:.2f}_dp'
    eval_name=args.line if trace_start==0 and trace_end==line['raw_full_normalized'].shape[1]-1 else f'{args.line}_holdout_tr{trace_start}_{trace_end}'
    out=ROOT/args.out_dir; out.mkdir(parents=True,exist_ok=True)
    np.save(out/f'{eval_name}_pred_softmask.npy',pred); np.save(out/f'{eval_name}_presence_prob.npy',pres_pred)
    if center_pred is not None:
        np.save(out/f'{eval_name}_center_softmask.npy',center_pred)
    if path_pred is not pred:
        np.save(out/f'{eval_name}_path_softmask.npy',path_pred)
    cmean,vmean,cdp,vdp,cgt,vgt,path_prob=write_centerline_csv(out,eval_name,path_pred,pres_pred,gt,float(line['dt_ns']),args.search_min_ns,args.search_max_ns,args.presence_thr,args.path_prob_thr,trace_start,args.dp_max_jump,args.dp_smooth_weight,args.dp_breakable,args.dp_min_segment)
    write_metrics(out,eval_name,pred,pres_pred,gt,status,label_w,float(line['dt_ns']),cmean,vmean,cdp,vdp,cgt,vgt,path_prob,args.presence_thr,args.path_prob_thr,trace_start,trace_end,args.dp_max_jump,args.dp_smooth_weight,curve_source,args.dp_breakable,args.dp_min_segment)
    if args.no_plot:
        print(out/f'{eval_name}_full_metrics.csv')
        return
    fig,ax=plt.subplots(1,5,figsize=(20,4.5))
    v=np.nanpercentile(np.abs(raw),98)
    extent=(trace_start,trace_end,pred.shape[0],0); xcoords=np.arange(trace_start,trace_end+1)
    view_title=args.line if eval_name==args.line else f'{args.line} holdout {trace_start}-{trace_end}'
    ax[0].imshow(raw,aspect='auto',origin='upper',extent=extent,cmap='gray',vmin=-v,vmax=v); ax[0].set_title(f'{view_title} 输入：原始 raw',fontproperties=FONT)
    ax[1].imshow(gt,aspect='auto',origin='upper',extent=extent,cmap='viridis',vmin=0,vmax=max(0.6,float(gt.max()))); ax[1].set_title(f'{view_title} 标签：响应带',fontproperties=FONT)
    ax[2].imshow(path_pred,aspect='auto',origin='upper',extent=extent,cmap='viridis',vmin=0,vmax=max(0.6,float(path_pred.max()))); ax[2].set_title(f'{view_title} 路径概率',fontproperties=FONT)
    ax[3].plot(xcoords,pres_pred); ax[3].set_ylim(-0.05,1.05); ax[3].set_title('presence：每道可拾取概率',fontproperties=FONT); ax[3].set_xlabel('道号 / trace',fontproperties=FONT); ax[3].set_ylabel('概率',fontproperties=FONT)
    ax[4].imshow(raw,aspect='auto',origin='upper',extent=extent,cmap='gray',vmin=-v,vmax=v); ax[4].imshow(path_pred,aspect='auto',origin='upper',extent=extent,cmap='magma',alpha=np.clip(path_pred*0.85,0,0.65)); ax[4].plot(xcoords, cdp, linewidth=1.0); ax[4].set_title(f'{view_title} 叠加图 + DP中心线',fontproperties=FONT)
    for a in [ax[0],ax[1],ax[2],ax[4]]: a.set_xlabel('道号 / trace',fontproperties=FONT); a.set_ylabel('采样点 / sample',fontproperties=FONT)
    fig.subplots_adjust(left=0.035, right=0.995, bottom=0.12, top=0.86, wspace=0.24)
    fig.savefig(out/f'{eval_name}_stitched_prediction_cn.png',dpi=160); plt.close(fig)
    print(out/f'{eval_name}_stitched_prediction_cn.png'); print(out/f'{eval_name}_full_metrics.csv')
if __name__=='__main__': main()
