from pathlib import Path
import json,csv,random,sys,platform
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pgdacsnet.model_raw_unet import build_model, compress_raw
from pgdacsnet.font_utils import get_chinese_font
FONT=get_chinese_font()

def add_terrain_channels(x, row, cfg, data_root):
    feature_names=cfg.get('terrain_feature_names', [])
    if not cfg.get('use_terrain_features', False) or not feature_names:
        return x
    line=row['line']; s=int(row['start']); e=int(row['end'])+1
    feature_dir=cfg.get('terrain_feature_dir','terrain_features')
    fpath=data_root/feature_dir/f'{line}_terrain_features.npz'
    z=np.load(fpath,allow_pickle=False)
    names=[str(v) for v in z['feature_names']]
    idx=[names.index(name) for name in feature_names]
    feat=torch.from_numpy(z['features'][idx,s:e]).float()
    H,W=x.shape[-2],x.shape[-1]
    feat=F.interpolate(feat[None,:,None,:],(H,W),mode='bilinear',align_corners=False)[0]
    return torch.cat([x,feat],dim=0)

def normalize_raw_channel_3d(x, cfg):
    if not cfg.get('per_trace_robust_norm', False):
        return x
    clip=float(cfg.get('per_trace_robust_clip',6.0))
    eps=float(cfg.get('per_trace_robust_eps',1e-4))
    raw=x[:1]
    med=raw.median(dim=1,keepdim=True).values
    mad=(raw-med).abs().median(dim=1,keepdim=True).values
    norm=torch.clamp((raw-med)/(1.4826*mad+eps),-clip,clip)/clip
    x=x.clone()
    x[:1]=norm
    return x

def resolve_data_root(cfg):
    data_root=Path(cfg.get('data_root','data'))
    return data_root if data_root.is_absolute() else ROOT/data_root

class DS(Dataset):
    def __init__(self,split,cfg):
        self.cfg=cfg; self.split=split; self.rows=[]
        self.data_root=resolve_data_root(cfg)
        rows=list(csv.DictReader(open(self.data_root/'window_index.csv',encoding='utf-8')))
        train_lines=set(cfg.get('train_lines', ['Line3','LineL1']))
        val_lines=set(cfg.get('val_lines', ['Line7']))
        test_lines=set(cfg.get('test_lines', ['Line9']))
        review_lines=set(cfg.get('review_lines', ['Line6']))
        if split=='train': allow=train_lines
        elif split=='val': allow=val_lines
        elif split=='test': allow=test_lines
        elif split=='review': allow=review_lines
        else: allow=set()
        for r in rows:
            if r['line'] not in allow:
                continue
            include_ids=set(cfg.get(f'{split}_sample_ids', []))
            if include_ids and r['sample_id'] not in include_ids:
                continue
            line_include_ids=cfg.get(f'{split}_line_sample_ids', {})
            if r['line'] in line_include_ids and r['sample_id'] not in set(line_include_ids[r['line']]):
                continue
            exclude_ids=set(cfg.get(f'exclude_{split}_sample_ids', []))
            if exclude_ids and r['sample_id'] in exclude_ids:
                continue
            trace_ranges=cfg.get(f'{split}_trace_ranges',{})
            if r['line'] in trace_ranges:
                lo,hi=map(int,trace_ranges[r['line']])
                if int(r['start'])<lo or int(r['end'])>hi:
                    continue
            self.rows.append(r)
        if split=='train':
            extra=[]
            for r in self.rows:
                # Only train split oversampling. Validation/test/review remain untouched.
                if int(r['present'])>80: extra.append(r)
                if int(r['weak'])>120: extra.append(r)
                if int(r['no_pick'])>80:
                    extra += [r] * int(cfg.get('no_pick_window_repeats',4))
            self.rows += extra
            line_repeat=cfg.get('train_line_repeat_factors',{})
            if line_repeat:
                repeated=[]
                for r in self.rows:
                    repeat=max(1,int(line_repeat.get(r['line'],1)))
                    repeated += [r] * repeat
                self.rows=repeated
            random.shuffle(self.rows)
            if cfg.get('max_train_samples',0): self.rows=self.rows[:cfg['max_train_samples']]
        if split in ('val','test','review') and cfg.get(f'max_{split}_samples',0):
            self.rows=self.rows[:cfg[f'max_{split}_samples']]
    def __len__(self): return len(self.rows)
    def augment_train(self,x,y,pres,pres_valid,lw,ignore):
        aug=self.cfg.get('augment',{})
        if self.split!='train' or not aug.get('enabled',False):
            return x,y,pres,pres_valid,lw,ignore
        # raw-domain only augmentation.
        if aug.get('amp_scale_min') is not None and aug.get('amp_scale_max') is not None:
            scale=random.uniform(float(aug.get('amp_scale_min',0.9)), float(aug.get('amp_scale_max',1.1)))
            x=x.clone(); x[:1]=x[:1]*scale
        noise_std=float(aug.get('noise_std',0.0))
        if noise_std>0:
            x=x.clone(); x[:1]=x[:1]+torch.randn_like(x[:1])*noise_std
        meta_drop=float(self.cfg.get('terrain_metadata_dropout_prob',0.0))
        if meta_drop>0 and x.shape[0]>1:
            keep=(torch.rand(x.shape[0]-1,1,1,device=x.device)>meta_drop).float()
            x=x.clone(); x[1:]=x[1:]*keep
        drop_prob=float(aug.get('trace_dropout_prob',0.0))
        if drop_prob>0:
            W=x.shape[-1]
            mask=(torch.rand(W)>drop_prob).float()[None,:]
            x=x*mask
        flip_prob=float(aug.get('horizontal_flip_prob',0.0))
        if flip_prob>0 and random.random()<flip_prob:
            x=torch.flip(x,dims=[-1]); y=torch.flip(y,dims=[-1]); pres=torch.flip(pres,dims=[-1]); pres_valid=torch.flip(pres_valid,dims=[-1]); lw=torch.flip(lw,dims=[-1]); ignore=torch.flip(ignore,dims=[-1])
        # Spectral augmentation: random lowpass/bandpass filtering
        # Forces the model to learn frequency-invariant features,
        # which helps bridge the sim-real spectral gap.
        spec_aug_prob = float(aug.get('spectral_aug_prob', 0.0))
        if spec_aug_prob > 0 and random.random() < spec_aug_prob:
            x = x.clone()
            xfft = torch.fft.rfft(x, dim=2)
            nt = x.shape[2]
            # Random cutoff: low pass between 0.3 and 0.7 of Nyquist
            cutoff = random.uniform(0.3, 0.7)
            rolloff = random.uniform(0.05, 0.15)
            freq = torch.linspace(0, 1, xfft.shape[2], device=x.device)
            # Smooth sigmoid mask: 1 for low freqs, 0 for high
            mask = torch.sigmoid(-(freq - cutoff) / rolloff)
            # Random strength: 0.3 to 1.0
            strength = random.uniform(0.3, 1.0)
            xfft = xfft * (1.0 - strength + strength * mask)
            x[:1] = torch.fft.irfft(xfft, n=nt, dim=2)
        return x,y,pres,pres_valid,lw,ignore
    def __getitem__(self,i):
        r=self.rows[i]
        z=np.load(self.data_root/'windows'/(r['sample_id']+'.npz'))
        x=torch.from_numpy(z['x_raw'][None]).float()
        y=torch.from_numpy(z['y_mask'][None]).float()
        ignore_arr=z['ignore_mask'] if 'ignore_mask' in z.files else np.zeros_like(z['y_mask'],dtype=np.float32)
        ignore=torch.from_numpy(ignore_arr[None]).float()
        status=torch.from_numpy(z['status_code']).long()
        lw=torch.from_numpy(z['label_weight']).float()
        weak=status.eq(2).float()
        weak_target=float(self.cfg.get('loss',{}).get('weak_presence_target',0.5))
        pres=torch.zeros_like(status,dtype=torch.float32)
        pres=status.eq(1).float() + status.eq(2).float()*weak_target
        pres=pres[None]
        pres_valid=status.ne(2).float()[None]
        H,W=self.cfg['height_resize'],self.cfg['width_resize']
        x=F.interpolate(x[None],(H,W),mode='bilinear',align_corners=False)[0]
        x=compress_raw(x, self.cfg.get('input_log_scale',1e-3))
        x=normalize_raw_channel_3d(x,self.cfg)
        x=add_terrain_channels(x, r, self.cfg, self.data_root)
        y=F.interpolate(y[None],(H,W),mode='bilinear',align_corners=False)[0]
        ignore=F.interpolate(ignore[None],(H,W),mode='nearest')[0].clamp(0.0,1.0)
        pres=F.interpolate(pres[None,None],(1,W),mode='nearest')[0,0]
        pres_valid=F.interpolate(pres_valid[None,None],(1,W),mode='nearest')[0,0]
        lw=F.interpolate(lw[None,None,None],(1,W),mode='nearest')[0,0,0]
        weak=F.interpolate(weak[None,None,None],(1,W),mode='nearest')[0,0,0]
        lp=self.cfg.get('loss',{})
        if float(lp.get('label_weight_power',1.0)) != 1.0:
            lw=lw.clamp(0.0,1.0).pow(float(lp.get('label_weight_power',1.0)))
        weak_scale=float(lp.get('weak_label_weight_scale',1.0))
        if weak_scale != 1.0:
            lw=lw*torch.where(weak>0.5,torch.full_like(lw,weak_scale),torch.ones_like(lw))
        x,y,pres,pres_valid,lw,ignore=self.augment_train(x,y,pres,pres_valid,lw,ignore)
        core_thr=float(self.cfg.get('loss',{}).get('core_threshold',0.55))
        y_core=(y>=core_thr).float()
        return {'x':x,'y':y,'y_core':y_core,'presence':pres,'presence_valid':pres_valid,'weight':lw,'ignore_mask':ignore,'id':r['sample_id'],'line':r['line']}

def dice_loss_from_prob(pred,target,weight):
    eps=1e-6; inter=(pred*target*weight).sum((1,2,3)); den=((pred+target)*weight).sum((1,2,3))+eps
    return (1-2*inter/den).mean()

def unpack_model_output(out):
    from pgdacsnet.model_interfaces import unpack_pgda_output
    packed = unpack_pgda_output(out)
    return packed[0], packed[1], packed[2]

def centerline_aux_losses(center_logits, target, weight, cfg, ignore=None):
    if center_logits is None:
        zero = target.mean() * 0.0
        return zero, zero
    lp = cfg.get('loss', {})
    prob = torch.sigmoid(center_logits)
    H = prob.shape[2]
    ys = torch.linspace(0.0, 1.0, H, device=prob.device, dtype=prob.dtype)[None, None, :, None]
    target_mass = target.sum(dim=2).clamp_min(1e-6)
    pred_mass = prob.sum(dim=2).clamp_min(1e-6)
    target_valid = (target.sum(dim=2) > float(lp.get('center_valid_min_sum', 1e-3))).float()
    if ignore is not None:
        target_valid = target_valid * (ignore.mean(dim=2) < 0.5).float()
    target_center = (target * ys).sum(dim=2) / target_mass
    pred_center = (prob * ys).sum(dim=2) / pred_mass
    col_w = (0.25 + weight[:, None, :]) * target_valid
    center_l1 = (torch.abs(pred_center - target_center) * col_w).sum() / col_w.sum().clamp_min(1e-6)
    if pred_center.shape[-1] > 1:
        smooth_valid = target_valid[..., 1:] * target_valid[..., :-1]
        smooth = torch.abs(pred_center[..., 1:] - pred_center[..., :-1])
        continuity = (smooth * smooth_valid).sum() / smooth_valid.sum().clamp_min(1e-6)
    else:
        continuity = pred_center.mean() * 0.0
    return center_l1, continuity


def spectral_consistency_loss(pred_prob, target, cfg):
    """Penalise divergence in f-k spectral energy between predicted and target masks.
    Encourages the model to preserve the spectral structure of the interface signal.
    Only applied in regions where target has energy (>0.01)."""
    lp = cfg.get('loss', {})
    lam = float(lp.get('spectral_consistency_weight', 0.0))
    if lam <= 0:
        return pred_prob.mean() * 0.0
    # Only compute on high-energy regions to avoid noise
    mask = (target > 0.01).float()
    if mask.sum() < 100:
        return pred_prob.mean() * 0.0
    pred_masked = pred_prob * mask
    tgt_masked = target * mask
    # 2D FFT
    pred_fft = torch.fft.rfft2(pred_masked, norm="ortho")
    tgt_fft = torch.fft.rfft2(tgt_masked, norm="ortho")
    # Compare magnitude spectra (phase-agnostic)
    pred_mag = pred_fft.abs()
    tgt_mag = tgt_fft.abs()
    # Log-scale to focus on relative differences
    log_pred = torch.log1p(pred_mag)
    log_tgt = torch.log1p(tgt_mag)
    spec_loss = F.mse_loss(log_pred, log_tgt)
    return lam * spec_loss

def compute_loss(model,b,device,cfg):
    lp=cfg.get('loss',{})
    x=b['x'].to(device); y=b['y'].to(device); y_core=b['y_core'].to(device)
    pres=b['presence'].to(device); pres_valid=b['presence_valid'].to(device); lw=b['weight'].to(device)
    ignore=b.get('ignore_mask')
    ignore=ignore.to(device) if ignore is not None else torch.zeros_like(y)
    valid_pix=(1.0-ignore).clamp(0.0,1.0)
    valid_denom=valid_pix.sum().clamp_min(1.0)
    output = model(x)

    # GprMambaSep model — use extended decomposition losses
    if hasattr(output, 'A_hat') and output.A_hat is not None:
        from pgdacsnet.losses_pgda import compute_segmentation_losses as compute_seg
        from scripts.losses_gprmambasep import compute_gprmambasep_loss
        batch = {'x': x, 'y': y, 'y_core': y_core, 'presence': pres, 'presence_valid': pres_valid,
                 'weight': lw, 'valid_pix': valid_pix, 'valid_denom': valid_denom}
        total_loss, parts = compute_gprmambasep_loss(output, batch, cfg, model)
        return total_loss, parts

    # Standard model — original loss
    logits,pres_logits,center_logits=unpack_model_output(output); prob=torch.sigmoid(logits)
    pix_w=(float(lp.get('base_pixel_weight',0.10))+lw[:,None,None,:])*valid_pix
    pos_boost=float(lp.get('positive_pixel_boost',4.0))
    bce_w=pix_w*(1.0+pos_boost*y)
    band_bce=(F.binary_cross_entropy_with_logits(logits,y,reduction='none')*bce_w).sum()/valid_denom
    band_dice=dice_loss_from_prob(prob,y,pix_w)
    core_w=pix_w*(0.5+y_core)
    core_bce=(F.binary_cross_entropy_with_logits(logits,y_core,reduction='none')*core_w).sum()/valid_denom
    outside=(y<float(lp.get('outside_margin',0.05))).float()
    outside_penalty=(prob*outside*(0.15+pix_w)*valid_pix).sum()/valid_denom
    # Hard negative mining: explicitly penalize the brightest false-positive background pixels.
    bg=(y<float(lp.get('outside_margin',0.05))) & (valid_pix>0.5)
    bg_prob=prob[bg]
    if bg_prob.numel()>0:
        frac=float(lp.get('hard_negative_topk_frac',0.02))
        k=max(1,int(bg_prob.numel()*frac))
        hard_negative=torch.topk(bg_prob.flatten(),k).values.mean()
    else:
        hard_negative=prob.mean()*0.0
    pres_bce=F.binary_cross_entropy_with_logits(pres_logits,pres,reduction='none')
    neg_boost=float(lp.get('presence_negative_weight',5.0))
    pres_class_w=torch.where(pres<=0.05,torch.full_like(pres,neg_boost),torch.ones_like(pres))
    pres_w=(0.25+lw[:,None,:])*pres_class_w*pres_valid[:,None,:]
    pres_loss=(pres_bce*pres_w).sum()/pres_w.sum().clamp_min(1e-6)
    center_l1, continuity = centerline_aux_losses(center_logits, y, lw, cfg, ignore)
    spec_loss = spectral_consistency_loss(prob, y, cfg)
    loss=band_bce+float(lp.get('dice_weight',0.5))*band_dice+float(lp.get('core_weight',0.25))*core_bce+float(lp.get('outside_weight',0.40))*outside_penalty+float(lp.get('hard_negative_weight',0.35))*hard_negative+float(lp.get('presence_weight',0.25))*pres_loss+float(lp.get('centerline_weight',0.0))*center_l1+float(lp.get('continuity_weight',0.0))*continuity+spec_loss
    parts={'loss':float(loss.detach().cpu()),'band_bce':float(band_bce.detach().cpu()),'band_dice':float(band_dice.detach().cpu()),'core_bce':float(core_bce.detach().cpu()),'outside_penalty':float(outside_penalty.detach().cpu()),'hard_negative':float(hard_negative.detach().cpu()),'presence_loss':float(pres_loss.detach().cpu()),'centerline_l1':float(center_l1.detach().cpu()),'continuity':float(continuity.detach().cpu()),'spec_loss':float(spec_loss.detach().cpu())}
    return loss, parts

def reduce_parts(parts):
    if not parts: return {'loss':float('nan')}
    return {k:float(np.mean([p[k] for p in parts])) for k in parts[0]}

def run_epoch(model,loader,device,cfg,opt=None):
    is_train=opt is not None; model.train(is_train); parts=[]
    for b in loader:
        with torch.set_grad_enabled(is_train):
            loss,p=compute_loss(model,b,device,cfg)
            if is_train:
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        parts.append(p)
    return reduce_parts(parts)

def env_info(device):
    return {'python':sys.version.split()[0],'platform':platform.platform(),'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'device':str(device),'cuda_device_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}

def run(cfg_path):
    cfg=json.load(open(cfg_path,encoding='utf-8'))
    data_root=resolve_data_root(cfg)
    torch.set_num_threads(max(1, min(int(cfg.get('torch_threads',4)), torch.get_num_threads())))
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); random.seed(cfg['seed'])
    if cfg.get('deterministic', False):
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
    device=torch.device('cuda' if torch.cuda.is_available() and not cfg.get('force_cpu',False) else 'cpu')
    run_dir=ROOT/cfg['run_dir']; (run_dir/'previews').mkdir(parents=True,exist_ok=True); (run_dir/'logs').mkdir(exist_ok=True)
    model=build_model(cfg).to(device)
    # Warm-start: load weights from a pretrained checkpoint (missing keys are left random)
    ws = cfg.get("warm_start_from", "")
    if ws:
        ws_path = ROOT / ws if not Path(ws).is_absolute() else Path(ws)
        if ws_path.exists():
            ws_data = torch.load(ws_path, map_location=device, weights_only=False)
            ws_state = ws_data.get("model", ws_data)
            msd = model.state_dict()
            loaded = sum(1 for k in ws_state if k in msd and ws_state[k].shape == msd[k].shape)
            loaded_keys = [k for k in ws_state if k in msd and ws_state[k].shape == msd[k].shape]
            msd.update({k: ws_state[k] for k in loaded_keys})
            model.load_state_dict(msd, strict=False)
            print(f"WARM_START loaded {loaded}/{len(msd)} params from {ws_path.name}")
        else:
            print(f"WARM_START file not found: {ws_path}")
    opt=torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if "sgm" in n], "lr": cfg["lr"] * 0.5, "weight_decay": float(cfg.get("sgm_weight_decay", 0.05))},
        {"params": [p for n, p in model.named_parameters() if "sgm" not in n], "lr": cfg["lr"], "weight_decay": float(cfg.get("weight_decay", 1e-4))},
    ])
    train_real=DataLoader(DS('train',cfg),batch_size=cfg['batch_size'],shuffle=True,num_workers=int(cfg.get('num_workers',0)))
    # Simulation data loader (optional mixed training)
    train_sim = None
    sim_ratio = float(cfg.get('sim_batch_ratio', 0.0))
    if sim_ratio > 0 and cfg.get('sim_data_root'):
        sim_cfg = cfg.copy()
        sim_cfg['data_root'] = cfg['sim_data_root']
        sim_cfg['train_lines'] = cfg.get('sim_train_lines', [])
        # If sim_train_lines empty, auto-detect from sim window_index.csv
        if not sim_cfg['train_lines']:
            sim_idx = resolve_data_root(sim_cfg) / 'window_index.csv'
            if sim_idx.exists():
                import csv
                sim_lines = set()
                with open(sim_idx, encoding='utf-8') as f:
                    for row in csv.DictReader(f):
                        sim_lines.add(row['line'])
                sim_cfg['train_lines'] = list(sim_lines)
        if sim_cfg['train_lines']:
            train_sim = DataLoader(DS('train', sim_cfg), batch_size=cfg['batch_size'], shuffle=True, num_workers=0)
            print(f'SIM_DATA: {len(sim_cfg["train_lines"])} lines, {len(train_sim.dataset)} samples')
    val_ds=DS('val',cfg)
    val=DataLoader(val_ds,batch_size=1,shuffle=False,num_workers=0)
    hist=[]; best_val=1e9; best_epoch=0; info=env_info(device)
    json.dump(info,open(run_dir/'environment.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
    json.dump(cfg,open(run_dir/'used_config.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
    print('ENV',info,flush=True); print('DATA_ROOT',str(data_root),flush=True); print('TRAIN_LINES',cfg.get('train_lines'), 'VAL_LINES',cfg.get('val_lines'), 'TEST_LINES',cfg.get('test_lines'), flush=True); print('VAL_COUNT', len(val_ds), flush=True)
    # Create combined train loader if sim data is configured
    if train_sim:
        from torch.utils.data import ConcatDataset
        combined = ConcatDataset([train_real.dataset, train_sim.dataset])
        train = DataLoader(combined, batch_size=cfg['batch_size'], shuffle=True, num_workers=0)
        print(f'TRAIN combined: {len(combined)} samples (real={len(train_real.dataset)}, sim={len(train_sim.dataset)})')
    else:
        train = train_real
    for ep in range(1,cfg['epochs']+1):
        tr=run_epoch(model,train,device,cfg,opt)
        va=run_epoch(model,val,device,cfg,None) if len(val_ds)>0 else {'loss': float('nan')}
        va=run_epoch(model,val,device,cfg,None) if len(val_ds)>0 else {'loss': float('nan')}
        rec={'epoch':ep,'device':str(device)}; rec.update({f'train_{k}':v for k,v in tr.items()}); rec.update({f'val_{k}':v for k,v in va.items()})
        hist.append(rec); print(rec,flush=True)
        torch.save({'model':model.state_dict(),'cfg':cfg,'history':hist,'env':info,'epoch':ep},run_dir/'checkpoint_last.pt')
        monitor = va.get('loss', float('nan')) if len(val_ds)>0 else tr.get('loss', float('nan'))
        if np.isfinite(monitor) and monitor<best_val:
            best_val=monitor; best_epoch=ep; torch.save({'model':model.state_dict(),'cfg':cfg,'history':hist,'env':info,'epoch':ep,'best_monitor_loss':best_val,'monitor':'val_loss' if len(val_ds)>0 else 'train_loss'},run_dir/'checkpoint_best.pt')
    json.dump({'best_epoch':best_epoch,'best_monitor_loss':best_val,'monitor':'val_loss' if len(val_ds)>0 else 'train_loss','history':hist},open(run_dir/'history.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
    if (run_dir/'checkpoint_best.pt').exists():
        ckpt=torch.load(run_dir/'checkpoint_best.pt',map_location=device,weights_only=False); model.load_state_dict(ckpt['model'])
    
    if len(val_ds)>0:
        preview(model,DataLoader(DS('val',cfg),batch_size=1,shuffle=False),device,run_dir,'val',max_items=int(cfg.get('max_preview_val',4)))

def preview(model,loader,device,run_dir,prefix,max_items=4):
    model.eval()
    with torch.no_grad():
        for k,b in enumerate(loader):
            if k>=max_items: break
            x=b['x'].to(device); y=b['y'][0,0].numpy(); core=b['y_core'][0,0].numpy(); logits,pres,center=unpack_model_output(model(x)); pred=torch.sigmoid(logits)[0,0].cpu().numpy(); raw=b['x'][0,0].numpy()
            fig,ax=plt.subplots(1,5,figsize=(18,4),constrained_layout=True)
            ax[0].imshow(raw,aspect='auto',origin='upper',cmap='gray'); ax[0].set_title('输入：原始 raw',fontproperties=FONT)
            ax[1].imshow(y,aspect='auto',origin='upper',cmap='viridis',vmin=0,vmax=max(0.6,float(y.max()))); ax[1].set_title('验证标签：响应带',fontproperties=FONT)
            ax[2].imshow(core,aspect='auto',origin='upper',cmap='viridis',vmin=0,vmax=1); ax[2].set_title('辅助监督：核心窄带',fontproperties=FONT)
            ax[3].imshow(pred,aspect='auto',origin='upper',cmap='viridis',vmin=0,vmax=max(0.6,float(pred.max()))); ax[3].set_title('验证预测：响应带概率',fontproperties=FONT)
            ax[4].imshow(raw,aspect='auto',origin='upper',cmap='gray'); ax[4].imshow(pred,aspect='auto',origin='upper',cmap='magma',alpha=np.clip(pred*0.85,0,0.65)); ax[4].set_title('验证叠加图',fontproperties=FONT)
            for a in ax: a.set_xlabel('道号 / trace',fontproperties=FONT); a.set_ylabel('采样点 / sample',fontproperties=FONT)
            fig.savefig(run_dir/'previews'/f'{prefix}_{b["id"][0]}_preview.png',dpi=140); plt.close(fig)
if __name__=='__main__': run(sys.argv[1] if len(sys.argv)>1 else str(ROOT/'configs/fast_cpu_check.json'))
