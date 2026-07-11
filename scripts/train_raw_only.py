from pathlib import Path
import json,csv,random,sys,platform,hashlib
import os
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
os.environ.setdefault('PYTORCH_NO_DYNAMO', '1')
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,ConcatDataset,WeightedRandomSampler
import torch.nn.functional as F
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pgdacsnet.model_raw_unet import build_model, compress_raw
from pgdacsnet.font_utils import get_chinese_font
from pgdacsnet.experiment_contract import (
    ContractError,
    FORMAL_RUN_TYPES,
    enforce_simulation_holdout_policy,
    inspect_window_dataset,
    inspect_full_line_dataset,
    load_dataset_usage_policy,
    load_json_file,
    resolve_window_npz,
    validate_experiment_config,
)
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

def flip_directional_terrain_channels(x, cfg):
    """Correct directional metadata semantics after horizontal mirroring.

    Terrain channels begin after the raw channel. Reversing trace order changes
    the sign of the along-track slope and of the centred trace-position
    coordinate. Scalar elevation/height channels are reversed but not negated.
    """
    names=list(cfg.get('terrain_feature_names', []))
    if x.shape[0] <= 1 or not names:
        return x
    x=x.clone()
    for name in ('terrain_slope_z','trace_position'):
        if name in names:
            channel=1+names.index(name)
            if channel < x.shape[0]:
                x[channel]=-x[channel]
    return x


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


def _sha256_file(path, chunk_size=1024*1024):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(chunk_size),b''):
            h.update(chunk)
    return h.hexdigest()


def _read_index_rows(data_root):
    summary=inspect_window_dataset(Path(data_root),require_windows=True)
    with summary.index_path.open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f)),summary


def validate_training_dataset_root(data_root, *, purpose='training'):
    """Validate a window dataset and return its resolved index summary."""
    try:
        return inspect_window_dataset(Path(data_root), require_windows=True)
    except ContractError as exc:
        raise ContractError(f'{purpose} dataset contract failed: {exc}') from exc


def audit_sim_dataset_policy(cfg, sim_root):
    """Apply formal holdout and explicit training-permission policy checks."""
    try:
        return enforce_simulation_holdout_policy(cfg, Path(sim_root))
    except ContractError as exc:
        raise ContractError(f'Simulation dataset policy failed: {exc}') from exc


def audit_training_label_contract(cfg,data_root,index_rows):
    train_lines=set(cfg.get('train_lines') or [])
    rows=[r for r in index_rows if r.get('line') in train_lines]
    counts={'status_0':0,'status_1':0,'status_2':0,'other':0,'windows':len(rows)}
    for row in rows:
        path=resolve_window_npz(Path(data_root),row)
        with np.load(path,allow_pickle=False) as z:
            if 'status_code' not in z.files:
                raise ContractError(f'Window lacks status_code: {path}')
            values,n=np.unique(np.asarray(z['status_code']).astype(np.int64),return_counts=True)
        for value,count in zip(values.tolist(),n.tolist()):
            key=f'status_{value}' if value in (0,1,2) else 'other'
            counts[key]+=int(count)
    lp=cfg.get('loss',{}) or {}
    rejection_enabled=(
        float(lp.get('presence_weight',0.0) or 0.0)>0
        or float(lp.get('global_no_target_weight',0.0) or 0.0)>0
        or bool(cfg.get('enable_global_no_target_head',False))
    )
    formal=str(cfg.get('run_type','')).lower() in {'lolo_eval','holdout_eval','baseline_eval','paper_eval','paper_train'}
    if rejection_enabled and counts['status_0']<=0:
        message=('Rejection supervision is enabled, but the training split contains no confirmed '
                 'status_code=0 traces. Presence/no-target heads cannot learn a negative class.')
        if formal:
            raise ContractError(message)
        counts['warning']=message
    return counts


def audit_arrival_height_contract(cfg,index_rows):
    lp=cfg.get('loss',{}) or {}
    weight=max(float(lp.get('arrival_prior_weight',0.0) or 0.0),float(lp.get('arrival_time_prior_weight',0.0) or 0.0))
    if weight<=0:
        return {'arrival_prior_enabled':False}
    aliases=('antenna_height_agl_m','flight_height_agl_m','altitude','flight_height','flight_height_m','height_m','antenna_height_m')
    columns=set(index_rows[0]) if index_rows else set()
    height_columns=[name for name in aliases if name in columns]
    policy=str(cfg.get('arrival_prior_missing_height_policy','')).lower()
    valid_column='antenna_height_agl_valid' if 'antenna_height_agl_valid' in columns else None
    valid_rows=0
    if height_columns:
        for row in index_rows:
            raw_valid=str(row.get(valid_column,'true')).strip().lower() if valid_column else 'true'
            if raw_valid not in {'1','true','yes','y'}:
                continue
            try:
                value=float(row.get(height_columns[0], 'nan'))
            except (TypeError,ValueError):
                continue
            if np.isfinite(value) and value>0:
                valid_rows+=1
    if policy=='error' and not height_columns:
        raise ContractError('arrival prior is enabled, but window_index.csv has no recognised AGL height column')
    if policy=='error' and valid_rows<=0:
        raise ContractError('arrival prior is enabled, but no window has valid measured AGL height metadata')
    return {
        'arrival_prior_enabled':True,
        'height_columns':height_columns,
        'height_valid_column':valid_column,
        'valid_height_rows':valid_rows,
        'missing_height_policy':policy,
    }


def build_mixed_train_loader(train_real, train_sim, batch_size, sim_ratio, num_workers=0):
    """Build a train loader with an actual configurable real/sim sampling ratio."""
    combined = ConcatDataset([train_real.dataset, train_sim.dataset])
    real_n = len(train_real.dataset)
    sim_n = len(train_sim.dataset)
    ratio = float(max(0.0, min(1.0, sim_ratio)))
    if real_n <= 0 or sim_n <= 0:
        return DataLoader(combined, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    real_weight = (1.0 - ratio) / max(real_n, 1)
    sim_weight = ratio / max(sim_n, 1)
    weights = torch.cat([
        torch.full((real_n,), real_weight, dtype=torch.float32),
        torch.full((sim_n,), sim_weight, dtype=torch.float32),
    ])
    sampler = WeightedRandomSampler(weights, num_samples=real_n + sim_n, replacement=True)
    return DataLoader(combined, batch_size=batch_size, sampler=sampler, num_workers=num_workers)


OPTIONAL_COMPONENT_ARRAY_ALIASES = {
    'Y_air': ('Y_air', 'y_air', 'air_only', 'air_only_bscan'),
    'Y_target_without_G': ('Y_target_without_G', 'y_target_without_g'),
    'X_clean': ('X_clean', 'x_clean', 'clean', 'clean_bscan', 'x_target_clean'),
    'G_target': ('G_target', 'g_target'),
}


def _finite_array(a, *, name='array'):
    arr = np.asarray(a, dtype=np.float32)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return arr


def _load_optional_component_array(z, aliases):
    for key in aliases:
        if key in z.files:
            return _finite_array(z[key], name=key)
    return None

def is_gprmambasep_arch(cfg):
    arch = str(cfg.get('model_arch', '')).lower()
    return arch in (
        'v2_0_gprmambasep', 'gprmambasep',
        'v2_1_gprmambasep_lite', 'gprmambasep_lite',
        'v2_1_curvegassist_lite', 'curvegassist_lite', 'g_assisted_curvemamba',
    )


def component_supervision_space(cfg):
    return str(cfg.get('component_supervision_space', 'linear_standard')).lower()


def _component_linear_scale(x_linear, cfg):
    explicit = float(cfg.get('component_linear_scale', 0.0))
    if explicit > 0:
        return torch.as_tensor(explicit, dtype=x_linear.dtype, device=x_linear.device)
    q = float(cfg.get('component_linear_scale_quantile', 0.98))
    q = max(0.50, min(0.999, q))
    scale = torch.quantile(x_linear.abs().flatten(), q).clamp_min(float(cfg.get('component_linear_eps', 1e-6)))
    return scale


def _transform_component_tensor(t, x_linear_ref, cfg):
    """Transform component targets into the configured supervision space."""
    space = component_supervision_space(cfg)
    if space in ('linear', 'linear_standard', 'linear_amplitude'):
        scale = _component_linear_scale(x_linear_ref, cfg)
        clip = float(cfg.get('component_linear_clip', 8.0))
        return torch.clamp(t / scale, -clip, clip) / clip
    # Backward-compatible tensor-space path.  This is not physically linear.
    t = compress_raw(t, cfg.get('input_log_scale', 1e-3))
    return normalize_raw_channel_3d(t, cfg)


def _transform_reconstruction_target(x_linear_ref, cfg):
    space = component_supervision_space(cfg)
    if space in ('linear', 'linear_standard', 'linear_amplitude'):
        scale = _component_linear_scale(x_linear_ref, cfg)
        clip = float(cfg.get('component_linear_clip', 8.0))
        return torch.clamp(x_linear_ref / scale, -clip, clip) / clip
    t = compress_raw(x_linear_ref.clone(), cfg.get('input_log_scale', 1e-3))
    return normalize_raw_channel_3d(t, cfg)


def _fft_lowpass_2d(t, cutoff, rolloff, strength):
    """Apply the same vertical-frequency low-pass used by spectral augmentation."""
    if t is None:
        return None
    nt = t.shape[1]
    xfft = torch.fft.rfft(t, dim=1)
    freq = torch.linspace(0, 1, xfft.shape[1], device=t.device)
    mask = torch.sigmoid(-(freq - cutoff) / rolloff)
    xfft = xfft * (1.0 - strength + strength * mask)[None, :, None]
    return torch.fft.irfft(xfft, n=nt, dim=1)


def build_preprocess_signature(cfg):
    return {
        'model_arch': str(cfg.get('model_arch', '')),
        'variant': str(cfg.get('variant', '')),
        'base_ch': int(cfg.get('base_ch', 0)),
        'ssm_impl': str(cfg.get('ssm_impl', cfg.get('mamba_impl', ''))),
        'mamba_state_dim': int(cfg.get('mamba_state_dim', 0)),
        'mamba_d_conv': int(cfg.get('mamba_d_conv', cfg.get('ssm_kernel', 0))),
        'height_resize': int(cfg.get('height_resize', 0)),
        'width_resize': int(cfg.get('width_resize', 0)),
        'input_log_scale': float(cfg.get('input_log_scale', 1e-3)),
        'per_trace_robust_norm': bool(cfg.get('per_trace_robust_norm', False)),
        'per_trace_robust_clip': float(cfg.get('per_trace_robust_clip', 6.0)),
        'component_supervision_space': component_supervision_space(cfg),
        'component_linear_scale_quantile': float(cfg.get('component_linear_scale_quantile', 0.98)),
        'component_linear_clip': float(cfg.get('component_linear_clip', 8.0)),
        'component_linear_scale': float(cfg.get('component_linear_scale', 0.0)),
        'use_terrain_features': bool(cfg.get('use_terrain_features', False)),
        'terrain_feature_names': list(cfg.get('terrain_feature_names', [])),
    }


def _component_supervision_weight_from_cfg(cfg):
    lp = cfg.get('loss', {})
    return max(
        float(lp.get('sim_supervised_component_weight', 0.0)),
        float(lp.get('sim_supervised_weight', 0.0)),
        float(lp.get('component_supervision_weight', 0.0)),
    )


def _min_component_target_coverage(cfg):
    lp = cfg.get('loss', {})
    if 'min_component_target_coverage' in lp:
        return float(lp.get('min_component_target_coverage', 0.0))
    if 'min_component_target_coverage' in cfg:
        return float(cfg.get('min_component_target_coverage', 0.0))
    # If component supervision is requested at all, at least some batches must
    # carry component targets.  This catches silently-missing Y_air/X_clean/G
    # arrays while still permitting mixed real+sim training.
    return 0.01 if _component_supervision_weight_from_cfg(cfg) > 0 else 0.0


def _trace_range_pair_is_disjoint(cfg, line):
    train_range = (cfg.get('train_trace_ranges') or {}).get(line)
    test_range = (cfg.get('test_trace_ranges') or {}).get(line)
    if not train_range or not test_range:
        return False
    try:
        tr0, tr1 = map(int, train_range)
        te0, te1 = map(int, test_range)
    except Exception:
        return False
    return tr1 < te0 or te1 < tr0


def audit_config(cfg, config_path=None):
    """Validate experiment semantics plus model-specific compatibility fields."""
    contract = validate_experiment_config(
        cfg,
        ROOT,
        config_path=Path(config_path) if config_path else None,
        require_run_type=True,
    )
    warnings = list(contract.get('warnings', []))
    arch = str(cfg.get('model_arch', '')).lower()
    if is_gprmambasep_arch(cfg):
        if 'model_dropout' in cfg and 'decoder_dropout' not in cfg:
            warnings.append('v2 config uses model_dropout; mapped to decoder_dropout for compatibility.')
        if 'ssm_kernel' in cfg and 'mamba_d_conv' not in cfg:
            warnings.append('v2 config uses ssm_kernel; mapped to mamba_d_conv for compatibility.')
        if 'attention_heads' in cfg:
            warnings.append('attention_heads is unused by GprMambaSep/SSM-lite blocks.')
        if arch in ('v2_1_gprmambasep_lite', 'gprmambasep_lite'):
            warnings.append('v2_1_gprmambasep_lite is a base_ch-reduced alias, not a distinct architecture.')
    if is_gprmambasep_arch(cfg) and _component_supervision_weight_from_cfg(cfg) > 0 and _min_component_target_coverage(cfg) <= 0:
        warnings.append('component supervision is enabled but min_component_target_coverage<=0; missing targets will not fail-fast.')
    contract['warnings'] = warnings
    return contract


def _ids_from_dataset(ds):
    return {r.get('sample_id', '') for r in getattr(ds, 'rows', [])}


def build_split_audit(train_ds=None, val_ds=None, test_ds=None, review_ds=None):
    datasets={'train':train_ds,'val':val_ds,'test':test_ds,'review':review_ds}
    ids={k:_ids_from_dataset(v) for k,v in datasets.items() if v is not None}
    lines={k:sorted({r.get('line','') for r in getattr(v,'rows',[])}) for k,v in datasets.items() if v is not None}
    counts={k:len(getattr(v,'rows',[])) for k,v in datasets.items() if v is not None}
    overlaps={}; trace_overlaps={}; names=sorted(ids)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            inter=sorted(x for x in (ids[a]&ids[b]) if x)
            if inter: overlaps[f'{a}_{b}']=inter[:20]
            rows_a=getattr(datasets[a],'rows',[]) if datasets.get(a) is not None else []
            rows_b=getattr(datasets[b],'rows',[]) if datasets.get(b) is not None else []
            hits=[]
            for ra in rows_a:
                for rb in rows_b:
                    if ra.get('line')!=rb.get('line'): continue
                    a0,a1=int(ra.get('start',0)),int(ra.get('end',-1))
                    b0,b1=int(rb.get('start',0)),int(rb.get('end',-1))
                    if max(a0,b0)<=min(a1,b1):
                        hits.append({'line':ra.get('line'),'a':ra.get('sample_id'),'b':rb.get('sample_id'),'overlap':[max(a0,b0),min(a1,b1)]})
                        if len(hits)>=20: break
                if len(hits)>=20: break
            if hits: trace_overlaps[f'{a}_{b}']=hits
    return {'counts':counts,'lines':lines,'sample_id_overlaps':overlaps,'trace_interval_overlaps':trace_overlaps}

def assert_nonempty_dataset(ds, split_name):
    if len(ds) <= 0:
        raise RuntimeError(f'{split_name} dataset is empty. Check data_root, *_lines, trace ranges, and window_index.csv.')


def build_simulation_train_loader(cfg, *, num_workers=0):
    """Resolve and validate simulation training data without silent fallback."""
    sim_ratio = float(cfg.get('sim_batch_ratio', 0.0) or 0.0)
    if sim_ratio <= 0:
        return None, None
    sim_root_value = str(cfg.get('sim_data_root', '')).strip()
    if not sim_root_value:
        raise ContractError('sim_batch_ratio > 0 requires sim_data_root')
    sim_root = resolve_data_root({'data_root': sim_root_value})
    policy = enforce_simulation_holdout_policy(cfg, sim_root)
    requested_lines = list(cfg.get('sim_train_lines', []) or [])
    summary = inspect_window_dataset(
        sim_root,
        required_lines=requested_lines or None,
        require_windows=True,
    )
    sim_cfg = dict(cfg)
    sim_cfg['data_root'] = str(sim_root)
    sim_cfg['is_sim_dataset'] = True
    sim_cfg['train_lines'] = requested_lines or list(summary.lines)
    sim_train_ds = DS('train', sim_cfg)
    assert_nonempty_dataset(sim_train_ds, 'sim_train')
    loader = DataLoader(
        sim_train_ds,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=int(num_workers),
    )
    audit = {
        'requested_sim_batch_ratio': sim_ratio,
        'resolved_sim_root': str(sim_root),
        'resolved_sim_lines': list(summary.lines),
        'resolved_sim_index_rows': int(summary.row_count),
        'resolved_sim_dataset_samples_after_train_sampling': len(sim_train_ds),
        'dataset_usage_policy': policy or {},
    }
    return loader, audit


class DS(Dataset):
    def __init__(self,split,cfg):
        self.cfg=cfg; self.split=split; self.rows=[]
        self.data_root=resolve_data_root(cfg)
        inspect_window_dataset(self.data_root, require_windows=True)
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
        # Cache measured per-trace AGL heights from canonical original-CSV line
        # archives. This avoids reopening large line NPZ files for every window.
        self.line_height_cache={}
        for line_name in sorted({r.get('line','') for r in self.rows if r.get('line')}):
            line_path=self.data_root/'lines'/f'{line_name}.npz'
            if not line_path.is_file():
                continue
            try:
                with np.load(line_path,allow_pickle=False) as line_data:
                    if 'flight_height_agl_m' in line_data.files:
                        values=np.asarray(line_data['flight_height_agl_m'],dtype=np.float32)
                        if values.ndim==1 and np.isfinite(values).all() and np.all(values>0):
                            self.line_height_cache[line_name]=values.copy()
            except Exception:
                # Formal dataset checks report malformed line metadata. Dataset
                # construction keeps the explicit index fallback for smoke data.
                continue
    def __len__(self): return len(self.rows)
    def augment_train(self,x,y,pres,pres_valid,lw,ignore,component_tensors=None,y_full_component=None,altitude=None,altitude_valid=None):
        """Apply train-time augmentation while keeping component supervision aligned.

        Geometry/gain/spectral transforms that change the physical B-scan are
        synchronised to component targets and the reconstruction target.  Input
        corruption transforms such as additive noise remain input-only.
        """
        component_tensors = component_tensors or {}
        aug=self.cfg.get('augment',{})
        if self.split!='train' or not aug.get('enabled',False):
            return x,y,pres,pres_valid,lw,ignore,component_tensors,y_full_component,altitude,altitude_valid
        # Gain augmentation is covariant: scale input raw and component targets.
        if aug.get('amp_scale_min') is not None and aug.get('amp_scale_max') is not None:
            scale=random.uniform(float(aug.get('amp_scale_min',0.9)), float(aug.get('amp_scale_max',1.1)))
            x=x.clone(); x[:1]=x[:1]*scale
            if y_full_component is not None: y_full_component=y_full_component*scale
            component_tensors={k:v*scale for k,v in component_tensors.items()}
        # Additive noise is input-only: targets remain clean.
        noise_std=float(aug.get('noise_std',0.0))
        if noise_std>0:
            x=x.clone(); x[:1]=x[:1]+torch.randn_like(x[:1])*noise_std
        meta_drop=float(self.cfg.get('terrain_metadata_dropout_prob',0.0))
        if meta_drop>0 and x.shape[0]>1:
            keep=(torch.rand(x.shape[0]-1,1,1,device=x.device)>meta_drop).float()
            x=x.clone(); x[1:]=x[1:]*keep
        # Trace dropout is input corruption; labels/components stay clean.
        drop_prob=float(aug.get('trace_dropout_prob',0.0))
        if drop_prob>0:
            W=x.shape[-1]
            mask=(torch.rand(W)>drop_prob).float()[None,:]
            x=x.clone(); x[:1]=x[:1]*mask
        flip_prob=float(aug.get('horizontal_flip_prob',0.0))
        if flip_prob>0 and random.random()<flip_prob:
            x=torch.flip(x,dims=[-1]); x=flip_directional_terrain_channels(x,self.cfg); y=torch.flip(y,dims=[-1]); pres=torch.flip(pres,dims=[-1]); pres_valid=torch.flip(pres_valid,dims=[-1]); lw=torch.flip(lw,dims=[-1]); ignore=torch.flip(ignore,dims=[-1])
            if y_full_component is not None: y_full_component=torch.flip(y_full_component,dims=[-1])
            component_tensors={k:torch.flip(v,dims=[-1]) for k,v in component_tensors.items()}
            if altitude is not None: altitude=torch.flip(altitude,dims=[-1])
            if altitude_valid is not None: altitude_valid=torch.flip(altitude_valid,dims=[-1])
        spec_aug_prob = float(aug.get('spectral_aug_prob', 0.0))
        if spec_aug_prob > 0 and random.random() < spec_aug_prob:
            cutoff = random.uniform(0.3, 0.7)
            rolloff = random.uniform(0.05, 0.15)
            strength = random.uniform(0.3, 1.0)
            x=x.clone(); x[:1] = _fft_lowpass_2d(x[:1], cutoff, rolloff, strength)
            if bool(aug.get('spectral_aug_sync_components', True)):
                if y_full_component is not None: y_full_component = _fft_lowpass_2d(y_full_component, cutoff, rolloff, strength)
                component_tensors={k:_fft_lowpass_2d(v, cutoff, rolloff, strength) for k,v in component_tensors.items()}
        return x,y,pres,pres_valid,lw,ignore,component_tensors,y_full_component,altitude,altitude_valid
    def __getitem__(self,i):
        r=self.rows[i]
        z=np.load(resolve_window_npz(self.data_root,r),allow_pickle=False)
        x_raw_linear=torch.from_numpy(_finite_array(z['x_raw'], name='x_raw')[None]).float()
        y=torch.from_numpy(_finite_array(z['y_mask'], name='y_mask')[None]).float().clamp(0.0, 1.0)
        ignore_arr=_finite_array(z['ignore_mask'], name='ignore_mask') if 'ignore_mask' in z.files else np.zeros_like(z['y_mask'],dtype=np.float32)
        ignore=torch.from_numpy(ignore_arr[None]).float()
        status=torch.from_numpy(np.nan_to_num(z['status_code'], nan=0).astype(np.int64)).long()
        lw=torch.from_numpy(_finite_array(z['label_weight'], name='label_weight')).float().clamp(0.0, 1.0)
        weak=status.eq(2).float()
        weak_target=float(self.cfg.get('loss',{}).get('weak_presence_target',0.5))
        pres=status.eq(1).float() + status.eq(2).float()*weak_target
        pres=pres[None]
        pres_valid=status.ne(2).float()[None]
        H,W=self.cfg['height_resize'],self.cfg['width_resize']

        # Prefer measured per-trace flight height from the canonical original
        # CSV line. Fall back to an explicit window-level scalar only for legacy
        # simulation/smoke datasets.
        altitude_orig=None
        altitude_valid_orig=None
        line_heights=self.line_height_cache.get(r.get('line',''))
        if line_heights is not None:
            start_idx,end_idx=int(r.get('start',0)),int(r.get('end',-1))
            if 0<=start_idx<=end_idx<line_heights.size:
                values=line_heights[start_idx:end_idx+1]
                if values.size==status.numel():
                    altitude_orig=torch.from_numpy(values.copy()).float()
                    altitude_valid_orig=torch.isfinite(altitude_orig).float()*(altitude_orig>0).float()
        if altitude_orig is None:
            altitude_value=float('nan'); altitude_flag=0.0
            for meta_key in (
                'antenna_height_agl_m','flight_height_agl_m','altitude','flight_height',
                'flight_height_m','height_m','antenna_height_m',
            ):
                if meta_key not in r or not str(r[meta_key]).strip():
                    continue
                try: candidate=float(r[meta_key])
                except (TypeError,ValueError): continue
                if np.isfinite(candidate) and candidate>0:
                    validity_key='antenna_height_agl_valid' if meta_key=='antenna_height_agl_m' else f'{meta_key}_valid'
                    raw_valid=str(r.get(validity_key,'true')).strip().lower()
                    altitude_value=candidate
                    altitude_flag=1.0 if raw_valid in {'1','true','yes','y'} else 0.0
                    break
            altitude_orig=torch.full((status.numel(),),altitude_value,dtype=torch.float32)
            altitude_valid_orig=torch.full((status.numel(),),altitude_flag,dtype=torch.float32)
        altitude=F.interpolate(altitude_orig[None,None],size=W,mode='linear',align_corners=False)[0,0]
        altitude_valid=F.interpolate(altitude_valid_orig[None,None],size=W,mode='nearest')[0,0]
        altitude_valid=altitude_valid*(torch.isfinite(altitude)&(altitude>0)).float()

        # Keep a linear reconstruction target for A/S/G when requested.  The
        # model input can still use compressed/robust-normalised raw.
        x_linear=F.interpolate(x_raw_linear[None],(H,W),mode='bilinear',align_corners=False)[0]
        y_full_component=_transform_reconstruction_target(x_linear, self.cfg)

        x=compress_raw(x_linear.clone(), self.cfg.get('input_log_scale',1e-3))
        x=normalize_raw_channel_3d(x,self.cfg)
        x=add_terrain_channels(x, r, self.cfg, self.data_root)
        y=F.interpolate(y[None],(H,W),mode='bilinear',align_corners=False)[0]
        ignore=F.interpolate(ignore[None],(H,W),mode='nearest')[0].clamp(0.0,1.0)
        pres=F.interpolate(pres[None,None],(1,W),mode='nearest')[0,0]
        pres_valid=F.interpolate(pres_valid[None,None],(1,W),mode='nearest')[0,0]
        lw=F.interpolate(lw[None,None,None],(1,W),mode='nearest')[0,0,0]
        weak=F.interpolate(weak[None,None,None],(1,W),mode='nearest')[0,0,0]

        component_tensors = {}
        component_valid = {}
        for batch_key, aliases in OPTIONAL_COMPONENT_ARRAY_ALIASES.items():
            arr = _load_optional_component_array(z, aliases)
            if arr is None or arr.ndim != 2:
                # Always return a shape-compatible placeholder so mixed real/sim
                # batches collate safely.  Validity flags decide whether losses
                # use the target.
                component_tensors[batch_key] = torch.zeros_like(y_full_component)
                component_valid[batch_key] = torch.tensor(0.0, dtype=torch.float32)
                continue
            t = torch.from_numpy(arr[None]).float()
            t = F.interpolate(t[None], (H, W), mode='bilinear', align_corners=False)[0]
            t = _transform_component_tensor(t, x_linear, self.cfg)
            component_tensors[batch_key] = t
            component_valid[batch_key] = torch.tensor(1.0, dtype=torch.float32)

        lp=self.cfg.get('loss',{})
        if float(lp.get('label_weight_power',1.0)) != 1.0:
            lw=lw.clamp(0.0,1.0).pow(float(lp.get('label_weight_power',1.0)))
        weak_scale=float(lp.get('weak_label_weight_scale',1.0))
        if weak_scale != 1.0:
            lw=lw*torch.where(weak>0.5,torch.full_like(lw,weak_scale),torch.ones_like(lw))

        x,y,pres,pres_valid,lw,ignore,component_tensors,y_full_component,altitude,altitude_valid=self.augment_train(
            x,y,pres,pres_valid,lw,ignore,component_tensors,y_full_component,altitude,altitude_valid
        )
        core_thr=float(self.cfg.get('loss',{}).get('core_threshold',0.55))
        y_core=(y>=core_thr).float()
        has_any=1.0 if any(float(v) > 0.5 for v in component_valid.values()) else 0.0
        sample = {
            'x':x,'y':y,'y_core':y_core,'presence':pres,'presence_valid':pres_valid,
            'weight':lw,'ignore_mask':ignore,'id':r['sample_id'],'line':r['line'],
            'Y_full_component':y_full_component,
            'has_component_targets':torch.tensor(has_any,dtype=torch.float32),
            'is_sim':torch.tensor(float(self.cfg.get('is_sim_dataset', False)),dtype=torch.float32),
        }
        # Measured AGL height is trace-aligned with the resized B-scan. The
        # loss supports both this vector form and legacy scalar metadata.
        sample['altitude'] = altitude.to(dtype=torch.float32)
        sample['altitude_valid'] = altitude_valid.to(dtype=torch.float32)
        sample.update(component_tensors)
        for key, flag in component_valid.items():
            sample[f'{key}_valid'] = flag
        return sample


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

def compute_loss(model,b,device,cfg,discriminator=None,grl_layer=None):
    lp=cfg.get('loss',{})
    x=b['x'].to(device); y=b['y'].to(device); y_core=b['y_core'].to(device)
    pres=b['presence'].to(device); pres_valid=b['presence_valid'].to(device); lw=b['weight'].to(device)
    ignore=b.get('ignore_mask')
    ignore=ignore.to(device) if ignore is not None else torch.zeros_like(y)
    valid_pix=(1.0-ignore).clamp(0.0,1.0)
    valid_denom=valid_pix.sum().clamp_min(1.0)
    # AeroPath-SSD consumes measured tracewise altitude to build its auxiliary
    # reduced-time view. Legacy models retain their one-argument forward path.
    if bool(getattr(model, 'accepts_altitude', False)):
        altitude = b.get('altitude')
        altitude = altitude.to(device) if hasattr(altitude, 'to') else None
        output = model(x, altitude=altitude)
    else:
        output = model(x)

    # GprMambaSep model — use extended decomposition losses
    if hasattr(output, 'A_hat') and output.A_hat is not None:
        from scripts.losses_gprmambasep import compute_gprmambasep_loss
        batch = {
            'x': x,
            'y': y,
            'y_core': y_core,
            'presence': pres,
            'presence_valid': pres_valid,
            'weight': lw,
            'valid_pix': valid_pix,
            'valid_denom': valid_denom,
            'ignore_mask': ignore,
        }
        for key in OPTIONAL_COMPONENT_ARRAY_ALIASES:
            if key in b:
                batch[key] = b[key].to(device)
            flag_key = f'{key}_valid'
            if flag_key in b:
                batch[flag_key] = b[flag_key].to(device)
        for key in ('Y_full_component','has_component_targets','is_sim','altitude','altitude_valid'):
            if key in b:
                batch[key] = b[key].to(device) if hasattr(b[key], 'to') else b[key]
        total_loss, parts = compute_gprmambasep_loss(
            output, batch, cfg, model,
            discriminator=discriminator, grl_layer=grl_layer,
            stage3=bool(cfg.get('stage3', False) or cfg.get('fine_tune_stage3', False)),
        )
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

def run_epoch(model,loader,device,cfg,opt=None,scaler=None,discriminator=None,grl_layer=None):
    is_train=opt is not None
    model.train(is_train)
    if discriminator is not None:
        discriminator.train(is_train)
    parts=[]
    amp_enabled=bool(cfg.get('amp',False)) and device.type=='cuda'
    grad_accum_steps=max(1,int(cfg.get('grad_accum_steps',1))) if is_train else 1
    if is_train:
        opt.zero_grad(set_to_none=True)
    for step,b in enumerate(loader, start=1):
        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                loss,p=compute_loss(model,b,device,cfg,discriminator=discriminator,grl_layer=grl_layer)
                loss_for_backward=loss/float(grad_accum_steps)
            if is_train:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss_for_backward).backward()
                    if step % grad_accum_steps == 0 or step == len(loader):
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params'] if p.grad is not None],5.0)
                        scaler.step(opt)
                        scaler.update()
                        opt.zero_grad(set_to_none=True)
                else:
                    loss_for_backward.backward()
                    if step % grad_accum_steps == 0 or step == len(loader):
                        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params'] if p.grad is not None],5.0)
                        opt.step()
                        opt.zero_grad(set_to_none=True)
        p=dict(p)
        p.setdefault('grad_accum_steps', float(grad_accum_steps))
        parts.append(p)
    return reduce_parts(parts)

def env_info(device):
    return {'python':sys.version.split()[0],'platform':platform.platform(),'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'device':str(device),'cuda_device_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}


def capture_rng_state():
    state={
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['torch_cuda_all']=torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return False
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if torch.cuda.is_available() and state.get('torch_cuda_all') is not None:
        torch.cuda.set_rng_state_all(state['torch_cuda_all'])
    return True


def build_checkpoint_payload(model, discriminator, opt, scaler, cfg, hist, info, epoch, preprocess_signature, cfg_audit, **extra):
    payload={
        'model':model.state_dict(),
        'discriminator':discriminator.state_dict() if discriminator is not None else None,
        'optimizer':opt.state_dict() if opt is not None else None,
        'scaler':scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        'rng_state':capture_rng_state(),
        'cfg':cfg,
        'history':hist,
        'env':info,
        'epoch':epoch,
        'preprocess_signature':preprocess_signature,
        'config_audit':cfg_audit,
        'checkpoint_contract_version':2,
    }
    payload.update(extra)
    return payload

def run(cfg_path):
    cfg=load_json_file(Path(cfg_path))
    cfg_audit=audit_config(cfg, cfg_path)
    data_root=resolve_data_root(cfg)
    real_index_rows, real_dataset_summary = _read_index_rows(data_root)
    real_dataset_policy = load_dataset_usage_policy(data_root) or {}
    real_training_allowed = real_dataset_policy.get(
        'training_allowed', real_dataset_policy.get('train_allowed', True)
    )
    if real_training_allowed is False and not bool(cfg.get('allow_incomplete_dataset', False)):
        raise ContractError(
            f'real dataset policy forbids training use: {data_root}; '
            f"reason={real_dataset_policy.get('reason', real_dataset_policy.get('blocking_reasons', 'unspecified'))}"
        )
    label_contract_audit = audit_training_label_contract(cfg, data_root, real_index_rows)
    height_contract_audit = audit_arrival_height_contract(cfg, real_index_rows)
    formal_run = str(cfg.get('run_type','')).lower() in FORMAL_RUN_TYPES
    full_line_audit = None
    if formal_run:
        full_line_audit = inspect_full_line_dataset(
            data_root,
            list(cfg.get('val_lines', [])) + list(cfg.get('test_lines', [])),
        )
    cfg_audit['real_dataset'] = {
        'root': str(data_root),
        'index_path': str(real_dataset_summary.index_path),
        'index_sha256': _sha256_file(real_dataset_summary.index_path),
        'index_rows': int(real_dataset_summary.row_count),
        'lines': list(real_dataset_summary.lines),
        'policy': real_dataset_policy,
        'label_contract': label_contract_audit,
        'height_contract': height_contract_audit,
        'full_line_contract': full_line_audit,
    }
    torch.set_num_threads(max(1, min(int(cfg.get('torch_threads',4)), torch.get_num_threads())))
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); random.seed(cfg['seed'])
    if cfg.get('deterministic', False):
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
    device=torch.device('cuda' if torch.cuda.is_available() and not cfg.get('force_cpu',False) else 'cpu')
    run_dir=ROOT/cfg['run_dir']; (run_dir/'previews').mkdir(parents=True,exist_ok=True); (run_dir/'logs').mkdir(exist_ok=True)
    model=build_model(cfg).to(device)
    discriminator=None; grl_layer=None
    lp_cfg=cfg.get('loss',{})
    sep_w=max(float(lp_cfg.get('component_separation_weight',0.0)), float(lp_cfg.get('contrastive_separation_weight',0.0)), float(lp_cfg.get('contrastive_weight',0.0)))
    if is_gprmambasep_arch(cfg) and sep_w>0:
        from scripts.losses_gprmambasep import ComponentDiscriminator, GRL
        discriminator=ComponentDiscriminator(input_dim=8, hidden_dim=int(lp_cfg.get('component_discriminator_hidden',32))).to(device)
        grl_layer=GRL(lambd=float(lp_cfg.get('grad_reverse_lambda',1.0)))
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
            if discriminator is not None and isinstance(ws_data, dict) and ws_data.get('discriminator'):
                try:
                    discriminator.load_state_dict(ws_data['discriminator'], strict=False)
                    print(f"WARM_START loaded discriminator from {ws_path.name}")
                except Exception as exc:
                    print(f"WARM_START discriminator skipped: {exc}")
            print(f"WARM_START loaded {loaded}/{len(msd)} params from {ws_path.name}")
        else:
            if bool(cfg.get('allow_missing_warm_start', False)):
                print(f"WARM_START file not found (explicitly allowed): {ws_path}")
            else:
                raise FileNotFoundError(
                    f"warm_start_from was requested but checkpoint is missing: {ws_path}. "
                    "Set allow_missing_warm_start=true only for an intentional ablation."
                )
    param_groups=[
        {"params": [p for n, p in model.named_parameters() if "sgm" in n], "lr": cfg["lr"] * 0.5, "weight_decay": float(cfg.get("sgm_weight_decay", 0.05))},
        {"params": [p for n, p in model.named_parameters() if "sgm" not in n], "lr": cfg["lr"], "weight_decay": float(cfg.get("weight_decay", 1e-4))},
    ]
    if discriminator is not None:
        param_groups.append({"params": discriminator.parameters(), "lr": float(cfg.get('component_discriminator_lr', cfg['lr'])), "weight_decay": float(cfg.get('component_discriminator_weight_decay', 1e-4))})
    opt=torch.optim.AdamW(param_groups)
    scaler = torch.amp.GradScaler('cuda', enabled=bool(cfg.get('amp', False)) and device.type == 'cuda')
    train_real_ds = DS('train', cfg)
    assert_nonempty_dataset(train_real_ds, 'train')
    train_real=DataLoader(train_real_ds,batch_size=cfg['batch_size'],shuffle=True,num_workers=int(cfg.get('num_workers',0)))
    # Simulation data loader: requested mixed training must never degrade silently.
    sim_ratio = float(cfg.get('sim_batch_ratio', 0.0) or 0.0)
    train_sim, sim_data_audit = build_simulation_train_loader(cfg, num_workers=0)
    if train_sim is not None:
        print(
            f"SIM_DATA: {len(sim_data_audit['resolved_sim_lines'])} lines, "
            f"{len(train_sim.dataset)} samples",
            flush=True,
        )
    val_ds=DS('val',cfg)
    test_ds=DS('test',cfg)
    review_ds=DS('review',cfg)
    val=DataLoader(val_ds,batch_size=1,shuffle=False,num_workers=0)
    test=DataLoader(test_ds,batch_size=1,shuffle=False,num_workers=0)
    hist=[]; best_val=1e9; best_epoch=0; info=env_info(device)
    preprocess_signature=build_preprocess_signature(cfg)
    json.dump(info,open(run_dir/'environment.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
    json.dump(cfg,open(run_dir/'used_config.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
    json.dump(cfg_audit,open(run_dir/'config_audit.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
    json.dump(preprocess_signature,open(run_dir/'preprocess_signature.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
    if sim_data_audit is not None:
        json.dump(sim_data_audit,open(run_dir/'sim_dataset_audit.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
    split_audit = build_split_audit(train_real_ds, val_ds, test_ds, review_ds)
    strict_run = str(cfg.get('run_type','')).lower() in {
        'lolo_eval','holdout_eval','baseline_eval','paper_eval','paper_train'
    }
    if strict_run and (split_audit.get('sample_id_overlaps') or split_audit.get('trace_interval_overlaps')):
        raise ContractError(
            'SPLIT_AUDIT failed: source samples/traces overlap in a formal run: '
            + json.dumps(split_audit, ensure_ascii=False)
        )
    json.dump(split_audit,open(run_dir/'split_audit.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
    run_manifest={
        'requested_sim_batch_ratio': sim_ratio,
        'resolved_real_samples': len(train_real.dataset),
        'resolved_sim_samples': len(train_sim.dataset) if train_sim is not None else 0,
        'actual_loader_mode': 'mixed_weighted' if train_sim is not None else 'real_only',
        'config_path': str(Path(cfg_path).resolve()),
        'config_sha256': _sha256_file(Path(cfg_path)),
        'data_root': str(data_root),
        'real_index_path': str(real_dataset_summary.index_path),
        'real_index_sha256': _sha256_file(real_dataset_summary.index_path),
        'real_dataset_policy': real_dataset_policy,
        'label_contract_audit': label_contract_audit,
        'height_contract_audit': height_contract_audit,
        'full_line_contract_audit': full_line_audit,
        'simulation_dataset_audit': sim_data_audit,
        'split_audit': split_audit,
    }
    json.dump(run_manifest,open(run_dir/'run_manifest.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
    print('ENV',info,flush=True); print('CONFIG_AUDIT',cfg_audit,flush=True); print('SPLIT_AUDIT',split_audit,flush=True); print('DATA_ROOT',str(data_root),flush=True); print('TRAIN_LINES',cfg.get('train_lines'), 'VAL_LINES',cfg.get('val_lines'), 'TEST_LINES',cfg.get('test_lines'), flush=True); print('VAL_COUNT', len(val_ds), 'TEST_COUNT', len(test_ds), flush=True)
    # Create mixed train loader with an actual configurable sim/real ratio
    if train_sim:
        train = build_mixed_train_loader(
            train_real,
            train_sim,
            batch_size=cfg['batch_size'],
            sim_ratio=sim_ratio,
            num_workers=0,
        )
        print(f'TRAIN mixed-ratio: real={len(train_real.dataset)} sim={len(train_sim.dataset)} sim_ratio={sim_ratio:.3f}')
    else:
        train = train_real
    min_component_cov = _min_component_target_coverage(cfg)
    for ep in range(1,cfg['epochs']+1):
        tr=run_epoch(model,train,device,cfg,opt,scaler=scaler,discriminator=discriminator,grl_layer=grl_layer)
        if min_component_cov > 0 and float(tr.get('component_has_any', 0.0)) < min_component_cov:
            raise RuntimeError(
                f"Component supervision is enabled but train component coverage is "
                f"{float(tr.get('component_has_any', 0.0)):.4f} < {min_component_cov:.4f}. "
                "Check simulation .npz component arrays or set loss.min_component_target_coverage explicitly."
            )
        va=run_epoch(model,val,device,cfg,None,scaler=None,discriminator=discriminator,grl_layer=grl_layer) if len(val_ds)>0 else {'loss': float('nan')}
        rec={'epoch':ep,'device':str(device)}; rec.update({f'train_{k}':v for k,v in tr.items()}); rec.update({f'val_{k}':v for k,v in va.items()})
        hist.append(rec); print(rec,flush=True)
        torch.save(
            build_checkpoint_payload(model,discriminator,opt,scaler,cfg,hist,info,ep,preprocess_signature,cfg_audit),
            run_dir/'checkpoint_last.pt'
        )
        monitor = va.get('loss', float('nan')) if len(val_ds)>0 else tr.get('loss', float('nan'))
        if np.isfinite(monitor) and monitor<best_val:
            best_val=monitor; best_epoch=ep
            torch.save(
                build_checkpoint_payload(
                    model,discriminator,opt,scaler,cfg,hist,info,ep,preprocess_signature,cfg_audit,
                    best_monitor_loss=best_val,
                    monitor='val_loss' if len(val_ds)>0 else 'train_loss',
                ),
                run_dir/'checkpoint_best.pt'
            )
    json.dump({'best_epoch':best_epoch,'best_monitor_loss':best_val,'monitor':'val_loss' if len(val_ds)>0 else 'train_loss','history':hist},open(run_dir/'history.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
    if (run_dir/'checkpoint_best.pt').exists():
        ckpt=torch.load(run_dir/'checkpoint_best.pt',map_location=device,weights_only=False); model.load_state_dict(ckpt['model']);
        (discriminator.load_state_dict(ckpt['discriminator']) if discriminator is not None and ckpt.get('discriminator') is not None else None)

    final_metrics = {}
    if len(val_ds)>0:
        final_metrics['val_best'] = run_epoch(model,val,device,cfg,None,scaler=None,discriminator=discriminator,grl_layer=grl_layer)
        preview(model,DataLoader(DS('val',cfg),batch_size=1,shuffle=False),device,run_dir,'val',max_items=int(cfg.get('max_preview_val',4)))
    if len(test_ds)>0 and bool(cfg.get('eval_test_after_train', True)):
        final_metrics['test_best'] = run_epoch(model,test,device,cfg,None,scaler=None,discriminator=discriminator,grl_layer=grl_layer)
        if bool(cfg.get('preview_test_after_train', False)):
            preview(model,DataLoader(DS('test',cfg),batch_size=1,shuffle=False),device,run_dir,'test',max_items=int(cfg.get('max_preview_test',4)))
    if final_metrics:
        json.dump(final_metrics,open(run_dir/'final_eval_metrics.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)

def _imshow_signed(ax, img, title):
    v=float(np.nanpercentile(np.abs(img),98)) if np.isfinite(img).any() else 1.0
    ax.imshow(img,aspect='auto',origin='upper',cmap='RdBu_r',vmin=-max(v,1e-6),vmax=max(v,1e-6)); ax.set_title(title,fontproperties=FONT)


def preview(model,loader,device,run_dir,prefix,max_items=4):
    model.eval()
    with torch.no_grad():
        for k,b in enumerate(loader):
            if k>=max_items: break
            x=b['x'].to(device); y=b['y'][0,0].numpy(); core=b['y_core'][0,0].numpy(); raw=b['x'][0,0].numpy()
            out=model(x); logits,pres,center=unpack_model_output(out); pred=torch.sigmoid(logits)[0,0].cpu().numpy()
            if hasattr(out,'A_hat') and out.A_hat is not None:
                A=out.A_hat[0,0].detach().cpu().numpy(); S=out.S_hat[0,0].detach().cpu().numpy(); G=out.G_hat[0,0].detach().cpu().numpy()
                recon=A+S+G
                comp_ref = b.get('Y_full_component')
                if comp_ref is not None:
                    comp_ref_np = comp_ref[0,0].numpy()
                    resid = comp_ref_np - recon
                    ref_title = 'Y_full_component / component space'
                else:
                    comp_ref_np = raw
                    resid = raw - recon
                    ref_title = 'fallback raw / model space'
                gates = getattr(out, 'component_gates', None)
                if gates is not None:
                    gate_np = gates[0].detach().cpu().numpy()
                    fig,ax=plt.subplots(3,5,figsize=(20,11),constrained_layout=True)
                else:
                    gate_np = None
                    fig,ax=plt.subplots(2,5,figsize=(20,8),constrained_layout=True)
                _imshow_signed(ax[0,0], comp_ref_np, ref_title)
                _imshow_signed(ax[0,1], A, 'A_hat：空耦/早期')
                _imshow_signed(ax[0,2], S, 'S_hat：地表')
                _imshow_signed(ax[0,3], G, 'G_hat：地下地质')
                _imshow_signed(ax[0,4], resid, '同空间残差 ref-(A+S+G)')
                ax[1,0].imshow(y,aspect='auto',origin='upper',cmap='viridis',vmin=0,vmax=max(0.6,float(y.max()))); ax[1,0].set_title('标签：响应带',fontproperties=FONT)
                ax[1,1].imshow(core,aspect='auto',origin='upper',cmap='viridis',vmin=0,vmax=1); ax[1,1].set_title('核心窄带',fontproperties=FONT)
                ax[1,2].imshow(pred,aspect='auto',origin='upper',cmap='viridis',vmin=0,vmax=max(0.6,float(pred.max()))); ax[1,2].set_title('mask 概率',fontproperties=FONT)
                env=np.abs(G); env=env/(np.nanmax(env)+1e-6); ax[1,3].imshow(env,aspect='auto',origin='upper',cmap='magma',vmin=0,vmax=1); ax[1,3].set_title('|G_hat| envelope',fontproperties=FONT)
                ax[1,4].imshow(raw,aspect='auto',origin='upper',cmap='gray'); ax[1,4].imshow(pred,aspect='auto',origin='upper',cmap='magma',alpha=np.clip(pred*0.85,0,0.65)); ax[1,4].set_title('叠加图',fontproperties=FONT)
                if gate_np is not None:
                    names=['A gate','S gate','G gate']
                    for gi in range(3):
                        ax[2,gi].imshow(gate_np[gi],aspect='auto',origin='upper',cmap='viridis',vmin=0,vmax=1); ax[2,gi].set_title(names[gi],fontproperties=FONT)
                    ax[2,3].plot(gate_np.mean(axis=(1,2))); ax[2,3].set_ylim(0,1); ax[2,3].set_title('gate mean',fontproperties=FONT)
                    ax[2,4].axis('off')
                for a in ax.ravel(): a.set_xlabel('道号 / trace',fontproperties=FONT); a.set_ylabel('采样点 / sample',fontproperties=FONT)
            else:
                fig,ax=plt.subplots(1,5,figsize=(18,4),constrained_layout=True)
                ax[0].imshow(raw,aspect='auto',origin='upper',cmap='gray'); ax[0].set_title('输入：原始 raw',fontproperties=FONT)
                ax[1].imshow(y,aspect='auto',origin='upper',cmap='viridis',vmin=0,vmax=max(0.6,float(y.max()))); ax[1].set_title('验证标签：响应带',fontproperties=FONT)
                ax[2].imshow(core,aspect='auto',origin='upper',cmap='viridis',vmin=0,vmax=1); ax[2].set_title('辅助监督：核心窄带',fontproperties=FONT)
                ax[3].imshow(pred,aspect='auto',origin='upper',cmap='viridis',vmin=0,vmax=max(0.6,float(pred.max()))); ax[3].set_title('验证预测：响应带概率',fontproperties=FONT)
                ax[4].imshow(raw,aspect='auto',origin='upper',cmap='gray'); ax[4].imshow(pred,aspect='auto',origin='upper',cmap='magma',alpha=np.clip(pred*0.85,0,0.65)); ax[4].set_title('验证叠加图',fontproperties=FONT)
                for a in ax: a.set_xlabel('道号 / trace',fontproperties=FONT); a.set_ylabel('采样点 / sample',fontproperties=FONT)
            fig.savefig(run_dir/'previews'/f'{prefix}_{b["id"][0]}_preview.png',dpi=140); plt.close(fig)
if __name__=='__main__': run(sys.argv[1] if len(sys.argv)>1 else str(ROOT/'configs/fast_cpu_check.json'))
