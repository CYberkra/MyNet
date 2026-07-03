"""Resume training from checkpoint_last.pt in run_dir."""
import sys, json, random
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_raw_only import DS, run_epoch, env_info, build_model, preview, unpack_model_output
from torch.utils.data import ConcatDataset, DataLoader

config_path = sys.argv[1]
with open(config_path) as f:
    cfg = json.load(f)

device = torch.device('cuda' if torch.cuda.is_available() and not cfg.get('force_cpu', False) else 'cpu')
run_dir = ROOT / cfg['run_dir']
(run_dir / 'previews').mkdir(parents=True, exist_ok=True)
(run_dir / 'logs').mkdir(exist_ok=True)

# Load checkpoint
ckpt_path = run_dir / 'checkpoint_last.pt'
if not ckpt_path.exists():
    print(f"No checkpoint found at {ckpt_path}, starting from scratch.")
    ckpt = None
else:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"Loaded checkpoint: epoch {ckpt.get('epoch', 0)}, train_loss={ckpt.get('history', [{}])[-1].get('train_loss', 'N/A') if ckpt.get('history') else 'N/A'}")

# Build model and load state
model = build_model(cfg).to(device)
if ckpt:
    model.load_state_dict(ckpt['model'])
    print(f"Model state loaded ({len(ckpt['model'])} keys)")

# Set seed for resumption (don't re-seed — use epoch-based deterministic order for remaining epochs)
# Use a deterministic approach: seed = cfg['seed'] + resume_epoch to keep reproducibility
resume_epoch = ckpt['epoch'] if ckpt else 0

# Optimizer
opt = torch.optim.AdamW([
    {"params": [p for n, p in model.named_parameters() if "sgm" in n],
     "lr": cfg["lr"] * 0.5, "weight_decay": float(cfg.get("sgm_weight_decay", 0.05))},
    {"params": [p for n, p in model.named_parameters() if "sgm" not in n],
     "lr": cfg["lr"], "weight_decay": float(cfg.get("weight_decay", 1e-4))},
])

# Data
data_root = Path(cfg.get('data_root', 'data'))
if not data_root.is_absolute():
    data_root = ROOT / data_root

train_real = DataLoader(DS('train', cfg), batch_size=cfg['batch_size'], shuffle=True,
                        num_workers=int(cfg.get('num_workers', 0)))
train_sim = None
sim_ratio = float(cfg.get('sim_batch_ratio', 0.0))
if sim_ratio > 0 and cfg.get('sim_data_root'):
    sim_cfg = cfg.copy()
    sim_cfg['data_root'] = cfg['sim_data_root']
    sim_cfg['train_lines'] = cfg.get('sim_train_lines', [])
    if not sim_cfg['train_lines']:
        sim_idx = (ROOT / sim_cfg['data_root'] if not Path(sim_cfg['data_root']).is_absolute()
                   else Path(sim_cfg['data_root'])) / 'window_index.csv'
        if sim_idx.exists():
            import csv
            sim_lines = set()
            with open(sim_idx, encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    sim_lines.add(row['line'])
            sim_cfg['train_lines'] = list(sim_lines)
    if sim_cfg['train_lines']:
        train_sim = DataLoader(DS('train', sim_cfg), batch_size=cfg['batch_size'],
                               shuffle=True, num_workers=0)
        print(f'SIM_DATA: {len(sim_cfg["train_lines"])} lines, {len(train_sim.dataset)} samples')

val_ds = DS('val', cfg)
val = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

if train_sim:
    combined = ConcatDataset([train_real.dataset, train_sim.dataset])
    train = DataLoader(combined, batch_size=cfg['batch_size'], shuffle=True, num_workers=0)
    print(f'TRAIN combined: {len(combined)} samples (real={len(train_real.dataset)}, sim={len(train_sim.dataset)})')
else:
    train = train_real

# Restore history
hist = ckpt['history'] if ckpt and 'history' in ckpt else []
best_val = 1e9
best_epoch = 0

# Restore best_val from history
for rec in hist:
    monitor = rec.get('val_loss', float('nan'))
    if np.isfinite(monitor) and monitor < best_val:
        best_val = monitor
        best_epoch = rec['epoch']

print(f"Resuming from epoch {resume_epoch}/{cfg['epochs']}, best_val={best_val:.4f} @ epoch {best_epoch}")

for ep in range(resume_epoch + 1, cfg['epochs'] + 1):
    tr = run_epoch(model, train, device, cfg, opt)
    va = run_epoch(model, val, device, cfg, None) if len(val_ds) > 0 else {'loss': float('nan')}
    va = run_epoch(model, val, device, cfg, None) if len(val_ds) > 0 else {'loss': float('nan')}
    rec = {'epoch': ep, 'device': str(device)}
    rec.update({f'train_{k}': v for k, v in tr.items()})
    rec.update({f'val_{k}': v for k, v in va.items()})
    hist.append(rec)
    print(rec, flush=True)

    torch.save({'model': model.state_dict(), 'cfg': cfg, 'history': hist,
                'env': env_info(device), 'epoch': ep}, run_dir / 'checkpoint_last.pt')

    monitor = va.get('loss', float('nan')) if len(val_ds) > 0 else tr.get('loss', float('nan'))
    if np.isfinite(monitor) and monitor < best_val:
        best_val = monitor
        best_epoch = ep
        torch.save({'model': model.state_dict(), 'cfg': cfg, 'history': hist,
                    'env': env_info(device), 'epoch': ep,
                    'best_monitor_loss': best_val,
                    'monitor': 'val_loss' if len(val_ds) > 0 else 'train_loss'},
                   run_dir / 'checkpoint_best.pt')

json.dump({'best_epoch': best_epoch, 'best_monitor_loss': best_val,
           'monitor': 'val_loss' if len(val_ds) > 0 else 'train_loss',
           'history': hist},
          open(run_dir / 'history.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

if (run_dir / 'checkpoint_best.pt').exists():
    ckpt_best = torch.load(run_dir / 'checkpoint_best.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt_best['model'])

if len(val_ds) > 0:
    preview(model, DataLoader(DS('val', cfg), batch_size=1, shuffle=False),
            device, run_dir, 'val', max_items=int(cfg.get('max_preview_val', 4)))

print(f"Training complete. Best epoch: {best_epoch}, best monitor: {best_val:.4f}")
