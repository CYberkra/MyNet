from pathlib import Path
import json, hashlib, sys, re, csv, numpy as np
ROOT=Path(__file__).resolve().parents[1]
BAD=0

def fail(msg):
    global BAD
    print('RELEASE_BAD:', msg); BAD+=1

def ok(msg): print('OK:', msg)
# 1. Source package guard: ignore outputs because this check may also run after training.
#    Checkpoints are allowed under outputs/, but not in source/config/data/script directories.
for p in ROOT.rglob('*.pt'):
    rel=p.relative_to(ROOT).as_posix()
    if not rel.startswith('outputs/'):
        fail(f'checkpoint should not be packaged outside outputs: {rel}')
# 2. Required scripts
required=[
'00_check_raw_only_schema.sh','01_fast_cpu_check_raw_only.sh','02_gpu_train_cv_medium.sh','03_train_final_locked_from_cv.sh','04_calibrate_pick_thresholds_from_cv.sh','05_eval_line9_final_locked.sh','06_eval_line6_review_final_locked.sh',
'scripts/check_dataset.py','scripts/check_configs.py','scripts/check_search_window_policy.py','scripts/train_raw_only.py','scripts/train_final_from_cv.py','scripts/eval_full_line.py'
]
for r in required:
    if not (ROOT/r).exists(): fail(f'missing {r}')
# 3. Config guardrails
for p in (ROOT/'configs').glob('*.json'):
    if p.name.startswith('_runtime'): continue
    cfg=json.load(open(p,encoding='utf-8'))
    if p.name.startswith('gpu_train_fold') or p.name.startswith('gpu_loss') or p.name.startswith('gpu_train_final'):
        if cfg.get('height_resize')!=512: fail(f'{p.name} height_resize should be 512')
        if cfg.get('width_resize')!=256: fail(f'{p.name} width_resize should be 256')
        if not cfg.get('augment',{}).get('enabled',False): fail(f'{p.name} should enable train-only raw augmentation')
        loss=cfg.get('loss',{})
        if float(loss.get('positive_pixel_boost',0)) < 4.0: fail(f'{p.name} missing v0.9.5 positive_pixel_boost')
        if float(loss.get('hard_negative_weight',0)) < 0.30: fail(f'{p.name} missing v0.9.5 hard_negative_weight')
        if float(loss.get('outside_weight',0)) < 0.55: fail(f'{p.name} outside_weight too low for v0.9.5 false-positive fix')
    tr=set(cfg.get('train_lines',[])); va=set(cfg.get('val_lines',[]))
    if (tr|va)&{'Line6','LineX1'}: fail(f'{p.name} has forbidden train/val line: {sorted((tr|va)&{"Line6","LineX1"})}')
    if 'Line9' in va: fail(f'{p.name} cannot use Line9 for validation')
    if 'Line9' in tr:
        if cfg.get('train_trace_ranges',{}).get('Line9') != [0,1407] or cfg.get('test_trace_ranges',{}).get('Line9') != [1664,2377]:
            fail(f'{p.name} violates locked Line9 trace ranges')
# 4. Raw-only window schema
for p in (ROOT/'data/windows').glob('*.npz'):
    z=np.load(p,allow_pickle=True); keys=set(z.files)
    if 'x_raw' not in keys: fail(f'{p.name} missing x_raw')
    for k in keys:
        if re.search(r'bg501|agc|processed|background|teacher|target_only', k, re.I): fail(f'{p.name} forbidden key {k}')
# 5. Training code should not read qa_views or forbidden processed inputs
train_text=(ROOT/'scripts/train_raw_only.py').read_text(encoding='utf-8')
if 'qa_views' in train_text: fail('train_raw_only.py references qa_views')
for term in ['bg501','agc9','processed_view','target_only','response_teacher']:
    if term in train_text.lower(): fail(f'train_raw_only.py references forbidden term {term}')
# 6. CV/final policy report exists or can be generated later
if not (ROOT/'reports').exists(): fail('missing reports directory')
# 7. Legacy split is only metadata; actual fold configs exist
for name in ['gpu_train_fold0_medium.json','gpu_train_fold1_medium.json','gpu_train_fold2_medium.json','gpu_train_final_locked_template.json']:
    if not (ROOT/'configs'/name).exists(): fail(f'missing config {name}')

# 8. Search-window guard: prevent DP/post-processing from falling back to the shallow 300 ns window.
eval_text=(ROOT/'scripts/eval_full_line.py').read_text(encoding='utf-8')
cal_text=(ROOT/'scripts/calibrate_pick_thresholds_from_cv.py').read_text(encoding='utf-8')
if 'default=320.0' not in eval_text or 'default=560.0' not in eval_text:
    fail('eval_full_line.py should default to 320-560 ns DP search window')
if 'search_min_ns=320.0' not in cal_text or 'search_max_ns=560.0' not in cal_text:
    fail('calibrate_pick_thresholds_from_cv.py should default to 320-560 ns')
if BAD==0:
    ok('RELEASE_INTEGRITY_OK')
sys.exit(1 if BAD else 0)
