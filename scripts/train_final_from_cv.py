from pathlib import Path
import csv,json,statistics,subprocess,sys,shutil
ROOT=Path(__file__).resolve().parents[1]
cv=ROOT/'reports/cv_summary.csv'
if not cv.exists():
    print('缺少 reports/cv_summary.csv，请先运行 02_gpu_train_cv_medium。', file=sys.stderr); sys.exit(1)
rows=list(csv.DictReader(open(cv,encoding='utf-8')))
required={'outputs/run_gpu_fold0_medium','outputs/run_gpu_fold1_medium','outputs/run_gpu_fold2_medium'}
seen={r.get('run_dir','').replace('\\','/') for r in rows}
missing=required-seen
if missing:
    print(f'CV 结果不完整，缺少: {sorted(missing)}', file=sys.stderr); sys.exit(1)
eps=[]
for r in rows:
    try:
        if r.get('best_epoch'): eps.append(int(float(r['best_epoch'])))
    except Exception: pass
if len(eps)<3:
    print('cv_summary.csv 中 best_epoch 少于 3 个，不能锁定 final 训练。', file=sys.stderr); sys.exit(1)
tpl=json.load(open(ROOT/'configs/gpu_train_final_locked_template.json',encoding='utf-8'))
median_epoch=max(1,int(round(statistics.median(eps))))
min_ep=int(tpl.get('min_final_epochs',1)); max_ep=int(tpl.get('max_final_epochs',median_epoch))
final_epochs=max(min_ep,min(max_ep,median_epoch))
tpl['epochs']=final_epochs
tpl['final_locked']=True
tpl['cv_best_epochs']=eps
tpl['cv_median_best_epoch']=median_epoch
tpl['final_epoch_rule']=f'clamp(median_best_epoch, min={min_ep}, max={max_ep})'
out_cfg=ROOT/'configs/_runtime_gpu_train_final_locked.json'
json.dump(tpl,open(out_cfg,'w',encoding='utf-8'),ensure_ascii=False,indent=2)
print(f'CV_BEST_EPOCHS={eps}')
print(f'CV_MEDIAN_BEST_EPOCH={median_epoch}')
print(f'FINAL_LOCKED_EPOCHS={final_epochs}')
subprocess.check_call([sys.executable, str(ROOT/'scripts/train_raw_only.py'), str(out_cfg)])
run_dir=ROOT/tpl['run_dir']
last=run_dir/'checkpoint_last.pt'
final=run_dir/'checkpoint_final.pt'
if last.exists():
    shutil.copy2(last, final)
    print(f'FINAL_CHECKPOINT={final}')
else:
    print('未找到 checkpoint_last.pt，final checkpoint 未生成。', file=sys.stderr); sys.exit(1)
