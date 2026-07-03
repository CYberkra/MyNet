from pathlib import Path
import json,csv,sys,math
ROOT=Path(__file__).resolve().parents[1]
run_dirs=sys.argv[1:] or ['outputs/run_gpu_fold0_medium','outputs/run_gpu_fold1_medium','outputs/run_gpu_fold2_medium']
out=ROOT/'reports/cv_summary.csv'
with open(out,'w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['run_dir','best_epoch','best_val_loss','train_lines','val_lines'])
    for rd in run_dirs:
        hp=ROOT/rd/'history.json'; cp=ROOT/rd/'used_config.json'
        if not hp.exists(): continue
        h=json.load(open(hp,encoding='utf-8')); cfg=json.load(open(cp,encoding='utf-8')) if cp.exists() else {}
        w.writerow([rd,h.get('best_epoch'),h.get('best_val_loss', h.get('best_monitor_loss')),';'.join(cfg.get('train_lines',[])),';'.join(cfg.get('val_lines',[]))])
print(out)
