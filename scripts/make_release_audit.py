from pathlib import Path
import csv, json
ROOT=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader(open(ROOT/'data/window_index.csv',encoding='utf-8')))
with open(ROOT/'reports/data_split_audit.csv','w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['legacy_split','line','window_count','present_sum','weak_sum','no_pick_sum'])
    for key in sorted(set((r['split'],r['line']) for r in rows)):
        rr=[r for r in rows if (r['split'],r['line'])==key]
        w.writerow([key[0],key[1],len(rr),sum(int(r['present']) for r in rr),sum(int(r['weak']) for r in rr),sum(int(r['no_pick']) for r in rr)])
with open(ROOT/'reports/cv_split_policy.csv','w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['stage','fold','train_lines','val_lines','test_lines','review_lines','note'])
    w.writerow(['cv','fold0','Line3;LineL1','Line7','Line9','Line6','Line9 not used for model selection'])
    w.writerow(['cv','fold1','Line7;LineL1','Line3','Line9','Line6','Line9 not used for model selection'])
    w.writerow(['cv','fold2','Line3;Line7','LineL1','Line9','Line6','Line9 not used for model selection'])
    w.writerow(['final_locked','final','Line3;Line7;LineL1','','Line9','Line6','epochs from CV median best_epoch, clamped'])
print(ROOT/'reports/data_split_audit.csv')
print(ROOT/'reports/cv_split_policy.csv')
