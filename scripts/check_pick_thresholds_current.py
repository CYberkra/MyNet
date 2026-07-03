from pathlib import Path
import json,hashlib,sys
ROOT=Path(__file__).resolve().parents[1]

def short_sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()[:12]

p=ROOT/'reports/pick_thresholds.json'
if not p.exists():
    print('缺少 reports/pick_thresholds.json，请先运行 04_calibrate_pick_thresholds_from_cv',file=sys.stderr); sys.exit(1)
obj=json.load(open(p,encoding='utf-8'))
BAD=0
for rec in obj.get('fold_hashes',[]):
    rd=rec['run_dir']; line=rec['line']
    ck=ROOT/rd/'checkpoint_best.pt'; lf=ROOT/'data/lines'/f'{line}.npz'
    if not ck.exists(): print(f'缺少 {ck}',file=sys.stderr); BAD+=1; continue
    ch=short_sha256(ck); lh=short_sha256(lf)
    if ch!=rec.get('checkpoint_sha256_12'):
        print(f'阈值过期: {rd} checkpoint hash 当前 {ch} != 校准 {rec.get("checkpoint_sha256_12")}',file=sys.stderr); BAD+=1
    if lh!=rec.get('line_sha256_12'):
        print(f'阈值过期: {line} data hash 当前 {lh} != 校准 {rec.get("line_sha256_12")}',file=sys.stderr); BAD+=1
if BAD:
    print('PICK_THRESHOLDS_STALE，请重新运行 04_calibrate_pick_thresholds_from_cv',file=sys.stderr); sys.exit(1)
print('PICK_THRESHOLDS_CURRENT_OK')
