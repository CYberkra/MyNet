from pathlib import Path
import hashlib, csv
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'reports/release_manifest_sha256.csv'
rows=[]
for p in sorted(ROOT.rglob('*')):
    if not p.is_file(): continue
    rel=p.relative_to(ROOT).as_posix()
    if rel.startswith('outputs/') or rel=='reports/release_manifest_sha256.csv': continue
    h=hashlib.sha256(p.read_bytes()).hexdigest()
    rows.append((rel,p.stat().st_size,h))
out.parent.mkdir(exist_ok=True)
with open(out,'w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['path','size_bytes','sha256']); w.writerows(rows)
print(out)
