from pathlib import Path
import json, re, sys
ROOT=Path(__file__).resolve().parents[1]
BAD=0
def fail(m):
    global BAD
    BAD+=1; print('SEARCH_WINDOW_BAD:',m)
# eval defaults
text=(ROOT/'scripts/eval_full_line.py').read_text(encoding='utf-8')
if "default=320.0" not in text or "default=560.0" not in text:
    fail('eval_full_line.py default search window should be 320-560 ns')
# calibration defaults
ctext=(ROOT/'scripts/calibrate_pick_thresholds_from_cv.py').read_text(encoding='utf-8')
if "search_min_ns=320.0" not in ctext or "search_max_ns=560.0" not in ctext:
    fail('calibration default search window should be 320-560 ns')
# if thresholds already exist, guard them
p=ROOT/'reports/pick_thresholds.json'
if p.exists():
    j=json.load(open(p,encoding='utf-8'))
    if float(j.get('search_min_ns',0)) < 320:
        fail('pick_thresholds.json search_min_ns too shallow; rerun calibration')
    if float(j.get('search_max_ns',9999)) > 600:
        fail('pick_thresholds.json search_max_ns too deep/broad; rerun calibration')
if BAD==0:
    print('SEARCH_WINDOW_POLICY_OK')
sys.exit(1 if BAD else 0)
