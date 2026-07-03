from pathlib import Path
import sys,json,csv,itertools,hashlib
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from eval_full_line import stitch_one, centerline, dp_ridge_centerline

def short_sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()[:12]

def binary_metrics(pred_bool, pos_mask, neg_mask):
    pred_bool=pred_bool.astype(bool)
    pos_mask=pos_mask.astype(bool); neg_mask=neg_mask.astype(bool)
    tp=np.logical_and(pred_bool,pos_mask).sum()
    fn=np.logical_and(~pred_bool,pos_mask).sum()
    fp=np.logical_and(pred_bool,neg_mask).sum()
    tn=np.logical_and(~pred_bool,neg_mask).sum()
    precision=float(tp/max(tp+fp,1))
    recall=float(tp/max(tp+fn,1))
    specificity=float(tn/max(tn+fp,1)) if (tn+fp)>0 else 1.0
    false_pick=float(fp/max(fp+tn,1)) if (fp+tn)>0 else 0.0
    f1=float(2*precision*recall/max(precision+recall,1e-6))
    return dict(f1=f1,precision=precision,recall=recall,specificity_no_pick=specificity,false_pick_rate_no_pick=false_pick,tp=int(tp),fp=int(fp),fn=int(fn),tn=int(tn))

def load_or_predict_cache(run_dir,line_name):
    cache=ROOT/'reports/calibration_cache'; cache.mkdir(parents=True,exist_ok=True)
    ckpt=ROOT/run_dir/'checkpoint_best.pt'
    if not ckpt.exists():
        print(f'缺少 {ckpt}', file=sys.stderr); sys.exit(1)
    ckpt_hash=short_sha256(ckpt)
    line_file=ROOT/'data/lines'/f'{line_name}.npz'
    line_hash=short_sha256(line_file)
    stem=f'{Path(run_dir).name}_{line_name}_{ckpt_hash}_{line_hash}'
    pred_p=cache/f'{stem}_pred.npy'; pres_p=cache/f'{stem}_presence.npy'; meta_p=cache/f'{stem}_meta.json'
    if pred_p.exists() and pres_p.exists() and meta_p.exists():
        print(f'使用校准缓存: {stem}', flush=True)
        return np.load(pred_p), np.load(pres_p)
    print(f'生成校准预测缓存: {run_dir} -> {line_name} ckpt={ckpt_hash} line={line_hash}', flush=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred,pres,cfg=stitch_one(Path(run_dir),line_name,'best',device)
    np.save(pred_p,pred); np.save(pres_p,pres)
    json.dump({'run_dir':run_dir,'line':line_name,'checkpoint':'checkpoint_best.pt','checkpoint_sha256_12':ckpt_hash,'line_sha256_12':line_hash,'pred_file':pred_p.name,'presence_file':pres_p.name},open(meta_p,'w',encoding='utf-8'),ensure_ascii=False,indent=2)
    return pred,pres

def eval_cached(pred,pres,line_name,pres_thr,path_thr,search_min_ns=320.0,search_max_ns=560.0,dp_cache=None):
    line=np.load(ROOT/'data/lines'/f'{line_name}.npz')
    gt=line['soft_mask_train'].astype(np.float32); status=line['status_code'].astype(np.int16); dt=float(line['dt_ns'])
    search_min=int(round(search_min_ns/dt)); search_max=int(round(search_max_ns/dt))
    key=(line_name,float(pres_thr))
    if dp_cache is not None and key in dp_cache:
        cdp,vdp=dp_cache[key]
    else:
        cdp,vdp=dp_ridge_centerline(pred,max_jump=8,smooth_weight=0.08,min_presence=(pres>=pres_thr),search_min_sample=search_min,search_max_sample=search_max)
        if dp_cache is not None: dp_cache[key]=(cdp,vdp)
    H,W=pred.shape; path_prob=np.full(W,np.nan,np.float32); final=np.zeros(W,dtype=bool)
    for i in range(W):
        if vdp[i] and np.isfinite(cdp[i]):
            yi=int(np.clip(round(float(cdp[i])),0,H-1)); path_prob[i]=pred[yi,i]; final[i]=path_prob[i]>=path_thr
    # Calibration discipline:
    #   present(1) = positive, no_pick(0) = hard negative, weak_visible(2) = ignored for threshold selection.
    # This prevents weak traces from forcing very low thresholds and reduces Line6/uncertain false picking.
    pos=status==1
    neg=status==0
    m=binary_metrics(final,pos,neg)
    cgt,vgt=centerline(gt,1e-3); both=final & vgt & np.isfinite(cdp)
    # Missed positives are already represented by recall. Use the bounded search
    # width here so an empty match cannot dominate the entire threshold score.
    mae=float(np.nanmean(np.abs(cdp[both]-cgt[both]))) if both.any() else float(search_max-search_min)
    pick_rate=float(final.mean())
    # Penalize no-pick false picking strongly; Line9 is not used here.
    score=(0.25*m['f1'] + 0.15*m['precision'] + 0.35*m['specificity_no_pick'] + 0.15*m['recall']
           -0.0005*mae -0.25*m['false_pick_rate_no_pick'])
    return {'line':line_name,'presence_thr':pres_thr,'path_prob_thr':path_thr,**m,'mae_sample':mae,'pick_rate':pick_rate,'score':float(score)}

def main():
    folds=[
        ('outputs/run_gpu_fold0_medium','Line7'),
        ('outputs/run_gpu_fold1_medium','Line3'),
        ('outputs/run_gpu_fold2_medium','LineL1'),
    ]
    for rd,_ in folds:
        if not (ROOT/rd/'checkpoint_best.pt').exists():
            print(f'缺少 {rd}/checkpoint_best.pt，请先跑完 02_gpu_train_cv_medium',file=sys.stderr); sys.exit(1)
    fold_arrays=[]; fold_hashes=[]
    for rd,line in folds:
        pred,pres=load_or_predict_cache(rd,line)
        ckpt_hash=short_sha256(ROOT/rd/'checkpoint_best.pt')
        line_hash=short_sha256(ROOT/'data/lines'/f'{line}.npz')
        fold_hashes.append({'run_dir':rd,'line':line,'checkpoint_sha256_12':ckpt_hash,'line_sha256_12':line_hash})
        fold_arrays.append((rd,line,pred,pres))
    pres_grid=[0.30,0.40,0.50,0.55,0.60,0.65,0.70,0.80,0.90,0.95]
    path_grid=[0.10,0.20,0.30,0.40,0.45,0.50,0.55,0.60,0.70,0.80]
    rows=[]; best=None; dp_cache={}
    keys=['f1','precision','recall','specificity_no_pick','false_pick_rate_no_pick','mae_sample','pick_rate','score']
    for pt,qt in itertools.product(pres_grid,path_grid):
        rs=[eval_cached(pred,pres,line,pt,qt,dp_cache=dp_cache) for rd,line,pred,pres in fold_arrays]
        avg={k:float(np.mean([r[k] for r in rs])) for k in keys}
        rec={'presence_thr':pt,'path_prob_thr':qt,**{f'avg_{k}':v for k,v in avg.items()}}
        rows.append(rec)
        if best is None or rec['avg_score']>best['avg_score']: best=rec
    out=ROOT/'reports'; out.mkdir(exist_ok=True)
    with open(out/'pick_threshold_calibration.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    result={'presence_thr':best['presence_thr'],'path_prob_thr':best['path_prob_thr'],'search_min_ns':320.0,'search_max_ns':560.0,'source':'cv_validation_grid_search_hashed_cache_specificity_penalized_v096_search_window_guard','best':best,'cache_dir':'reports/calibration_cache','fold_hashes':fold_hashes,'selection_note':'present=positive; no_pick=hard negative; weak_visible ignored for threshold selection to avoid low thresholds driven by uncertain traces'}
    json.dump(result,open(out/'pick_thresholds.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
    print(out/'pick_threshold_calibration.csv')
    print(out/'pick_thresholds.json')
    print(result)
if __name__=='__main__': main()
