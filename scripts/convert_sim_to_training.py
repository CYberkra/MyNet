"""
Convert gprMax simulation outputs to training-compatible .npz windows.
Preprocessing now matches REAL data: P99 normalization (no compress_raw!).
"""
import h5py, numpy as np, csv
from pathlib import Path
from scipy.ndimage import zoom

OUT = Path(r'D:\Claude\PGDA-CSNet\workspace\transfer_20260627_142748\PGDA-CSNet_transfer_bundle_20260627_142748\PGDA_CSNet_v0_9_6_SEARCH_WINDOW_GUARD\data\simulation_pretrain_v1\windows')
OUT.mkdir(parents=True, exist_ok=True)

def convert(bg_file, basal_file, case_id, n_windows=12):
    if not bg_file.exists() or not basal_file.exists(): return []
    with h5py.File(bg_file, 'r') as f: bg = f['rxs/rx1/Ez'][:].astype(np.float32)
    with h5py.File(basal_file, 'r') as f: basal = f['rxs/rx1/Ez'][:].astype(np.float32)
    diff = basal - bg; nt, nx = bg.shape
    factor = 501 / nt
    raw = zoom(basal, (factor, 1.0), order=1).astype(np.float32)
    df = zoom(diff, (factor, 1.0), order=1).astype(np.float32)

    # KEY: Same P99 normalization as real data
    p99 = np.percentile(np.abs(raw), 99) + 1e-8
    raw = raw / p99
    mask = np.abs(df) / (np.abs(df).max() + 1e-8)

    rows = []
    for w in range(n_windows):
        if nx >= 256:
            start = np.random.randint(0, nx - 255)
            rw = raw[:, start:start+256]
            mw = mask[:, start:start+256]
        else:
            pad = (256 - nx) // 2
            rw = np.pad(raw, ((0,0),(pad, 256-nx-pad)), mode='reflect')
            mw = np.pad(mask, ((0,0),(pad, 256-nx-pad)), mode='reflect')
        gi = (mw.max(axis=0) > 0.05).astype(np.int16)
        sid = f"{case_id}_w{w:02d}"
        np.savez_compressed(OUT / f"{sid}.npz",
            x_raw=rw.astype(np.float32), y_mask=mw.astype(np.float32),
            status_code=np.where(gi,1,0).astype(np.int16),
            label_weight=np.maximum(mw.max(axis=0),0.3).astype(np.float32))
        rows.append({"sample_id":sid,"line":case_id,"start":0,"end":255,
                     "split":"train","present":int(gi.sum()),"weak":0,"no_pick":int((1-gi).sum())})
    return rows

np.random.seed(42)
v74 = Path(r'D:\Claude\PGDA-CSNet\workspace\transfer_20260627_142748\PGDA-CSNet_transfer_bundle_20260627_142748\PGDA_gprMax_v0_7_4_RELEASE_QC_LOCKED\PGDA_gprMax_v0_7_4_RELEASE_QC_LOCKED\outputs_gprmax')
v753 = Path(r'D:\Claude\PGDA-CSNet\workspace\transfer_20260627_142748\PGDA-CSNet_transfer_bundle_20260627_142748\PGDA_gprMax_v0_7_5_3_SUPER_SMOOTH_REAL_GPRMAX_GEOMETRY\PGDA_gprMax_v0_7_5_3_SUPER_SMOOTH_REAL_GPRMAX_GEOMETRY\outputs_gprmax')
v751 = Path(r'D:\Claude\PGDA-CSNet\workspace\transfer_20260627_142748\PGDA-CSNet_transfer_bundle_20260627_142748\PGDA_gprMax_v0_7_5_1_TERRAIN_X_2P5D_AUDIT_FIXED\PGDA_gprMax_v0_7_5_1_TERRAIN_X_2P5D_AUDIT_FIXED\outputs_gprmax')

pairs = [
    (v74,'ST01_line9_zk08_background_no_basal','ST02_line9_zk08_basal_present','st_l9'),
    (v74,'ST04_line3_zk07_zk08_background_no_basal','ST05_line3_zk07_zk08_basal_present','st_l3'),
    (v74,'ST06_zk09_line6_background_no_basal','ST07_zk09_line6_basal_present','st_l6'),
    (v74,'ST08_l1_12_20_background_no_basal','ST09_l1_12_20_basal_present','st_l1'),
    (v753,'TXSS2D01_supersmooth_terrain_background_no_basal','TXSS2D02_supersmooth_terrain_basal_present','ss'),
    (v751,'TX2D01_box_terrain_background_no_basal','TX2D04_box_terrain_basal_present','bx'),
    (v751,'TX2D02_smooth_terrain_background_no_basal','TX2D05_smooth_terrain_basal_present','sm'),
    (v751,'TX2D03_limited_terrain_background_no_basal','TX2D06_limited_terrain_basal_present','lm'),
]

all_rows = []
for src, bg, ba, cid in pairs:
    r = convert(src/f'{bg}_merged.out', src/f'{ba}_merged.out', cid, 12)
    all_rows.extend(r)
    print(f'{cid}: {len(r)} windows (P99 norm)')

idx = OUT.parent / "window_index.csv"
with idx.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["sample_id","line","start","end","split","present","weak","no_pick"])
    w.writeheader()
    w.writerows(all_rows)
print(f'\nTotal: {len(all_rows)} windows with P99 normalization')
print(f'This matches real data preprocessing!')
