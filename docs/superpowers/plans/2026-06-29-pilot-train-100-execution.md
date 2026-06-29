# Pilot-Train 100 场景仿真执行计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成 100 场景 gprMax 仿真数据，转换为 ~200 个训练窗口（vs 当前 20），用新数据重新训练 LOLO-CV Line9 fold 验证效果。

**Architecture:** 基于 SimLab + gprMax 管线逐步推进：配置验证 → 场景生成 → 几何预检 → GPU 批量仿真 → 后处理合并 → NPZ 转换 → LOLO-CV 训练验证。

**Tech Stack:** gprMax v3.1.7, uavgpr_simlab, RTX 3060 6GB, Python 3.10

**设计文档:** `docs/superpowers/specs/2026-06-29-pilot-train-100-design.md`

## Global Constraints

- 使用 `E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe` 作为 Python 解释器
- 不使用 `E:\python\python.exe`（系统 Python，torch.cuda=False）
- FDTD 参数锁定：dx=0.05m, 700ns, 100MHz Ricker, 12m 飞行高度
- 仿真变体只跑 raw + target_only + background_only（不跑 air_only）
- 外部杂波不做（电线/树木/建筑关闭）
- 输出路径：`uavgpr_simlab/workspace/pilot_train_v1/`

---

### Task 1: 验证并更新 run_plan 配置

**Files:**
- Modify: `uavgpr_simlab/configs/run_plan_3060_pilot_train_v1.yaml`
- Create: `uavgpr_simlab/configs/environment_3060_laptop.yaml`（如缺失则创建）

**Interfaces:**
- Consumes: 设计文档配置参数
- Produces: 验证通过的 run_plan YAML

- [ ] **Step 1: 检查 run_plan YAML 的 scene 参数是否与设计一致**

当前 `run_plan_3060_pilot_train_v1.yaml`：
```yaml
scene:
  trace_count: 128
  trace_interval_m: 0.50
  scan_length_m: 64
  dx_m: 0.05
  time_window_ns: 700
  center_frequency_hz: 100000000
  flight_height_m: 12
  samples: 501
  domain_depth_m: 28
```

验证逻辑：`trace_count × trace_interval_m = 64m = scan_length_m` ✅

- [ ] **Step 2: 检查 environment YAML 是否存在，缺失则创建**

创建 `uavgpr_simlab/configs/environment_3060_laptop.yaml`：
```yaml
environment:
  name: 3060_laptop
  description: RTX 3060 Laptop 6GB
run:
  gpu_ids: [0]
  python_exe: "E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe"
install:
  gprmax_root: "E:/gprMax/gprMax-v.3.1.7"
  msvc_env: "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build"
```

- [ ] **Step 3: 打印确认配置摘要**

```bash
python3 -c "
import yaml
p = yaml.safe_load(open('uavgpr_simlab/configs/run_plan_3060_pilot_train_v1.yaml'))
print(f'Scenes: {p[\"scene_count\"]}')
print(f'Traces: {p[\"scene\"][\"trace_count\"]} × {p[\"scene\"][\"trace_interval_m\"]}m')
print(f'Domain: {p[\"scene\"][\"scan_length_m\"]}m × {p[\"scene\"][\"domain_depth_m\"]}m')
print(f'Families: {p[\"geology\"][\"scenario_family\"]}')
print(f'Variants: {p[\"components\"]}')
print(f'Randomization: {p[\"domain_randomization\"][\"profile\"]}')
"
```

预期输出：所有参数与设计文档一致。

- [ ] **Step 4: 提交**

```bash
git add uavgpr_simlab/configs/run_plan_3060_pilot_train_v1.yaml
git add uavgpr_simlab/configs/environment_3060_laptop.yaml
git commit -m "config: verify Pilot-Train run_plan params"
```

---

### Task 2: 生成 100 场景并几何预检

**Files:**
- Run: `uavgpr_simlab` CLI generate command
- Verify: `uavgpr_simlab/workspace/pilot_train_v1/sceneworld/` 输出

**Interfaces:**
- Consumes: Task 1 的 run_plan YAML
- Produces: 100 个 case 目录 + 每个目录 `.in` 文件 + `scene_world.json`

- [ ] **Step 1: 用 SimLab CLI 生成 100 场景（几何和 .in 文件，不跑仿真）**

```bash
"E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe" -m uavgpr_simlab.cli generate \
  --plan "uavgpr_simlab/configs/run_plan_3060_pilot_train_v1.yaml" \
  --workspace "uavgpr_simlab/workspace/pilot_train_v1" \
  --count 100
```

预期输出：100 个 `case_000001 ~ case_000100` 目录，每个包含 `scene_world.json` + `.in` 文件（raw/target_only/background_only）。

- [ ] **Step 2: 对第一个 case 做 geometry dry-run 验证**

```bash
python3 -c "
from uavgpr_simlab.core.runner import run_geometry_dry_run
r = run_geometry_dry_run(
    'uavgpr_simlab/workspace/pilot_train_v1/models/case_000001/raw.in',
    python_exe='E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe'
)
print(r.status)
"
```

预期输出：`success`（gprMax 几何验证通过）。

- [ ] **Step 3: 批量 dry-run 前 5 个场景确认无几何错误**

```bash
python3 -c "
import glob, json
from uavgpr_simlab.core.runner import run_geometry_dry_run
cases = sorted(glob.glob('uavgpr_simlab/workspace/pilot_train_v1/models/case_*/raw.in'))[:5]
for f in cases:
    r = run_geometry_dry_run(f, python_exe='E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe')
    print(f'{f.split(\"/\")[-2]}: {r.status}')
"
```

预期输出：5 个 `success`。

- [ ] **Step 4: 检查场景家族分布**

```bash
python3 -c "
import json, glob
families = {}
for sw in sorted(glob.glob('uavgpr_simlab/workspace/pilot_train_v1/models/case_*/scene_world.json')):
    f = json.load(open(sw))['family']
    families[f] = families.get(f, 0) + 1
for k,v in sorted(families.items()):
    print(f'  {k}: {v}')
print(f'Total: {sum(families.values())}')
"
```

预期：5 个家族按 PILOT_FAMILY_CYCLE 分布（~25/25/25/17/8 比例）。

- [ ] **Step 5: 生成 manifest CSV**

```bash
"E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe" -c "
import csv, glob, json
from pathlib import Path
rows = []
models = Path('uavgpr_simlab/workspace/pilot_train_v1/models')
for c in sorted(models.glob('case_*')):
    cid = c.name
    sw = json.load(open(c / 'scene_world.json'))
    n_tr = sw['trace_count']
    for variant in ['raw', 'target_only', 'background_only']:
        inp = c / f'{variant}.in'
        if inp.exists():
            rows.append({'case_id': cid, 'variant': variant, 'input_file': str(inp), 'n_traces': str(n_tr)})
out = models.parent / 'manifest.csv'
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['case_id','variant','input_file','n_traces'])
    w.writeheader(); w.writerows(rows)
print(f'Manifest: {out} ({len(rows)} tasks)')
"
```

预期输出：`manifest.csv`，300 行（100 场景 × 3 变体）。

- [ ] **Step 6: 提交**

```bash
git add uavgpr_simlab/workspace/pilot_train_v1/
git commit -m "feat: generate 100 Pilot-Train scenes with geometry validation"
```

---

### Task 3: 批量 GPU 仿真（后台长时间任务）

**Files:**
- Run: `uavgpr_simlab/scripts/run_batch_safe_3060.py`
- Monitor: `uavgpr_simlab/workspace/pilot_train_v1/logs/`

**Interfaces:**
- Consumes: Task 2 的 manifest.csv + .in 文件
- Produces: 每个 case 的 `*_merged.out` 文件

- [ ] **Step 1: 确认 GPU 状态**

```bash
"E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe" -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB')"
```

预期：CUDA: True, VRAM: ~6.0GB。无其他占用 GPU 的进程。

- [ ] **Step 2: 先跑前 3 个场景验证仿真时间**

```bash
"E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe" \
  "uavgpr_simlab/scripts/run_batch_safe_3060.py" \
  --manifest "uavgpr_simlab/workspace/pilot_train_v1/manifest.csv" \
  --variants raw,target_only,background_only \
  --limit 3 \
  --n-traces 128
```

预期：3 个场景 × 3 变体 = 9 个任务成功。记录单任务耗时作为后续估算依据。

- [ ] **Step 3: 启动完整批次**

```bash
"E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe" \
  "uavgpr_simlab/scripts/run_batch_safe_3060.py" \
  --manifest "uavgpr_simlab/workspace/pilot_train_v1/manifest.csv" \
  --variants raw,target_only,background_only \
  --n-traces 128
```

预计运行时间：~50~60h GPU。可以后台断点续跑（batch runner 跳过已有 merged.out 的完成项）。

- [ ] **Step 4: 监控进度**

每小时检查一次：
```bash
echo "已完成: $(ls uavgpr_simlab/workspace/pilot_train_v1/models/case_*/raw_merged.out 2>/dev/null | wc -l)/100"
echo "target_only: $(ls uavgpr_simlab/workspace/pilot_train_v1/models/case_*/target_only_merged.out 2>/dev/null | wc -l)/100"
echo "bg_only: $(ls uavgpr_simlab/workspace/pilot_train_v1/models/case_*/background_only_merged.out 2>/dev/null | wc -l)/100"
```

- [ ] **Step 5: 完成后验证完整性**

```bash
python3 -c "
import glob
variants = ['raw', 'target_only', 'background_only']
for v in variants:
    files = sorted(glob.glob(f'uavgpr_simlab/workspace/pilot_train_v1/models/case_*/{v}_merged.out'))
    print(f'{v}: {len(files)}/100')
"
```

预期：3 个变体各 100 个 merged.out。

- [ ] **Step 6: 提交**

```bash
git add uavgpr_simlab/workspace/pilot_train_v1/
git commit -m "feat: complete 100-scene Pilot-Train gprMax simulation"
```

---

### Task 4: 后处理（merged.out → .npy）

**Files:**
- Modify: `scripts/batch_postprocess_pilot.py`（更新 WORKSPACE 路径指向新数据）
- Run: postprocess on new workspace

**Interfaces:**
- Consumes: Task 3 的 merged.out 文件
- Produces: 每个 case 的 `outputs/raw_bscan.npy`, `target_only_bscan.npy`, 等

- [ ] **Step 1: 更新 batch_postprocess_pilot.py 的 WORKSPACE 路径**

改 WORKSPACE 为：
```python
WORKSPACE = Path("D:/Claude/PGDA-CSNet/uavgpr_simlab/workspace/pilot_train_v1/yingshan_pilot_3060_v1")
MODELS = WORKSPACE / "models"
```

- [ ] **Step 2: 运行后处理**

```bash
"E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe" scripts/batch_postprocess_pilot.py
```

预期输出：每个 case 的 `outputs/` 目录下有 `raw_bscan.npy`, `target_only_bscan.npy`, `background_only_bscan.npy`, `clutter_gt_bscan.npy`。

- [ ] **Step 3: 验证后处理完整性**

```bash
python3 -c "
import numpy as np, glob
for v in ['raw_bscan','target_only_bscan','background_only_bscan']:
    files = sorted(glob.glob(f'uavgpr_simlab/workspace/pilot_train_v1/models/case_*/outputs/{v}.npy'))
    d = np.load(files[0])
    print(f'{v}: {len(files)}/100, shape={d.shape}')
"
```

预期：每变体 100 个文件，shape=(501, 64)：

- [ ] **Step 4: 提交**

```bash
git add scripts/batch_postprocess_pilot.py
git commit -m "feat: postprocess Pilot-Train merged.out to .npy"
```

---

### Task 5: 重写 convert_pilot_to_training.py（滑动窗口 + soft mask + 统一 P99）

**Files:**
- Rewrite: `scripts/convert_pilot_to_training.py`

**Interfaces:**
- Consumes: Task 4 的 `.npy` 文件
- Produces: `data/simulation_pretrain_v3/windows/*.npz` + `window_index.csv`

- [ ] **Step 1: 重写数据转换脚本**

```python
#!/usr/bin/env python3
"""Convert Pilot-Train B-scans to training .npz windows with sliding windows + soft mask + unified P99.

Raw gprMax B-scans are (501, 64). Use sliding window to extract multiple
256-wide windows per case. Save soft mask (not binarized). Normalize by
global P99 across all cases.
"""
import csv
import sys
from pathlib import Path

import numpy as np

WORKSPACE = Path(
    "D:/Claude/PGDA-CSNet/uavgpr_simlab/workspace/pilot_train_v1/"
    "yingshan_pilot_3060_v1"
)
MODELS = WORKSPACE / "models"
OUT = Path("D:/Claude/PGDA-CSNet/data/simulation_pretrain_v3/windows")
IDX = OUT.parent / "window_index.csv"
WINDOW_WIDTH = 256
N_SAMPLES = 501

def compute_global_p99(case_dirs):
    """Compute unified P99 across all cases."""
    all_vals = []
    for c in case_dirs:
        raw = np.load(c / "outputs" / "raw_bscan.npy").astype(np.float32)
        all_vals.append(np.abs(raw).ravel())
    all_vals = np.concatenate(all_vals)
    p99 = float(np.percentile(all_vals, 99))
    if p99 < 1e-12:
        p99 = 1e-12
    return p99

def create_padded_window(raw, mask, global_p99):
    """Pad (501, 128) to (501, 256) with center-padding.

    128 traces < 256 window width. Center-pad to 256,
    normalize by global P99, keep soft mask.
    Returns one window per case.
    """
    n_traces = raw.shape[1]
    pad_left = (WINDOW_WIDTH - n_traces) // 2
    pad_right = WINDOW_WIDTH - n_traces - pad_left

    rw = np.pad(raw, ((0, 0), (pad_left, pad_right)), mode="reflect")
    mw = np.pad(mask, ((0, 0), (pad_left, pad_right)), mode="reflect")

    original_cols = np.zeros(WINDOW_WIDTH, dtype=bool)
    original_cols[pad_left:pad_left + n_traces] = True

    rw = rw / global_p99
    mw = np.clip(mw, 0.0, 1.0).astype(np.float32)
    mw[:, ~original_cols] = 0.0

    peak = mw.max(axis=0)
    status = np.full(WINDOW_WIDTH, 2, dtype=np.int16)
    status[peak > 0.5] = 1
    status[peak < 0.1] = 0
    status[~original_cols] = 0

    weight = np.maximum(peak, 0.3).astype(np.float32)
    weight[~original_cols] = 0.0

    return rw.astype(np.float32), mw, status, weight

def main():
    case_dirs = sorted(MODELS.glob("case_*"))
    OUT.mkdir(parents=True, exist_ok=True)

    print("Computing global P99 across all cases...")
    global_p99 = compute_global_p99(case_dirs)
    print(f"Global P99: {global_p99:.4f}")

    all_rows = []
    converted = 0

    for c in case_dirs:
        case_id = c.name
        raw = np.load(c / "outputs" / "raw_bscan.npy").astype(np.float32)
        mask = np.load(c / "labels" / "interface_mask_bscan.npy").astype(np.float32)
        if not np.isfinite(raw).all() or raw.size == 0:
            print(f"  {case_id}: NaN or empty raw, skipping")
            continue

        rw, mw, sc, lw = create_padded_window(raw, mask, global_p99)
        sid = f"pilot_{case_id}_w00"
        npz_path = OUT / f"{sid}.npz"
        np.savez_compressed(npz_path, x_raw=rw, y_mask=mw, status_code=sc, label_weight=lw)

        n_present = int((sc == 1).sum())
        n_weak = int((sc == 2).sum())
        n_absent = int((sc == 0).sum())
        all_rows.append({
            "sample_id": sid, "line": f"pilot_{case_id}",
            "start": 0, "end": WINDOW_WIDTH - 1, "split": "train",
            "present": n_present, "weak": n_weak, "no_pick": n_absent,
        })
        converted += 1
                "weak": n_weak,
                "no_pick": n_absent,
            })
            converted += 1

        print(f"  {case_id}: {len(windows)} windows, "
              f"x_raw=[{rw.min():.3f},{rw.max():.3f}]")

    # Write index
    if all_rows:
        fieldnames = ["sample_id", "line", "start", "end", "split",
                      "present", "weak", "no_pick"]
        with open(IDX, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nWrote {len(all_rows)} entries to {IDX}")
    print(f"\nSummary: {converted} windows from {len(case_dirs)} cases")

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: 创建输出目录并运行**

```bash
mkdir -p "D:/Claude/PGDA-CSNet/data/simulation_pretrain_v3/windows"
"E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe" scripts/convert_pilot_to_training.py
```

预期输出：
```
Global P99: <value>
  case_000001: 1 window
  ...
Total: ~100 windows
```

- [ ] **Step 3: 验证输出 NPZ 质量**

```bash
python3 -c "
import numpy as np, csv
from pathlib import Path
data = Path('data/simulation_pretrain_v3')
with open(data/'window_index.csv') as f:
    rows = list(csv.DictReader(f))
print(f'Total samples: {len(rows)}')
status = {0:0,1:0,2:0}
for r in rows:
    status[int(r['present']>0)] += 1  # present traces
    status[0] += int(r['no_pick'])
# 抽查第一个 NPZ
d = np.load(list((data/'windows').glob('*.npz'))[0])
print(f'Sample NPZ: x_raw={d[\"x_raw\"].shape}, y_mask={d[\"y_mask\"].shape}')
print(f'x_raw range: [{d[\"x_raw\"].min():.3f}, {d[\"x_raw\"].max():.3f}]')
print(f'y_mask range: [{d[\"y_mask\"].min():.3f}, {d[\"y_mask\"].max():.3f}]')  # 应该是连续值
"
```

关键检查：
- `y_mask` 范围是 [0, 1] 浮点（非二值化）
- `x_raw` 幅值范围一致（统一 P99 归一化）
- status_code 中 absent/present/weak 分布合理

- [ ] **Step 4: 提交**

```bash
git add scripts/convert_pilot_to_training.py
git add data/simulation_pretrain_v3/
git commit -m "feat: convert Pilot-Train to training NPZ with sliding windows + soft mask + unified P99"
```

---

### Task 6: 数据质量检查（QA）

**Files:**
- Create (or use existing): `scripts/check_dataset.py`
- Output: QA 报告

**Interfaces:**
- Consumes: Task 5 的 NPZ 文件
- Produces: QA 报告 + B-scan 预览图

- [ ] **Step 1: 运行现有数据检查脚本**

```bash
"E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe" scripts/check_dataset.py \
  --data-root data/simulation_pretrain_v3
```

如果脚本不存在，则运行以下快速检查：
```python
python3 -c "
import numpy as np, csv, glob
from pathlib import Path

data = Path('data/simulation_pretrain_v3')
npzs = sorted(glob.glob(str(data/'windows/*.npz')))
print(f'Total NPZ files: {len(npzs)}')

# Check all NPZs
errors = []
for p in npzs:
    d = np.load(p)
    if d['x_raw'].shape != (501, 256):
        errors.append(f'{p.name}: x_raw shape {d[\"x_raw\"].shape}')
    if d['y_mask'].shape != (501, 256):
        errors.append(f'{p.name}: y_mask shape {d[\"y_mask\"].shape}')
    if not np.isfinite(d['x_raw']).all():
        errors.append(f'{p.name}: NaN/inf in x_raw')

if errors:
    for e in errors[:10]:
        print(f'ERROR: {e}')
else:
    print(f'All {len(npzs)} NPZs pass shape/finite check')

# Check window index has matching NPZ
with open(data/'window_index.csv') as f:
    rows = list(csv.DictReader(f))
idx_ids = {r['sample_id'] for r in rows}
npz_ids = {p.stem for p in npzs}
missing = idx_ids - npz_ids
extra = npz_ids - idx_ids
if missing: print(f'Missing NPZ in index: {missing}')
if extra: print(f'Extra NPZ not in index: {extra}')
if not missing and not extra:
    print(f'Index/NPZ match: {len(rows)} rows, {len(npzs)} files')
"
```

- [ ] **Step 2: 生成 B-scan 预览图**

```bash
python3 -c "
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
data = Path('data/simulation_pretrain_v3/windows')
npzs = sorted(data.glob('*.npz'))[:8]
fig, axes = plt.subplots(2, 4, figsize=(16, 6))
for i, ax_row in enumerate(axes):
    for j, ax in enumerate(ax_row):
        idx = i*4+j
        if idx < len(npzs):
            d = np.load(npzs[idx])
            vmin, vmax = np.percentile(d['x_raw'], [1, 99])
            ax.imshow(d['x_raw'], aspect='auto', cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(npzs[idx].stem[:20], fontsize=8)
        ax.axis('off')
plt.tight_layout()
plt.savefig('C:/Users/17844/Desktop/pilot_train_v3_preview.png', dpi=120)
print('Saved: desktop/pilot_train_v3_preview.png')
"
```

- [ ] **Step 3: 看一眼数据覆盖**

计算所有窗口的总道数：
```bash
python3 -c "
import csv
from pathlib import Path
with open(Path('data/simulation_pretrain_v3/window_index.csv')) as f:
    rows = list(csv.DictReader(f))
total_present = sum(int(r['present']) for r in rows)
total_weak = sum(int(r['weak']) for r in rows)
total_no_pick = sum(int(r['no_pick']) for r in rows)
total = total_present + total_weak + total_no_pick
print(f'Present: {total_present} ({total_present/total*100:.1f}%)')
print(f'Weak:    {total_weak} ({total_weak/total*100:.1f}%)')
print(f'Absent:  {total_no_pick} ({total_no_pick/total*100:.1f}%)')
print(f'Total:   {total} traces')
"
```

预期：present ~15~25%, weak ~5~15%, absent ~60~80%（自然界线出现密度）。

- [ ] **Step 4: 提交 QA 结果**

```bash
git add data/simulation_pretrain_v3/
git commit -m "qa: Pilot-Train v3 dataset verification passed"
```

---

### Task 7: 更新训练配置并创建新 LOLO 配置

**Files:**
- Create: `configs/gpu_train_v4_pilot_mixed.json`（基于 v3，指向新数据 + 调整参数）
- Create: `configs/` 下的 LOLO configs for Line9 fold

**Interfaces:**
- Consumes: Task 5-6 的 `simulation_pretrain_v3` 数据
- Produces: 新训练配置

- [ ] **Step 1: 创建 v4 训练配置**

```json
{
  "data_root": "data_corrected_v1_4_terrain_direction",
  "paper_split_file": "configs/paper_splits_v1_6.json",
  "height_resize": 512,
  "width_resize": 256,
  "batch_size": 4,
  "epochs": 60,
  "lr": 0.0005,
  "base_ch": 20,
  "model_dropout": 0.06,
  "num_workers": 0,
  "seed": 1902,
  "train_lines": ["Line3", "Line6", "Line7", "LineL1"],
  "val_lines": ["Line9"],
  "test_lines": ["Line9"],
  "test_trace_ranges": {},
  "review_lines": ["LineX1"],
  "loss": {
    "core_weight": 0.55,
    "outside_weight": 0.36,
    "dice_weight": 0.85,
    "presence_weight": 0.42,
    "presence_negative_weight": 2.8,
    "core_threshold": 0.55,
    "outside_margin": 0.05,
    "weak_presence_target": 0.65,
    "positive_pixel_boost": 8.0,
    "hard_negative_weight": 0.24,
    "hard_negative_topk_frac": 0.02,
    "centerline_weight": 0.2,
    "continuity_weight": 0.035,
    "center_valid_min_sum": 0.001,
    "spectral_consistency_weight": 0
  },
  "augment": {
    "enabled": true,
    "amp_scale_min": 0.88,
    "amp_scale_max": 1.12,
    "noise_std": 0.0001,
    "trace_dropout_prob": 0.015,
    "horizontal_flip_prob": 0.35
  },
  "deterministic": false,
  "input_log_scale": 0.001,
  "no_pick_window_repeats": 2,
  "model_arch": "v1_9d_mambavision_hybrid",
  "max_preview_val": 0,
  "run_dir": "outputs/run_gpu_v4_pilot_mixed_loo_Line9",
  "version": "v4_pilot_mixed_loo_Line9_seed1902",
  "note": "v4: Pilot-Train 100 scenes (~100 windows), sim_batch_ratio=0.6, unified P99, soft mask, batch_size=4, epochs=60",
  "ssm_kernel": 31,
  "attention_heads": 4,
  "sim_batch_ratio": 0.6,
  "sim_data_root": "D:/Claude/PGDA-CSNet/data/simulation_pretrain_v3",
  "sim_train_lines": ["pilot_case_000001", ..., "pilot_case_000100"]
}
```

注意替换 `sim_train_lines` 为实际 100 个 case 的列表。

- [ ] **Step 2: 为 3 个种子（1901/1902/1903）生成 LOLO configs**

用 `scripts/make_v3_pilot_mixed_loo_configs.py` 做模板，修改 `version` 和 `run_dir` 指向 v4 数据。

- [ ] **Step 3: 提交**

```bash
git add configs/gpu_train_v4_pilot_mixed_loo_*.json
git commit -m "config: create v4 Pilot-Train LOLO configs"
```

---

### Task 8: 训练 LOLO-CV Line9 fold（验证性训练）

**Files:**
- Run: `scripts/train_raw_only.py` with v4 config
- Output: `outputs/run_gpu_v4_pilot_mixed_loo_Line9_seed190*/`

**Interfaces:**
- Consumes: Task 7 的 config 文件
- Produces: 模型 checkpoint + 训练日志

- [ ] **Step 1: GPU 就绪检查**

```bash
"E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe" -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_properties(0).total_mem/1e9)"
```

- [ ] **Step 2: 启动 3 个种子训练（串行）**

```bash
# seed1902 (best validation seed 先用)
"E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe" -u scripts/train_raw_only.py \
  configs/gpu_train_v4_pilot_mixed_loo_Line9_seed1902.json 2>&1 | tee logs/v4_loo_Line9_seed1902.log
```

完成后依次跑 seed1901 和 seed1903：
```bash
"E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe" -u scripts/train_raw_only.py \
  configs/gpu_train_v4_pilot_mixed_loo_Line9_seed1901.json
"E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe" -u scripts/train_raw_only.py \
  configs/gpu_train_v4_pilot_mixed_loo_Line9_seed1903.json
```

- [ ] **Step 3: 监控训练**

关键指标：
- best_val_loss 是否 < 0.8（优于当前 ~0.93）
- best_epoch 是否在 20~40 之间（正常收敛）
- train-val gap 是否 < 0.3

- [ ] **Step 4: 提交训练结果**

```bash
git add outputs/run_gpu_v4_pilot_mixed_loo_Line9_*/
git commit -m "train: v4 Pilot-Train LOLO Line9 fold 3-seed"
```

---

### Task 9: 集成评估与对比

**Files:**
- Run: `scripts/eval_full_line.py`
- Output: `outputs/eval_v4_pilot_mixed_loo_Line9_3seed_ensemble/`

**Interfaces:**
- Consumes: Task 8 的 3 个 checkpoint
- Produces: 评估指标 + 对比 baseline

- [ ] **Step 1: 运行 3-seed 集成评估**

```bash
"E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe" scripts/eval_full_line.py \
  --line Line9 \
  --run-dirs outputs/run_gpu_v4_pilot_mixed_loo_Line9_seed1901 \
             outputs/run_gpu_v4_pilot_mixed_loo_Line9_seed1902 \
             outputs/run_gpu_v4_pilot_mixed_loo_Line9_seed1903 \
  --out-dir outputs/eval_v4_pilot_mixed_loo_Line9_3seed_ensemble \
  --dp-breakable --center-fusion-weight 1.0
```

- [ ] **Step 2: 对比 baseline**

```bash
python3 -c "
import csv
# 读新结果
with open('outputs/eval_v4_pilot_mixed_loo_Line9_3seed_ensemble/Line9_full_metrics.csv') as f:
    new = {r[0]:r[1] for r in csv.reader(f)}
# 读旧结果
with open('outputs/eval_v3_pilot_mixed_loo_Line9_3seed_ensemble/Line9_full_metrics.csv') as f:
    old = {r[0]:r[1] for r in csv.reader(f)}
keys = ['dp_center_mae_ns', 'pick_rate', 'center_mae_smp', 'iou_at_0_3']
print(f'{\"Metric\":25s} {\"v3(169sp)\":12s} {\"v4(600sp)\":12s} {\"Change\":10s}')
print('-'*60)
for k in keys:
    if k in new and k in old:
        o, n = float(old[k]), float(new[k])
        print(f'{k:25s} {o:>10.4f}  {n:>10.4f}  {n-o:>+9.4f}')
"
```

- [ ] **Step 3: 更新项目状态报告**

```bash
python3 scripts/report_on_eval.py
```

- [ ] **Step 4: 提交评估结果**

```bash
git add outputs/eval_v4_pilot_mixed_loo_Line9_3seed_ensemble/
git commit -m "eval: v4 Pilot-Train LOLO Line9 3-seed ensemble results"
```

---

## 验证标准汇总

| 检查项 | 预期 | 责任 Task |
|--------|------|:---------:|
| 100 场景几何生成成功 | case_000001~case_000100 存在 | Task 2 |
| 几何预检通过 | geometry dry-run 全 success | Task 2 |
| 100 场景仿真完成 | 每 case 3 个 merged.out | Task 3 |
| P99 归一化统一 | 所有 x_raw 幅值在统一范围 | Task 5 |
| y_mask 为 soft (0~1) | 非二值化，连续值 | Task 5 |
| status_code 分布合理 | absent>present>weak | Task 6 |
| val_loss 降低 > 0.2 | 从 ~0.93 降到 < 0.73 | Task 9 |
| DP MAE 降低 | 从 37.19ns 预期降低 | Task 9 |

---

## 执行顺序

```
Task 1 (配置验证) → Task 2 (场景生成+预检) → Task 3 (GPU仿真,~56h) 
→ Task 4 (后处理) → Task 5 (NPZ转换) → Task 6 (QA)
→ Task 7 (训练配置) → Task 8 (训练,~10h) → Task 9 (评估)
```

Task 1-2 可连续执行（~30min），Task 3 是长时间 GPU 任务（建议后台运行），Task 4-6 在 Task 3 完成后执行（~30min），Task 7-9 是最终验证阶段（~12h）。
