# CLAUDE.md

## ⚠️ 致命铁律：汇报仿真结果必须附带三件套

**任何仿真跑完后，汇报时必须同时包含：**
1. **几何模型图**（材料分层、TX/RX位置、图例）
2. **原始 B-scan**
3. **处理后 B-scan**（AGC 或 air_only 减法）

**缺一不可。先出齐三件套再说话。否则等着挨骂。**

**使用方式：**
```bash
python .claude/skills/sim-report/sim_report.py <仿真输出目录>
```
自动出 几何模型图 + 原始B-scan + 处理后图 + 清理.vti

## Project Overview

**PGDA-CSNet** (Physics-guided Domain-Adaptive Clutter Suppression Network) — deep learning-based background clutter suppression for low-frequency UAV-borne Ground Penetrating Radar (UavGPR), targeting landslide bedrock interface detection.

The core challenge: low-frequency airborne GPR suffers from strong clutter (direct wave, ground reflection, platform altitude variation, horizontal bedding) that masks weak bedrock reflections. Only **one real survey dataset** exists with no clean ground-truth B-scan labels. The solution uses **gprMax FDTD simulation** to generate paired labeled data, combined with physics-guided branches and unsupervised domain adaptation to transfer to real data.

## Reference Document

The full research plan is in `UavGPR_机器学习背景杂波抑制全项研究计划.docx`.
Extract text via pandoc or python-docx (see docx skill).

## Current State (2026-07-08)

### Architecture
**GprMambaSep v2.1 (B-Guarded GprMambaSepLite)** — code in `pgdacsnet/model_gprmambasep.py`.
ConvNeXt encoder + axial SSM mixer → split bottleneck → **A/S/G explicit decomposition** (three `_ComponentDecoder`s) → G-branch task heads (mask/presence/center).

Old `v1_9d_mambavision_hybrid` (UNet-based) archived at `legacy/unet-arch` branch.

### Training Progress

| Stage | Status | Line9 DP MAE | Key Takeaway |
|-------|--------|-------------|--------------|
| 0: P0-3 Baseline | ✅ | **3.27 ns** | Center fusion baseline, no decomposition |
| 1: Sim-only pretrain (v2.1) | ✅ 80 ep | — | AMP enabled, 192 NPZ from Pilot-Mini, stable |
| 2: Mixed sim-real Line9 holdout | ✅ 80 ep | **25.24 ns** | Self-consistency=0.003 but G branch drifts — no component supervision |

**Critical finding**: The model achieves near-perfect reconstruction (`A_hat+S_hat+G_hat ≈ Y_full`, self-consistency loss=0.003) but the G component is **not aligned to the correct geological interface**. DP MAE=25.24ns vs P0-3 baseline of 3.27ns — the model picks correctly (56% pick rate) but picks the wrong depth. Root cause: no L2 component supervision (`sim_supervised_weight=0.0` in config), so G self-organizes as an unconstrained residual bucket.

### Data Status

| Dataset | Samples | Has Y_air / X_clean / G_target? | Status |
|---------|---------|----------------------------------|--------|
| `simulation_pretrain_v1` | 192 NPZ | ❌ No | Stage 1 used |
| `data_corrected_v1_4` | 78 NPZ (6 lines) | ❌ No | Stage 2 used |
| `batch_003` | 20/24 cases with raw | ❌ No air_only | Not used yet |

**No component arrays exist in any current NPZ.** This is the primary bottleneck for L2 G supervision.

### Next Bottleneck
1. Generate Batch4 with `raw_full + background_only + basal_only` paired simulations
2. Inject component arrays into NPZ pipeline
3. Enable `sim_supervised_weight` with G_target / X_clean / Y_air supervision
4. Add G_band / distractor / global no-target losses

---

## Architecture Reference

**Current main: GprMambaSep (v2_0 / v2_1_gprmambasep_lite / gprmambasep_lite)** — `pgdacsnet/model_gprmambasep.py`

- `build_gprmambasep(cfg)` → `GprMambaSep`
- Dispatched via `build_model(cfg)` in `pgdacsnet/model_raw_unet.py`
- All `model_arch` aliases map to the same class: `"v2_0_gprmambasep"`, `"gprmambasep"`, `"v2_1_gprmambasep_lite"`, `"gprmambasep_lite"`, `"v2_1_curvegassist_lite"`, `"curvegassist_lite"`, `"g_assisted_curvemamba"`

### Architecture Diagram

```
Input (B, C, H, W)  C=1(radar)+N(aux)
  └─ Stem (conv3x3 → LN)
       └─ ConvNeXt Stages (×4, 64→128→256→512)
            └─ AxialSSMLiteBlock mixer
                 └─ Split Bottleneck
                      ├─ Decoder_A ──► A_hat (air wave)
                      ├─ Decoder_S ──► S_hat (surface)
                      └─ Decoder_G ──► g_task_feat
                              ├─ mask_head ──► mask_logits (B,1,H,W)
                              ├─ pres_head ──► presence_logits (B,1,W)
                              └─ center_head ──► center_logits (B,1,H,W)
                              (optional: curve_head / global_no_target_head / uncertainty_head)

**(Route 2 — G-assisted curve path)**: When `task_feature_mode="g_assisted"`, a
shared decoder from bottleneck + raw-local stem features are fused with G-decoder
features before the task heads, decoupling curve/presence/center from G-branch drift.
```

### Output Contract

`GprMambaSepOutput` (in `pgdacsnet/model_interfaces.py`):
- `mask_logits` / `presence_logits` / `center_logits` — standard PGDA (with backward-compat aliases `G_mask_logits` etc.)
- `A_hat` / `S_hat` / `G_hat` — component decomposition (B,1,H,W)
- `component_gates` — diagnostic per-trace A/S/G gating weights
- Tuple-unpackable: `(mask, pres, center, A_hat, S_hat, G_hat)` — 6 fields
- Dict/attribute access supported

### Key Architectural Features
- **`_SkipFuse`**: Gated skip connections that learn to fuse encoder features into decoders
- **`_ComponentDecoder`**: 3-stage full-resolution decoder (no resolution loss, stride-1 convs)
- **`component_gate`**: Softmax over pooled decoder features — models which component dominates each trace
- **Validity-masked losses**: `*_valid` flags in loss functions for mixed-batch safety (sim vs real)
- **AMP**: `torch.amp.GradScaler` — production 512×256 fits on 6GB RTX 3060

### PGDA Output Contract
- `mask_logits`: (B,1,H,W) — per-pixel target probability logits
- `presence_logits`: (B,1,W) — per-trace target presence (NOT (B,1,H,W))
- `center_logits`: (B,1,H,W) — per-column center depth
- `unpack_pgda_output()` / `unpack_model_output()` handles all output types

---

## Training Strategy

| Phase | Config | Data | Key Settings | Results |
|-------|--------|------|-------------|---------|
| Stage 1: Sim pretrain | `gpu_pretrain_v2_1_gprmambasep_lite.json` | `simulation_pretrain_v1` (192 NPZ) | lr=3e-4, ep=80, AMP, self_consistency=2.0 | Stable convergence, warm-start for Stage 2 |
| Stage 2: Mixed sim-real | `gpu_mixed_v2_1_gprmambasep_lite_line9holdout.json` | Real L3/L6/L7/L1 + sim v1 | lr=1e-4, ep=80, AMP, sim_batch_ratio=0.3, sim_supervised_weight=0.0 | DP MAE=25.24ns, PR=56%, self-consistency=0.003 |
| Stage 2a (Route 2): G-assisted CurveMamba | `gpu_mixed_v2_1_curvegassist_line9holdout_6g.json` (6G) / `_12g.json` (12G) | Real L3/L6/L7/L1 + sim v1 | task_feature_mode=g_assisted, curve_head, global_no_target, grad_accum_steps=4/2 | ⏳ Pending — curve P(t|trace) + fused task path |
| Stage 2.1 (planned): +Component supervision | TBD | +Batch4 component arrays | Enable sim_supervised_weight, add G_band/distractor/global losses | Target: DP MAE < 10ns |

### Critical Config Fields
```json
{
  "model_arch": "v2_1_gprmambasep_lite",   // dispatches to GprMambaSep
  "base_ch": 12,                            // stem channels (controls model width)
  "mamba_state_dim": 32,                    // SSM state dimension
  "ssm_kernel": 31,                         // SSM convolution kernel
  "attention_heads": 4,
  "sim_supervised_weight": 0.0,             // OFF — needs component arrays
  "self_consistency_weight": 1.5,           // A_hat+S_hat+G_hat ≈ Y_full
  "warm_start_from": "path/to/checkpoint_best.pt"
}
```

### Data Split Discipline (Critical)
- **按测线分割**，非随机 patch
- **Leave-one-line-out**: 至少一条完整测线用于测试
- 同测线不同功率档 → 同一 split

---

## Loss Functions

`scripts/losses_gprmambasep.py` — `compute_gprmambasep_loss()` orchestrates:

| Loss | Config Key | Receptive? | Notes |
|------|-----------|------------|-------|
| Segmentation BCE/Dice/Core | `core_weight`, `dice_weight` | ✅ | Standard PGDA, applied to mask_logits via g_task_feat |
| Outside penalty | `outside_weight` | ✅ | Penalizes predictions outside valid time window |
| Presence (per-trace) | `presence_weight` | ✅ | BCE with negative weighting, no-pick support |
| Centerline L1 | `centerline_weight` | ✅ | Supervises center depth |
| Continuity | `continuity_weight` | ✅ | Smoothness prior on center prediction |
| Self-consistency | `self_consistency_weight` | ✅ | L1+MSE: A_hat+S_hat+G_hat ≈ Y_full |
| L2 component supervision | `sim_supervised_weight` | ❌ OFF | Reads Y_air/X_clean/G_target from NPZ — none exist yet |
| Arrival prior | `arrival_prior_weight` | ✅ | Penalizes G_hat energy before expected arrival |
| Amplitude ratio prior | `amplitude_ratio_weight` | ✅ | Constrains A/S/G relative amplitudes |
| Co-prediction cycle | `co_prediction_cycle_weight` | ✅ | Cycle-consistency between heads |
| Curve distribution (Route 2) | `curve_distribution_weight` | ✅ | Trace-wise CE + center + smooth + curvature + shallow |
| Global no-target (Route 2) | `global_no_target_weight` | ✅ | Window-level abstention BCE |
| Uncertainty NLL (Route 2) | `uncertainty_weight` | ✅ | Heteroscedastic NLL calibration |
| G-envelope mask consistency | `g_envelope_mask_weight` | ✅ | Forces G_hat energy to match target mask |
| Component gate regularization | `component_gate_balance/entropy_weight` | ✅ | Prevents A/S/G gate collapse |

---

## Signal Model (Locked Definition)

```
A = 直达波 / 天线间空气耦合
S = 地表参考反射
G = 地下有效地质信号（基覆界面、互层、深部异常）
E = 外部杂波（电线、树木、建筑 — 当前阶段暂不生成）

Y_full   = A + S + G + E          (raw 变体)
Y_target = A + S + G              (target_only 变体)
Y_air    = A                      (air_only 变体, 全局模板, 固定几何下复用)

X_clean = Y_target - Y_air = S + G     (保留地表反射)
C_gt    = Y_full - X_clean  = A + E    (操作性杂波标签)
```

C_gt 是**操作性标签**，不等同于严格电磁场分解真值（FDTD 中存在多次散射）。

---

## 仿真数据管理 (PGDA_SYNTH_DATASET_V1)

所有正式仿真统归 `data/PGDA_SYNTH_DATASET_V1/`。

### 五层治理

| 层级 | 路径 | 说明 |
|------|------|------|
| 模型源 | `01_templates/` | 已验证的几何模板，作为生成源 |
| 仿真输出 | `03_runs/` | gprMax 实际运行结果 |
| 质检 | `04_qc/` | preflight + after_run 检查结果 |
| 训练数据 | `05_accepted_dataset/` | GREEN 通过后 promote 至此 |
| 归档 | `08_archives/` | .out 大文件压缩归档 |

### 流转规则

```
生成 → preflight_check → gprMax 跑 → after_run_qc → GREEN → promote_to_accepted
```

### 关键模板

| 模板 | 状态 | 存放 |
|------|------|------|
| V3.4/V3.5/V3.6 | 历史参考模板 | `01_templates/` |
| LINE9_LABEL_INSPIRED_V1 | 当前主模板，已 promote | `01_templates/` + `05_accepted_dataset/` |

### 工作空间

| 路径 | 内容 |
|------|------|
| `data_corrected_v1_4_terrain_direction/` | 实测训练数据（6 测线 × 3 功率档） |
| `data/simulation_pretrain_v1/` | 20 Pilot-Mini 仿真场景（192 NPZ） |
| `data/simulation_pretrain_v3/` | Pilot-Train 100 场景（待仿真） |
| `data/营山/` | 原始实测数据 + 钻孔资料 |
| `data/PGDA_SYNTH_DATASET_V1/` | 正式仿真数据治理目录 |
| `uavgpr_simlab/` | SimLab 仿真工具 |

---

## SimLab (仿真工具)

**位置**: `D:\Claude\PGDA-CSNet\uavgpr_simlab\`
**安装**: `pip install -e D:\Claude\PGDA-CSNet\uavgpr_simlab` (已安装在 gprMax .venv)
**gprMax**: v3.1.7 at `E:\gprMax\gprMax-v.3.1.7`，Python: `.venv\Scripts\python.exe`
**GPU**: `SafeGprMaxRunner` 自动注入 MSVC + CUDA 路径到 PATH，无需 conda 或 vcvars

**⚠️ GPU 复杂模型**: 带介电平滑的复杂模型（643+ box）直接 `python -m gprMax` 会因 MSVC INCLUDE/LIB 路径缺失导致 nvcc 编译失败。**必须使用 SafeGprMaxRunner** 或 `run_batch.py`（内部通过 `_inject_msvc_paths()` 注入 PATH/INCLUDE/LIB）。

**⚠️ MSVC 路径注入是最易被忽略的陷阱**: Bash 下直接 `python -m gprMax raw.in -gpu` 进程会挂死、GPU 0% 占用。详细排错见 `.claude/skills/gprmax-usage/` 技能。

**⚠️ .in 文件注释语法**: gprMax v3.1.7 `check_cmd_names` 会解析所有 `#` 行并分割 `:`。纯注释行（无冒号如 `# --- geometry boxes ---`）导致 IndexError。必须用 `#:` 或移除该行。

### gprMax 源码关键发现

- **`#geometry_objects_read` 覆盖不彻底**: 写入 G.solid + G.ID + G.rigidE/H，后续 `#box` 只改 G.solid 不改 G.ID/rigid（`input_cmds_geometry.py:136`）
- **`#triangle` 介电平滑**: 第 13 个参数 `y` 启用 dielectric smoothing，避免 H5 stair-step（`input_cmds_geometry.py:355`）
- **GPU 常量内存 64KB**: 每材料 128 bytes → 最多 ~1600 个材料（`fields_updates_gpu.py`）
- **介电平滑副作用**: 在材料边界创建平均材料（如 `air+weathered_bedrock`），增加材料总数，改变反射强度

### 关键配置文件

| 文件 | 用途 |
|------|------|
| `configs/run_plan_3060_pilot_train_v1.yaml` | Pilot-Train 100 场景生产配置 |
| `configs/gpu_train_v4_pilot_mixed.json` | v4 训练配置（100 仿真场景） |
| `scripts/run_pilot_train_resumable.py` | 续跑脚本（SafeGprMaxRunner，跳过已完成 case） |
| `scripts/postprocess_simulation_batch.py` | .out → raw_bscan_native.npy 批量后处理 |
| `scripts/convert_pilot_to_training.py` | 新版数据转换（线性插值 + 全局P99 + 软标签） |
| `data/PGDA_SYNTH_DATASET_V1/tools/run_batch.py` | 批量运行（PGDA 治理模式） |

### 常用命令

```bash
# 三件套（仿真跑完立即执行）
python .claude/skills/sim-report/sim_report.py <输出目录>

# case 生成（深度变体 batch）
python data/PGDA_SYNTH_DATASET_V1/tools/generate_cases.py batch_xxx --n-cases 30 --depth-range 6 24

# 跑前预检（必做）
python data/PGDA_SYNTH_DATASET_V1/tools/preflight_check.py <case_dir>

# 批量运行
python data/PGDA_SYNTH_DATASET_V1/tools/run_batch.py <batch_dir>

# 跑后质检
python data/PGDA_SYNTH_DATASET_V1/tools/after_run_qc.py <case_run_dir>

# 数据转换
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" scripts/convert_pilot_to_training.py
```

---

## Pipeline Bug History (参考)

以下为已修复 bug，仅摘要。详情见 commit log。

| Bug | 修复 |
|-----|------|
| mask 缺少空气走时 | 加 `air_twt` + 动态速度 |
| domain_y/dx 非整数 | `math.ceil(val/dx)*dx` |
| PML 未声明 | 强制 `#pml_cells: 60 60 0 60 60 0` |
| FFT ringing | `np.interp` 线性插值 |
| P99 独立计算 | 改为全局统一 |
| y_mask 时间错位 | 加空气走时，用有效平均速度 ~0.069 m/ns |
| generate_cases 坐标系反(2026-07-04) | SURFACE_BASE=0, 天线在空气区 |
| preflight min(y)推断地面(2026-07-04) | 改为按几何交点检测 |
| preflight PML只检查上边(2026-07-04) | 同时检查 y > domain_y - PML_y1 |
| run_batch 重复-n(2026-07-04) | 合并为一次 |
| run_batch VENV_PYTHON无.parent(2026-07-04) | 改为 Path(VENV_PYTHON) |
| run_batch QC失败仍completed(2026-07-04) | QC非零→qc_failed |
| promote缺bscan只warning(2026-07-04) | 必须存在，否则exit(1) |
| promote metadata从任意模板复制(2026-07-04) | 改为从run_info真实模板 |
| after_run_qc不查本地labels(2026-07-04) | 优先检查case_run_dir/labels |
| sim_report不支持raw/子目录(2026-07-04) | 自动识别run_dir/raw/ |

---

## Hardware & Data Constraints

- **Current (开发)**: RTX 3060 Laptop **6 GB** — dx=0.05m 已验证可用（~1.2GB VRAM/场景）
- **后续 (生产)**: RTX 4090 Laptop 16 GB — 可上 dx=0.025m
- **配 YAML config profile 切换**，不硬编码 GPU 参数
- **实测数据**: 营山 6 条测线 (L3/L6/L7/L9/L1/X1) × 3 功率档，501 samples × 700ns
- **钻孔**: ZK07 (~21-23m), ZK09 (~20-23m), 多个浅钻孔 (~9-17m)，数据在 `data/营山/`
- **决策**: 仿真用 gprMax 原生时域 Ricker 100 MHz，不做 SFCW 转换。当前阶段暂不做外部杂波（电线/树木/建筑），聚焦地质核心

---

## ⚠️ Python 解释器（关键陷阱）

**始终使用 gprMax venv Python**:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe"
```

**不要用** `E:\python\python.exe`（系统 Python）——它显示 `torch.cuda.is_available()=False`，但历史训练进程在某些条件下却能跑 CUDA（原因不明）。这导致了多次混淆和重复进程问题。

**不要用** `D:\Miniconda3\python.exe`。

验证 CUDA:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "import torch; print(torch.cuda.is_available())"
```

## 日常训练命令

```bash
# 训练（从头开始）
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -u scripts/train_raw_only.py configs/<config>.json

# 恢复训练（从 checkpoint）
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -u scripts/resume_train.py configs/<config>.json

# 集成评估（3-seed ensemble）
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" scripts/eval_full_line.py \
  --line Line9 \
  --run-dirs outputs/run_xxx_seed1901 outputs/run_xxx_seed1902 outputs/run_xxx_seed1903 \
  --out-dir outputs/eval_xxx_3seed --dp-breakable --center-fusion-weight 1.0

# 检查 checkpoint epoch
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "import torch; c=torch.load('<path>/checkpoint_last.pt',map_location='cpu',weights_only=False); print(c['epoch'])"
```

## GPU 注意事项 (RTX 3060 Laptop 6GB)

- **VRAM 上限**: 6144 MiB，训练占用 ~4000 MiB，不要同时跑多个训练
- **温度**: 训练时 80-87°C 正常，>90°C 有热降频风险
- **建议**: 训练前检查 GPU 状态 (`/gpu-health`)，确保无其他进程占用显存

### TDR 问题与解决方案

Windows GPU 超时检测（TDR）会在 GPU 无响应 >2s 时重置驱动，导致训练/仿真静默死亡。症状：日志突然停止，无 Python 错误。

**解决方案：nvidia-smi 周期性轮询**

原理：`nvidia-smi` 通过 NVML 通道查询 GPU，不干扰 CUDA 内核运行。定期调用会触发驱动 IO，重置 Windows TDR 计时器。

```bash
# 方法 1：batch runner 自带 watchdog（推荐）
python data/PGDA_SYNTH_DATASET_V1/tools/run_batch.py <batch_dir>
# 内部已集成 20s 间隔的 GPU 轮询

# 方法 2：独立 sidecar（手工跑仿真时使用）
# 开一个新终端先运行：
python data/PGDA_SYNTH_DATASET_V1/tools/gpu_keepalive.py --interval 15
# 然后在原终端跑 gprMax

# 方法 3：保持第二个终端运行 nvidia-smi
# 开一个新终端：
"E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe" -c "import subprocess,time; [subprocess.run(['nvidia-smi'],capture_output=True) or time.sleep(15) for _ in iter(int,1)]"
```

**registry 修改**（永久解决，需管理员权限）：
```
HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\GraphicsDrivers
  TdrDelay = 8  (默认 2 秒，改为 8 秒)
```
改后需重启。Batch runner 的 watchdog 轮询已足够覆盖 2 秒限制，一般不需改 registry。

## 自动化工具 (`.claude/`)

### Hooks (自动触发)
| Hook | 触发条件 | 作用 |
|------|---------|------|
| `training_process_guard` | Bash 训练命令 | 阻止重复训练进程 |
| `venv_python_guard` | Bash 训练/仿真命令 | 拦截错误 Python 解释器 |
| `checkpoint_sanity` | Bash 训练完成后 | 自动检查 checkpoint 完整性 |
| `report_on_eval` | Bash 评估完成后 | 提示生成报告 |
| `py_compile_on_edit` | Edit/Write .py | Python 语法检查 |
| `py_lint_on_edit` | Edit/Write .py | Ruff lint（尊重 ruff.toml，不强制 E501）|
| `npz_validation` | Bash 含 convert_pilot_to_training | NPZ 训练数据完整性验证 |

### 仿真数据管理脚本
| 脚本 | 用法 | 作用 |
|------|------|------|
| `tools/preflight_check.py` | `python ... <case_dir>` | 跑前 **19 项**预检 |
| `tools/generate_cases.py` | `python ... <batch_id> --n-cases 30 --depth-range 6 24` | 参数化 case 生成器（深度/地形/标签）|
| `tools/run_batch.py` | `python ... <batch_dir>` | 批量运行（自动 preflight + gprMax + QC，带 MSVC 注入、TDR 保护和续跑）|
| `tools/run_batch_standalone.py` | 双击桌面 `.bat` 或独立终端 | 独立批量仿真脚本（不依赖 Claude Code）|
| `tools/gpu_keepalive.py` | `python ... [--interval 15]` | TDR 预防 sidecar（nvidia-smi 周期性轮询）|
| `tools/after_run_qc.py` | `python ... <run_dir>` | 跑后质检（6 产物，含几何模型图）|
| `tools/promote_to_accepted.py` | `python ... <run_dir>` | GREEN → accepted_dataset |
| `scripts/run_batch003.bat` | 双击 | 独立 CMD 窗口启动 batch_003 仿真 |

### Skills (手动调用)
| Skill | 触发词 | 作用 |
|-------|--------|------|
| `/gprmax-usage` | "跑仿真""GPU没占用" | gprMax 仿真全流程指南，自动加载 |
| `/train-launch` | "开始训练" | 一键安全启动训练 |
| `/gpu-health` | "GPU状态" | GPU 健康快诊 |
| `/lolo-eval` | "评估 Line9" | LOLO-CV 集成评估 |
| `/exp-compare` | "对比结果" | 实验结果对比 |
| `/project-status` | "项目状态" | 项目快照 |
| `/data-check` | "检查数据" | 数据集完整性 |
| `/data-convert` | "转换数据" | 数据格式转换 |
| `/sim-batch` | "批量仿真" | 批量仿真启动 |
| `/sim-qc` | "仿真QC" | 仿真质量检查 |
| `/paper-figure` | "论文图" | 论文图表生成 |
| `/pkg-freeze` | "打包" | 成果冻结打包 |
| `/lit-review` | "文献调研" | GPR 文献调研 |
| `/train-log-analyzer` | "分析训练日志" | Loss 曲线诊断 |
| `/qc-report` | "QC报告" | 从 workspace 生成 QC 报告 |
| `/data-audit` | "审计数据" | NPZ 训练数据完整性验证 |

### Agents
| Agent | 调用方式 | 作用 |
|-------|---------|------|
| `training-launcher` | 默认 | 训练启动专家 |
| `pgda-experiment-auditor` | 默认 | 实验审计 |
| `simulation-auditor` | 默认 | 仿真数据审计 |
| `training-data-validator` | 默认 | 训练数据 NPZ 验证 |

### Workflows
| Workflow | 作用 |
|----------|------|
| `lolo-cv-full` | 完整 5折×3种子 LOLO-CV 流水线 |
