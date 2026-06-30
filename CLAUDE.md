# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PGDA-CSNet** (Physics-guided Domain-Adaptive Clutter Suppression Network) — deep learning-based background clutter suppression for low-frequency UAV-borne Ground Penetrating Radar (UavGPR), targeting landslide bedrock interface detection.

The core challenge: low-frequency airborne GPR suffers from strong clutter (direct wave, ground reflection, platform altitude variation, horizontal bedding) that masks weak bedrock reflections. Only **one real survey dataset** exists with no clean ground-truth B-scan labels. The solution uses **gprMax FDTD simulation** to generate paired labeled data, combined with physics-guided branches and unsupervised domain adaptation to transfer to real data.

## Reference Document

The full research plan is in `UavGPR_机器学习背景杂波抑制全项研究计划.docx`.
Extract text via pandoc or python-docx (see docx skill).

## Current State (2026-07-01)

**Pilot-Train 仿真**: 17/100 场景已完成（含续跑机制: `scripts/run_pilot_train_resumable.py`）
**v4 LOLO-CV 配置**: 15 个 config 已生成（5 折 × 3 种子），脚本 `scripts/make_v4_loo_configs.py`
**Line9 LOLO-CV**: DP MAE=37.19ns, Pick Rate=96.6% (Line9 fold)

**X Pattern 实验结论**: 地表起伏是直达波×地面反射交叉的唯一成因，平坦地表 + 固定 TX-RX 高度会产生对称反射路径交叉；起伏地形消除此现象（`data/gprmax_experiments/` 含 M0→M1→M2 渐变链）

**工作空间**：
| 路径 | 内容 |
|------|------|
| `workspace/transfer_20260627_142748/.../PGDA_CSNet_v0_9_6_SEARCH_WINDOW_GUARD/` | 主训练/评估工作目录（含 scripts/、configs/、outputs/） |
| `data_corrected_v1_4_terrain_direction/` | 当前训练数据（实测+标注） |
| `data/simulation_pretrain_v2/` | 20 Pilot-Mini 仿真场景 |
| `data/simulation_pretrain_v3/` | Pilot-Train 100 场景（待仿真）|
| `data/simulation_pretrain_v3_check/` | 单场景验证测试数据 |
| `data/营山/` | 原始实测数据 + 钻孔资料 |
| `uavgpr_simlab/` | SimLab 仿真工具 |

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

## Pipeline Fixes (2026-06-29)

### Mask Depth-Time Conversion
- **Bug**: `interface_mask_bscan.npy` 缺少空气走时（`2×(UAV-地面)/0.3`）
- **Fix**: `scene_variant_writer.py` — 加了 `air_twt` + `compute_cover_velocity()` 从材质 eps_r 动态算速度
- **Velocity**: cover 材质平均 eps_r ≈ 19 → v ≈ 0.069 m/ns

### Domain Grid
- **Bug**: `domain_y / dx` 非整数（如 48.309/0.05=966.18）
- **Fix**: `scene_world_generator.py` — `math.ceil(val/dx)*dx`

### PML
- **Fix**: `.in` 文件加 `#pml_cells: 10 10 0 10 10 0`

### 数据转换 (convert_pilot_to_training.py)
- **FFT ringing** → `np.interp` 线性插值
- **P99** → 全 100 场景全局统一而非每 case 独立
- **padding** → 零填充，label_weight=0
- **y_soft** → 高斯软标签 (Gaussian σ=8, 峰值=1.0)
- **y_mask 时间错位** → 加了空气走时，用有效平均速度 (~0.069 m/ns)

## SimLab (仿真工具)

**位置**: `D:\Claude\PGDA-CSNet\uavgpr_simlab\`  
**安装**: `pip install -e D:\Claude\PGDA-CSNet\uavgpr_simlab` (已安装在 gprMax .venv)  
**gprMax**: v3.1.7 at `E:\gprMax\gprMax-v.3.1.7`，Python: `.venv\Scripts\python.exe`  
**GPU**: `SafeGprMaxRunner` 自动注入 MSVC + CUDA 路径到 PATH，无需 conda 或 vcvars

**⚠️ GPU 复杂模型**: 带介电平滑的复杂模型（643+ box）直接 `python -m gprMax` 会因 MSVC INCLUDE/LIB 路径缺失导致 nvcc 编译失败。**必须使用 SafeGprMaxRunner**。

**⚠️ .in 文件注释语法**: gprMax v3.1.7 `check_cmd_names` 会解析所有 `#` 行并分割 `:`。纯注释行（无冒号如 `# --- geometry boxes ---`）导致 IndexError。必须用 `#:` 或移除该行。

### 常用命令
```bash
# 生成 SceneWorld 数据集
python -m uavgpr_simlab.cli generate --plan configs/run_plan_3060_pilot_v1.yaml --workspace workspace/my_run --count 20

# 几何预检（快速验证 .in 文件, ~3s）
python -c "
from uavgpr_simlab.core.runner import run_geometry_dry_run
r = run_geometry_dry_run('path/to/raw.in', python_exe='E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe')
print(r.status)
"

# 批量运行（manifest CSV → GPU 仿真）
python scripts/run_batch_safe_3060.py --manifest workspace/my_run/<name>_manifest.csv --dry-run
python scripts/run_batch_safe_3060.py --manifest workspace/my_run/<name>_manifest.csv --variants raw,target_only
```

### 关键配置文件
| 文件 | 用途 |
|------|------|
| `configs/run_plan_3060_pilot_v1.yaml` | Pilot-Mini 旧配置 |
| `configs/run_plan_3060_pilot_train_v1.yaml` | Pilot-Train 100 场景生产配置 |
| `configs/gpu_train_v4_pilot_mixed.json` | v4 训练配置（100 仿真场景） |
| `scripts/run_pilot_train_resumable.py` | 续跑脚本（SafeGprMaxRunner，跳过已完成 case） |
| `scripts/postprocess_simulation_batch.py` | .out → raw_bscan_native.npy 批量后处理 |
| `scripts/make_v4_loo_configs.py` | LOLO-CV 5折×3种子 config 生成器 |
| `configs/default_app.yaml` | 应用默认值（已更新为本地路径） |
| `configs/environment_3060_laptop.yaml` | 3060 环境配置 |
| `scripts/run_batch_safe_3060.py` | 批量运行脚本 |
| `scripts/convert_pilot_to_training.py` | 新版数据转换（线性插值 + 全局P99 + 软标签） |

### 数据转换
```bash
# 新版转换：.npy → NPZ 训练窗口
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" scripts/convert_pilot_to_training.py
```

## Hardware & Data Constraints

- **Current (开发)**: RTX 3060 Laptop **6 GB** — dx=0.05m 已验证可用（~1.2GB VRAM/场景）
- **后续 (生产)**: RTX 4090 Laptop 16 GB — 可上 dx=0.025m
- **配 YAML config profile 切换**，不硬编码 GPU 参数
- **实测数据**: 营山 6 条测线 (L3/L6/L7/L9/L1/X1) × 3 功率档，501 samples × 700ns
- **钻孔**: ZK07 (~21-23m), ZK09 (~20-23m), 多个浅钻孔 (~9-17m)，数据在 `data/营山/`
- **决策**: 仿真用 gprMax 原生时域 Ricker 100 MHz，不做 SFCW 转换。当前阶段暂不做外部杂波（电线/树木/建筑），聚焦地质核心

## Pilot 数据集方案

详见 `docs/PILOT_MODEL_PLAN.md`。

**已完成**: Pilot-Mini 20 场景，仿真数据在 `data/simulation_pretrain_v2/`。

**进行中**: Pilot-Train 100 场景，计划配置在 `configs/run_plan_3060_pilot_train_v1.yaml`。

**仿真参数**: 网格 ~1458×967×1, dx=0.05m, 时窗 700ns, 频率 100 MHz Ricker, #pml_cells 显式声明
**变体**: raw / background_only（target_only 因无外部杂波而移除）
**杂波**: 外部杂波暂关，水田/饱和带/地形保留
**材料 σ**: 0.002-0.015 S/m

## Architecture: PGDA-CSNet (v1_9d_mambavision_hybrid)

网络以原始 B-scan 为输入，输出干净 B-scan + 杂波图 + 不确定度图。已实现并训练。

### Backbone
- **Residual Dense U-Net** (4级 encoder-decoder, 通道 32→64→128→256→512)
- **Multi-scale receptive field**: 并行 3×3, 5×5, atrous conv (d=1/2/4)
- **Residual learning**: 预测杂波图 C_hat；clean = Input – C_hat

### Physics-Guided Branches
1. **f-k Spectral Branch**: 2D FFT per patch → 可学习频谱掩码 → IFFT 融合
2. **SVD Low-Rank Branch**: SVD 分量图 → Soft Singular Shrinkage (可微)
3. **Geometric Conditioning (FiLM)**: UAV 高度/地表高程/坡度 → scale/shift 调制

### Output Heads
- `Y_clean`: 预测干净 B-scan
- `C_hat`: 杂波图 (可解释性)
- `σ_hat`: 不确定度图 (MC Dropout / Deep Ensemble)

## Training Strategy

| Stage | Status | 说明 |
|-------|--------|------|
| 0: Baseline | ✅ 完成 | P0-3 Center Fusion: MAE=3.268 |
| 1: v3 Supervised Pretrain | ✅ 完成 | 20 Pilot-Mini 场景, LOLO-CV Line9: DP MAE=37.19ns |
| 2: Pilot-Train 100 场景 | 🔄 进行中(17/100) | 续跑脚本 `scripts/run_pilot_train_resumable.py` |
| 3: v4 LOLO-CV | ⏳ Pilot-Train 完成后 | 15 config 已生成, 用新数据重跑 5 折 × 3 种子 |

### Data Split Discipline (Critical)
- 按**测线**分割，非随机 patch
- **Leave-one-line-out**: 至少一条完整测线用于测试
- 同测线不同功率档 → 同一 split

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
- **TDR 风险**: Windows GPU 超时检测会在 GPU 无响应 >2s 时重置驱动，导致训练静默死亡。症状：日志突然停止，无 Python 错误
- **温度**: 训练时 80-87°C 正常，>90°C 有热降频风险
- **建议**: 训练前检查 GPU 状态 (`/gpu-health`)，确保无其他进程占用显存

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

### Skills (手动调用)
| Skill | 触发词 | 作用 |
|-------|--------|------|
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
