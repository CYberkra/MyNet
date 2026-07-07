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

## Current State (2026-07-04)

**batch_001**: 12 case (LINE9_STYLE_001~010 + TERRAIN_011~012)  
全部跑完 128/128 道 → QC GREEN → promote 到 `05_accepted_dataset/` ✅  
**trainset_v1_0_line9_style_12cases**: 已导出（13 case × 128 道 = 1664 trace，含原始 LINE9_STYLE_V1）  
**batch_002_depth_30cases**: 30 深度变体 case，**已用修后生成器重生成**（坐标修正，天线不再埋入地层），preflight 19项全 PASS ✅  
**batch_003**: 24 case（浅层干扰+中深度+hard negative）从 WSL 迁入，**仿真进行中 ✅**（独立 CMD 窗口运行）

**V3.x 控制实验系列**:
- **V3.2**: 宽域300m + PML60，验证X来自侧边界反射 ✅
- **V3.3**: 宽域+平滑起伏+`#triangle`介电平滑，迹间差异46.93
- **V3.4**: `#triangle averaging=y` 替代 H5，避免台阶效应，迹间差异45.37
- **V3.5**: +weak_cover_band，迹间差异45.37
- **V3.6**: +浅层扰动，完整y_soft/geom_onset/visible_phase标签
- **V3.7**: 浅（10m, 95/128迹）/深（18m, 128迹）✅

**LINE9_LABEL_INSPIRED_V1**: 480m域 Line9 地形追随模型，128 道，trace_var=44.53。  
标签匹配 Line9 V14 实测仅 **0.06ns misfit**（68% trace < 0.1ns），已在 `PGDA_SYNTH_DATASET_V1/` 完成建档。  
预检 18 项全 PASS，QC GREEN（target_local_peak_median=0.909, support=83.6%）。

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

**跑前预检 19 项**（不全 PASS 不进 gprMax）：
pml_cells, domain_grid, H5禁止, noair禁止, triangle平滑, 所有三角形必须 averaging=y, `#`注释语法, source/rx不在PML(x/y上下边界), 侧边界>700ns(空气速度+地层速度), 标签完整性, generator记录, 标签非平坦(range>0.5ns), TX/RX按几何交点检测是否埋入介质, 三角形不超出domain_y

**结果分级**：
GREEN_ACCEPTED → accepted_dataset / YELLOW_REVIEW → 人工审查 / RED_REJECTED → failed / GRAY_DEBUG_ONLY → 诊断

### 自动化工具

```bash
# 跑前预检（必做）
python data/PGDA_SYNTH_DATASET_V1/tools/preflight_check.py <case_dir>

# 跑后质检（一次性出6个QC文件）
python data/PGDA_SYNTH_DATASET_V1/tools/after_run_qc.py <case_run_dir>

# GREEN → promote
python data/PGDA_SYNTH_DATASET_V1/tools/promote_to_accepted.py <case_run_dir>
```

### 当前状态

| 模板 | 状态 | 存放 |
|------|------|------|
| V3.4/V3.5/V3.6 | 历史参考模板 | 01_templates/ |
| LINE9_LABEL_INSPIRED_V1 | 当前主模板，已 promote | 01_templates/ + 05_accepted_dataset/ |

详细规则见 `data/PGDA_SYNTH_DATASET_V1/00_docs/`。

**X Pattern 实验结论**: X = 空气耦合直达波 A + 地表反射 S 的早时时空交叠。根本成因：空气（ε_r=1, v≈0.3m/ns）与地面介质（ε_r≈6-19, v≈0.07-0.12m/ns）的巨大波速差→A 与 S 双曲曲率不同→交叉形成 X。去掉空气层（`case_000001_no_air_fill`，20 迹对照）后直达波被彻底消除：峰值从 90→30，均值迹零点从 1409→10。起伏地形仅破坏对称性但非 X 根源。`data/gprmax_experiments/` 含 M0→M1→M1b→M1c→M2 渐变链 + `case_000001_no_air_fill` 诊断。**生产模型必须保留空气层**，PGDA-CSNet 需学习在有强 A 的情况下同时分离 A/S/G。

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

### 常用命令
```bash
# 三件套（仿真跑完立即执行）
python .claude/skills/sim-report/sim_report.py <输出目录>

# case 生成（深度变体 batch）
python data/PGDA_SYNTH_DATASET_V1/tools/generate_cases.py batch_xxx --n-cases 30 --depth-range 6 24

# 生成 SceneWorld 数据集
python -m uavgpr_simlab.cli generate --plan configs/run_plan_3060_pilot_v1.yaml --workspace workspace/my_run --count 20

# 几何预检（快速验证 .in 文件, ~3s）
python -c "
from uavgpr_simlab.core.runner import run_geometry_dry_run
r = run_geometry_dry_run('path/to/raw.in', python_exe='E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe')
print(r.status)
"

# 批量运行（PGDA_SYNTH_DATASET_V1 模式）
python data/PGDA_SYNTH_DATASET_V1/tools/run_batch.py <batch_dir>
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
| `data/PGDA_SYNTH_DATASET_V1/tools/run_batch.py` | 批量运行（PGDA 治理模式） |
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

## Training Strategy

| Stage | Status | 说明 |
|-------|--------|------|
| 0: Baseline | ✅ 完成 | P0-3 Center Fusion: MAE=3.268 |
| 1: v3 Supervised Pretrain | ✅ 完成 | 20 Pilot-Mini 场景, LOLO-CV Line9: DP MAE=37.19ns |
| 2: batch_001 仿真训练 | ✅ 完成 | 12 LINE9-style 场景，50 epoch → 实测 pick rate 0%（域偏移）|
| 3: FiLM v1.8b | ✅ 完成 | +terrain features → MAE 256ns（略改善）|
| 4: UDA 训练 | ✅ 完成 | 域损失 0.91→0.43，对抗训练有效 |
| 5: Pilot-Train 100 场景 | 🔄 进行中(17/100) | 续跑脚本 `scripts/run_pilot_train_resumable.py` |
| 6: batch_002 深度多样仿真 | ✅ 完成 | 30 场景已用修后生成器重生成 |
| 7: batch_003 深度泛化仿真 | 🔄 进行中 | 24 case（浅层干扰+中深度+hard negative）|
| 8: v4 LOLO-CV | ⏳ batch_003 完成后 | 15 config 已生成, 用新数据重跑 5 折 × 3 种子 |

## Architecture Reference

**PGDA-CSNet (v1_9d_mambavision_hybrid)**: 网络架构代码在 `pgdacsnet/model_raw_unet.py`。
核心组件见 [[pgda-csnet-architecture]] — ConvNeXtStage 编码器 + DilatedBottleneck + AxialSSM + MetadataFiLM + GatedSequenceBlock。
训练入口: `train_raw_only.py` / `resume_train.py`，`build_model(cfg)` 构建完整管线。

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
