# PGDA-CSNet 仿真数据存储方案 & 质量标准 v1

## 一、目录结构（在 `data/PGDA_SYNTH_DATASET_V1/` 下）

沿用现有的 5 层治理结构，补充明确的质量分档规则。

```
data/PGDA_SYNTH_DATASET_V1/
├── 01_templates/           # 几何模板（不动）
├── 02_case_pool/           # case 定义（不动）
├── 03_runs/                # 仿真原始输出（.out HDF5 大文件）
│   ├── batch_xxx/          # 按批次组织
│   │   ├── CASE_001/
│   │   │   ├── raw/        # bscan.npy + raw.in + gprmax.log
│   │   │   ├── labels/     # interface_mask, y_soft 等
│   │   │   └── *.out       # HDF5 原始仿真（不提交到git）
│   │   └── ...
│   └── batch_summary.json  # 批次元信息
├── 04_qc/                  # 质量检查报告
├── 05_accepted_dataset/    # ✅ 通过质检，可直接用于训练的
│   ├── line9_style/
│   │   ├── mixed/          # 场景子目录
│   │   │   ├── CASE_001/
│   │   │   │   ├── input/    # raw_bscan.npy
│   │   │   │   ├── label/    # 标签
│   │   │   │   └── preview/  # 预览图
│   │   │   └── ...
│   │   └── ...
│   └── README.md           # 每个case的质量评分概要
├── 06_training_exports/    # NPZ 训练数据包
│   ├── trainset_v1_xxx/    # 每个训练集版本
│   └── ...
├── 07_failed_or_rejected/  # ❌ 废弃/失败的 case
├── 08_archives/            # .out 文件归档
└── DATA_QUALITY.md         # 质量标准文档
```

## 二、每个 case 的三维质量评分

每个仿真 case 按以下 3 个维度打分，决定它能否进入 accepted_dataset：

### A. 数据完整性（能否用于训练）

| 等级 | 定义 | 举例 |
|:----:|------|------|
| 🟢 **完整** | bscan + 全部 5 变体(raw/target_only/background_only/air_only/clutter_only) + 标签 全部有效 | — |
| 🟡 **部分** | 仅有 raw_bscan + 标签，变体缺失或 NaN | Batch1, Batch3, Pilot Val |
| 🔴 **不完整** | bscan 本身有 NaN，或标签缺失 | — |

### B. 基覆界面可见性（信号是否有意义）

| 等级 | 定义 | 标准 |
|:----:|------|:----:|
| 🟢 **可见** | 处理后界面 SNR > 10 dB | V4 Quick Test |
| 🟡 **微弱** | 处理后界面 SNR 0~10 dB | Batch1 深部, Pilot case_000001 |
| 🔴 **不可见** | SNR < 0 dB，完全沉没在噪声 | Pilot case_000002~005 (16.5~13.9m, -99~-118dB) |

### C. 几何合理性（场景是否有物理意义）

| 等级 | 定义 |
|:----:|------|
| 🟢 **合理** | 基岩面深度、材料参数、地形在物理范围内 |
| 🟡 **存疑** | 深度超出可探测范围，或材料参数过激进 |
| 🔴 **不合理** | 几何有 bug（天线埋入、域溢出等）|

## 三、综合判定规则

```
可入 accepted_dataset (05/) 的条件:
  A >= 🟡 部分  AND  B >= 🟡 微弱  AND  C >= 🟢 合理
  
可入 training (06/ NPZ) 的条件:
  A >= 🟡 部分  AND  B >= 🟡 微弱
  
需要标注"深度超限"但保留的条件:
  B = 🔴 不可见  BUT  有完整 HDF5 .out 原始数据
  → 放 03_runs/ 保留，不入 05_accepted
```

## 四、当前数据的质量评分

| 数据源 | A 完整性 | B 可见性 | C 合理性 | 综合 | 建议 |
|:-------|:-------:|:--------:|:--------:|:---:|:-----|
| **V4 Quick Test** | 🟡部分(仅raw) | 🟢可见(25.9dB) | 🟢合理(7m) | **🟢可用** | → 入accepted |
| **Batch 1** (12个) | 🟡部分(仅raw) | 🟡微弱(~-31dB) | 🟢合理 | **🟡存疑** | 需逐个评估 |
| **Batch 3** (20个) | 🟡部分(仅raw) | 🟡微弱(~-30dB) | 🟢合理 | **🟡存疑** | 需逐个评估 |
| **Pilot 001** | 🟡部分(仅raw) | 🟡微弱(-27dB FK) | 🟢合理(6.9m) | **🟡存疑** | 可入 |
| **Pilot 002** | 🟡部分(仅raw) | 🔴不可见(-89dB) | 🟢合理(16.5m) | **🔴不可用** | 深度超限 |
| **Pilot 003** | 🟡部分(仅raw) | 🟡微弱(-31dB FK) | 🟢合理(13.2m) | **🟡存疑** | 需评估 |
| **Pilot 004** | 🟡部分(仅raw) | 🔴不可见(-94dB) | 🟢合理(15m) | **🔴不可用** | 深度超限 |
| **Pilot 005** | 🟡部分(仅raw) | 🔴不可见(-70dB) | 🟢合理(13.9m) | **🔴不可用** | 深度超限 |

## 五、Git 提交规则

| 内容 | 提交？ | 说明 |
|:-----|:-----:|------|
| `bscan.npy` | ✅ | B-scan 数据 (501×128/180) |
| `raw.in` | ✅ | 仿真输入文件 |
| `labels/*.npy` | ✅ | 标签数据 |
| `preview/*.png` | ✅ | 预览图 |
| `metadata*.json` | ✅ | case 元信息 |
| `raw*.out` (HDF5) | ❌ | 太大(100KB×128=12.8MB/case)，放 archive |
| `geometry_v*.vti` | ❌ | 中间产物 |

## 六、你筛选后的操作流程

1. 你对照灰度 B-scan 图，告诉我每个 case **可用/存疑/不可用**
2. 我按你的判定更新质量评分
3. 🟢可用的 → `promote_to_accepted.py` 搬入 `05_accepted_dataset/`
4. 🟡存疑的 → 留在 `03_runs/` 不动，标记备注
5. 🔴不可用的 → 移入 `07_failed_or_rejected/`
6. 最后提交 `05_accepted_dataset/ + 03_runs/（仅npy+labels+preview）` 到远端
