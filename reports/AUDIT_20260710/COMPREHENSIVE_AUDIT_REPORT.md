# PGDA-CSNet 全面审计报告

审计日期：2026-07-10  
审计版本：`1818b25`  
范围：仓库、LFS、实测数据、仿真数据、人工分档、训练/评估链路、配置、测试与复现性。

## 总体结论

当前代码的核心单元测试和 GPU 前向能够通过，32 个新增仿真 case 的数组也完整可读；但数据治理和论文实验合同存在多项阻断性问题。现版本不应直接开始新的正式论文训练，也不能把已有“Line9 strict holdout”结果视为严格独立验证。

建议暂时冻结训练，先完成 P0 修复：建立无 Line9 标签泄漏的论文数据集、补齐实测数据合同、统一人工分档、让仿真训练失败时显式报错，并重新定义负样本。

## P0 阻断问题

### 1. Line9 holdout 被仿真标签污染

`05_accepted_dataset` 的 13 个 case 元数据明确声明：标签由 `Line9 V14` 实测标签中心生成。所有 case 的 `scene_world.json` 和 `design_metrics.csv` 又完全相同，均引用 Line9 标签源。若这些仿真参与训练，Line9 的曲线形状、时间分布和标签先验已经进入训练集，因此不能再称为严格 Line9 holdout。

处理：这些 case 只能标记为 `line9-conditioned development data`，不得进入 Line9 主测试实验。论文测试 Line9 时，仿真几何必须由其他测线、钻孔范围或独立随机先验生成。

### 2. 当前仓库不能复现训练和全线评估

`data_corrected_v1_4_terrain_direction` 只有 78 个 `windows/*.npz`，缺少训练器硬依赖的 `window_index.csv` 和评估器硬依赖的 `lines/*.npz`。`data/simulation_pretrain_v1`、主要 warm-start checkpoint 和 `configs/paper_splits_v1_6.json` 也缺失。

结果：`scripts/check_dataset.py` 报 `RAW_ONLY_SCHEMA_BAD 2`；正式训练和 `eval_full_line.py` 均无法按仓库内容运行。

### 3. 多个“混合训练”会静默退化为纯实测训练

15 个配置把 `05_accepted_dataset` 作为 `sim_data_root`，但该目录没有 `window_index.csv` 和训练 NPZ。训练器在找不到索引且 `sim_train_lines=[]` 时不会报错，只是不创建仿真 loader。另有 24 个混合配置指向完全不存在的仿真目录。

处理：当 `sim_batch_ratio > 0` 时，缺目录、索引、样本或 loader 必须立即失败；accepted case 必须先通过单一、受测的转换器导出训练 NPZ。

### 4. 人工分档自相矛盾

Batch 1 的 12 个 case 在 `AUDIT_MANIFEST.md` 中标为“存疑”，同时又作为“可用”存在于 accepted 目录。两处 raw 和五类关键标签逐文件完全相同。人工审计清单还写着 Batch 1/3 “未上传本体”，与当前仓库不符。

处理：建立唯一的 `human_audit_manifest.csv`，逐 case 保存 `label/auditor/date/method/note/source_sha256`。禁止用目录名或自动 QC grade 代替人工标签。

### 5. 没有真正负样本

78 个实测窗口中只出现 `status_code=1/2`，没有 `status_code=0`；每一列标签都非空。于是 presence head 没有负类，`presence_negative_weight` 无实际作用；global no-target head 的监督目标恒为“有目标”，无法学习拒识。

处理：增加经人工确认的无界面/不可判窗口，或明确把不可用仿真转为只训练拒识头的负样本，且不得给错误的界面曲线监督。

## P1 高风险问题

### 6. V4 标签语义与其他仿真不一致

V4 的 hard/soft 标签中心为几何到时约 `221.6 ns`，实际主可见相位峰为 `245.0 ns`，偏移 `23.4 ns`。accepted/Batch 1 使用的是 visible-phase hard label，二者不能直接混合。

处理：保留 V4 raw，但按统一 visible-phase 规则重建标签后才能进入训练。

### 7. 曲线头评估混用了分布与分割指标

启用 curve head 时，`eval_full_line.py` 将按时间归一化的 curve distribution 当作 `pred`，随后计算 soft Dice、IoU、BCE，并保存为 `pred_softmask.npy`。这些指标与像素 mask 不同尺度，不能和 baseline mask 指标直接比较。curve probability 与 sigmoid center map 的线性融合也未校准。

处理：路径 MAE/pick rate 使用 curve distribution；分割指标只使用 mask sigmoid；产物和表格分列命名。

### 8. 配置守卫存在硬错误且严格模式未启用

- 3 个配置违规使用已剔除的 X1 做 validation。
- 1 个 smoke JSON 语法无效。
- 5 个有 test 的配置没有 validation，best checkpoint 实际按 train loss 选择。
- 43 个配置全部缺少 `run_type`，许多严格 split 检查不会触发。
- `paper_split_file` 在代码中从未读取，属于无效配置字段。

### 9. 飞高先验错误

arrival prior 在索引没有高度字段时默认 `2.4 m`，而营山实测说明为约 `11-12 m`。主配置启用了 arrival prior，但未提供高度字段或正确默认值，导致早期地下信号约束与实际空中传播时间不一致。

### 10. accepted 元数据不可追溯

13 个 accepted case 的 `scene_world.json` 和 `design_metrics.csv` 哈希完全相同，而 raw、标签和预览不同。`promote_to_accepted.py` 又优先复制模板标签/元数据而非 case-local 产物，并把 `LINE9_TERRAIN_*` 错分到 `generic_smooth`。README 所述 `accepted_manifest.csv` 也不存在。

### 11. Batch 3 不能整批晋级

20 个 Batch 3 case 数值均完整，但按仓库当前 QC 规则只有 3 个 GREEN、6 个 YELLOW、11 个 RED。当前人工“整批存疑”可以保留，但不能整批加入训练。逐 case 建议见 `SIM_CASE_RECOMMENDATIONS.csv`。

### 12. 仿真多样性明显不足

Batch 1 标签曲线两两相关系数中位数约 `1.0000`，raw 中位相关约 `0.9997`；Batch 3 raw 中位相关约 `0.9930`。case 数量不能等价为独立样本量，直接扩窗会夸大数据规模并增加模板过拟合。

## P2 工程与文档问题

- 仓库声称含 Pilot validation 001-005，但实际不存在；三个不可用 Pilot 仍无法复核。
- `03_runs` 没有几何模型、预览、raw.in、case 元数据，违反项目“三件套”要求，也无法独立复现。
- Batch 3 目录名写 24 cases，实际只有 20。
- 营山 README 写 20/30/36 dBm，当前清单和训练窗口仅能确认 36 dBm。
- 多个转换脚本硬编码旧电脑的 `D:/Claude`、`E:/gprMax` 或桌面路径。
- 核心项目没有 requirements/lock 文件；目标环境缺 `pytest`，Matplotlib 保存图片触发原生 DLL 异常。
- 根目录缺统一 pytest 配置，直接运行会误收集运行脚本并失败。
- 15 个缓存/字节码文件被提交进 Git。
- `after_run_qc.py` 的 `dead_trace_ratio` 实际统计相邻道相关性低于 0.8，不是真正的死道比例；polarity mismatch 也不影响最终 grade。

## 通过项

- Git 与远端一致，Git LFS `fsck` 通过。
- 32 个新增 case 共 992 个 NPY 均可读取、无 NaN/Inf。
- 32 个 B-scan 均为 `5937x128`，标签时间轴均为 `0-700 ns` 且严格递增。
- 关键 mask 均为 `501x128`、范围 `[0,1]`、每列非空。
- 实测 78 个窗口均可读取、无 NaN/Inf；同测线重叠窗口逐元素一致。
- 显式运行核心测试：`110 passed, 1 warning`。
- Python 编译检查通过；RTX 5070 CUDA 前向 smoke 通过。

## 建议修复顺序

1. 建立 `dataset_contract_v2`：补齐 lines/index、唯一人工清单、哈希、来源与标签语义。
2. 将 Line9-inspired 仿真隔离为开发/消融数据，重新生成与测试线无关的仿真。
3. 修复 mixed-loader fail-fast、X1 配置、validation、run_type 和 curve 指标。
4. 统一 visible-phase 标签；V4 先重标，Batch 3 逐 case 决策。
5. 增加真实负样本和高度/地形字段，再重新训练 presence/no-target/arrival prior。
6. 补齐依赖锁、路径参数化、checkpoint 与可复现运行说明后，再启动正式多 seed 论文实验。

## 审计预览

- `batch1_12case_visual_audit.png`：Batch 1 统一增益与标签叠加。
- `batch3_20case_visual_audit.png`：Batch 3 统一增益与标签叠加。
- 红线表示 `target_visible_phase_time_ns`，预览仅用于审计，不替代原始几何和人工结论。
