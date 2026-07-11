# `MyNet_V15正式数据治理增量包_d6f9108.zip` 之后完成的工作

## A. 现有仿真数据全面审计

对现有 33 个注册仿真 case 做了逐 case 审计，结论不是“继续直接训练”，而是重新分级隔离：

- 12 个 `DEV_REGRESSION_ONLY`
- 11 个 `REJECT_POSITIVE_TRAINING`
- 6 个 `REVIEW_ONLY`
- 2 个 `DEV_HARD_POSITIVE_CANDIDATE`
- 1 个 `STRESS_TEST_ONLY`
- 1 个 `QUARANTINE`
- 1 个 `SMOKE_TEST_ONLY`

核心问题包括：Line9 条件泄漏、横向物理尺度不一致、scene metadata 复制、旧生成器传播时间/坐标/标签偏移缺陷、过度平滑与低秩域差、缺少匹配控制和合法负样本。所有旧 case 继续 `train_allowed=false`。

外层交付中的 `MyNet_现有仿真场景审计包_d6f9108.zip` 保留完整报告、逐 case CSV、生成器快照和可视化证据。

## B. 建立 gprMax V2 物理合同

新增 `PGDA_SIMULATION_CONTRACT_V2`，固定并审计：

- 256 traces
- 0.09 m trace spacing
- 22.95 m 有效扫描跨度
- canonical 训练轴 0–700 ns、501 samples、1.4 ns
- FDTD 网格 0.0225 m
- 扫描步长 4 cells
- 暂定 Tx/Rx separation 0.18 m（8 cells，待硬件确认）
- 100 MHz Ricker
- PML 与额外 guard 独立建模

## C. 新建四个控制场景

- `CTRL01_FLAT_SHALLOW_LOWLOSS_POS`：2 m AGL、7 m 浅层平界面。
- `CTRL02_FLAT_DEEP_MODERATE_POS`：8 m AGL、14 m 深层平界面。
- `CTRL03_SMOOTH_INTERFACE_POS`：5 m AGL、平滑曲面。
- `CTRL04_MATCHED_BACKGROUND_NEG`：与 CTRL02 匹配的无基覆对比控制。

每个正控制包含 full、no-basal-control、air-reference、geometry-check、manifest、哈希和逐道几何/到时数组。负控制保持零目标语义，不把参考到时伪装成目标标签。

## D. 完善生成、运行、后处理与导出链

新增或重写：

- `pgdacsnet/simulation_v2.py`
- `scripts/generate_physical_sim_v2.py`
- `scripts/validate_physical_sim_v2.py`
- `scripts/check_gprmax_runtime_v2.py`
- `scripts/run_physical_sim_v2_controls.py`
- `scripts/postprocess_physical_sim_v2.py`
- `scripts/export_sim_training_npz.py`

工程语义包括：逐道 AGL 跟随、分层材料连续覆盖、网格吸附、双基地分层传播到时、匹配控制差分、HDF5 元数据核验、训练时间轴重采样、负样本零 mask、禁止 V2 resize/padding 冒充物理采样。

## E. 第二轮 gprMax V2 物理/代码修复

在 39e178f 控制阶段之后继续修复：

1. solver 请求时窗与 canonical 训练时轴彻底分离；HDF5 必须证明最后样本覆盖 700 ns。
2. PML 外安全 guard 从 10 提升到 20 cells，并同步域边界和测试。
3. `Iterations` 必须与接收器数据长度一致，拒绝短输出。
4. 平层到时定义为水平分层双基地参考；曲面只称 columnar layered search reference。
5. CTRL02 正样本与 CTRL04 matched negative 建立双向 provenance 和逐数组一致性检查。
6. 生成器记录 commit、dirty 状态、seed、case spec hash 和不同空间尺度。
7. 安装脚本改为官方 release/`conda_env.yml` 的隔离环境流程。
8. 增加 Line9 泄漏扫描、legacy quarantine 和 formal-training gate 测试。

## F. 验证状态

本次重新打包时实际复跑：

- `compileall`：PASS
- `validate_physical_sim_v2.py`：4/4 PASS
- 三个工程测试文件：40 passed

历史完整环境曾记录：138 passed、84/84 V15 protected hashes unchanged；该完整测试依赖 V15 数据和临时物化的 66 个旧 quarantine 资产。本精简包没有重复携带大数据，因此不把历史 138 项结果冒充为本次复跑结果。

## G. 尚未完成

- gprMax 真实 geometry view
- CFL `dt` 与实际 HDF5 审计
- CTRL01–04 FDTD 求解
- PML 反射、振铃、相位极性和能量可见性验证
- 14 m 中等损耗控制的真实可见性
- 硬件真实 Tx/Rx 几何确认
- CTRL04 正式负样本放行
- 24-case pilot 和正式训练
