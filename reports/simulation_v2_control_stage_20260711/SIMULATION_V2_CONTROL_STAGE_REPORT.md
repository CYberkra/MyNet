# PGDA Simulation V2 物理控制阶段报告

日期：2026-07-11  
工作基线：重建的 `d6f9108` 精简交付版  
阶段状态：**四个 control 已完成 GPU 求解、后处理与自动审阅；正式训练仍禁止。**

## 1. 本轮目的

本轮只修正仿真基础链：几何、材料、传播时间、空间尺度、标签语义、正负对照和运行审计。按用户要求，本阶段不加入电线、树木、水体、测线边缘或随机外部杂波。

## 2. 已落实的官方 gprMax 约束

- 空间使用 SI 单位；模型原点位于左下角。
- 2-D 控制采用 x-y 平面、z 方向单 cell、z 极化 Hertzian dipole。
- `#pml_cells` 顺序固定为 `x0 y0 z0 xmax ymax zmax`，本轮为 `20 20 0 20 20 0`。
- 源/接收步长和 Tx/Rx 偏移必须是 `dl` 的整数倍。
- `dl=0.0225 m` 在 100 MHz、最大 εr=11.2、按 3 倍中心频率检查时满足最小波长约 10 cells 的要求。
- gprMax 自己使用 CFL 时间步；501×1.4 ns 只在 HDF5 输出后重采样，不冒充 solver step。
- 扫描运行采用 `-n 256 --geometry-fixed`，仅适用于静态几何和简单移动源/接收器。
- HDF5 后处理检查 `/rxs/rx1/Ez`、`dt`、`dx_dy_dz`、`srcsteps`、`rxsteps` 和 gprMax 版本。

## 3. 新物理合同

- 256 道，0.09 m/道，总跨度 22.95 m。
- 求解请求时窗 701 ns；训练 canonical 时间轴为 0–700 ns、501 点、dt=1.4 ns。真实 HDF5 必须覆盖 canonical 终点后才允许重采样。
- FDTD 网格 `dl=0.0225 m`；0.09 m 为 4 cells，暂定 Tx/Rx 偏移 0.18 m 为 8 cells；PML 20 cells 外另留 20 cells guard。
- 天线逐道满足 `antenna_y = ground_y + AGL`。
- 分层顺序固定为 cover → weathered → bedrock，无零厚度、无材料空洞、无越界。
- 平层几何到时使用空气、覆盖层、风化层的精确水平分层双基地参考；曲面场景使用 columnar layered 搜索参考，不再把它宣称为精确 specular-ray 到时。
- 几何到时只作物理先验，训练中心必须在求解后由 `full_scene - no_basal_contrast_control` 提取可见相位。
- background-only 负样本在后处理中生成严格零 mask，且不会自动得到训练放行。

## 4. 四个控制场景

| Case | 类型 | AGL | 界面 | 预期用途 | 静态结果 |
|---|---|---:|---|---|---|
| CTRL01 | 浅层低损耗正样本 | 2 m | 平界面 7 m | 坐标、到时、相位、导出链基准 | 145.49 ns，通过 |
| CTRL02 | 深层中等损耗正样本 | 8 m | 平界面 14 m | 主深度衰减和到时基准 | 344.12 ns，通过 |
| CTRL03 | 平滑曲面正样本 | 5 m | 10 m 基准+缓坡+正弦 | 逐道几何和真实横向尺度 | 194.34–248.98 ns，通过 |
| CTRL04 | 匹配 background-only 负样本 | 8 m | 与 CTRL02 上覆几何相同、取消基覆反差 | confirmed negative 语义 | 通过 |

四个 case 均满足 `line9_conditioned=false`、`formal_training_allowed=false`。

## 5. 关键实现文件

- `pgdacsnet/simulation_v2.py`
- `data/simulation_contract_v2/simulation_contract_v2.json`
- `data/simulation_contract_v2/materials_v1.json`
- `data/simulation_contract_v2/control_cases_v1.json`
- `scripts/generate_physical_sim_v2.py`
- `scripts/validate_physical_sim_v2.py`
- `scripts/check_gprmax_runtime_v2.py`
- `scripts/run_physical_sim_v2_controls.py`
- `scripts/postprocess_physical_sim_v2.py`
- `scripts/export_sim_training_npz.py`
- `tests/test_simulation_v2.py`

## 6. 已完成校验

- 四个 control 静态 preflight：4/4 通过。
- 四个 control 已在 RTX 5070 上使用 gprMax 3.1.7 完成求解、HDF5 合并和后处理。
- CTRL04 已生成严格零目标掩码；CTRL01-03 均有 full/no-basal/air 匹配输出。
- CTRL03 曾暴露 P95 连续性门禁遗漏，现已改为几何锚定连续路径与最大跳变联合检查。
- 生成器、分层到时、CFL 后重采样、可见相位提取、负样本后处理、`--geometry-fixed` 运行计划测试：通过。
- 权威 V15/canonical 数据已从交付包物化，78 个 canonical 窗口由全线数组重建用于验证。
- 完整测试套件：138 passed，0 failed，0 errors。
- 84/84 个 V15 受保护 NPZ SHA256 未变化。
- `compileall`：通过。

## 7. 当前明确未完成

自动求解与后处理已完成，但仍未完成：

- 目标可见性、深层衰减和边缘暂态的人工物理审阅；
- 实际硬件 Tx/Rx 几何的确认；
- 人工审计 manifest 中的正式数据资格决策；
- 任何 case 的正式训练批准。

当前完整结果以 `POSTRUN_REVIEW.md` 与 `postrun_review/` 为准。

## 8. 下一运行门禁

1. 在独立环境安装并记录 gprMax 3.1.7。
2. 先执行四个 case 的 geometry-only，并人工检查几何视图。
3. 先跑 CTRL01；若到时/相位/输出格式通过，再跑 CTRL02–04。
4. 正样本必须完成 full/no-basal/air 三套匹配运行。
5. 可见相位逐道残差去除统一相位偏移后应不超过 5.6 ns；平界面应优先达到 2.8 ns 级别。
6. CTRL04 必须保持严格零目标 mask。
7. 人工签字前 `metadata_trusted=false`、`formal_training_allowed=false` 不得修改。

## 9. 尚需现场确认的单一参数

`tx_rx_offset_m=0.18` 目前是暂定值。它已经被显式标记为 provisional，不会被伪装成已知硬件参数。取得实际 UAV-GPR 天线发射/接收几何后，需要重新生成四个 controls 并重跑全部到时测试。
