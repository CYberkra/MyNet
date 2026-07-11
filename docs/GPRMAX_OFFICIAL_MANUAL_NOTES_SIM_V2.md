# gprMax 官方手册约束摘要：PGDA Simulation V2

本文件记录本轮实现实际采用的官方约束。它不是替代官方手册的二手说明。

## 官方来源

- Input commands: https://gprmax.readthedocs.io/en/latest/input.html
- Guidance on GPR modelling: https://gprmax.readthedocs.io/en/latest/gprmodelling.html
- Introductory 2-D examples: https://gprmax.readthedocs.io/en/latest/examples_simple_2D.html
- Output HDF5 structure: https://gprmax.readthedocs.io/en/latest/output.html
- Antenna user library: https://gprmax.readthedocs.io/en/latest/user_libs_antennas.html
- Official repository: https://github.com/gprMax/gprMax

## 已落实的规则

1. **单位和坐标**：空间单位为 m，时间单位为 s，频率单位为 Hz；原点位于模型左下角。V2 使用 `x` 为测线方向、`y` 向上、`z` 为单细胞不变方向。
2. **网格取整**：gprMax 会把空间坐标转换为 FDTD cell。所有扫描步长和 Tx/Rx 偏移均设计为 `dl` 的整数倍，避免每道取整造成几何漂移。
3. **空间离散**：官方建议最小传播波长至少由 10 个 cell 表示；Ricker 脉冲在中心频率以上 2–3 倍仍有显著频率成分。V2 采用 `dl=0.0225 m`，100 MHz、最大 εr=11.2、3 倍中心频率时仍满足 λ/10。
4. **2-D 模式**：按官方 cylinder 例子，模型在 z 方向只有一个 cell，使用 z 极化 Hertzian dipole；z 两侧 PML 关闭。
5. **PML 顺序**：`#pml_cells` 的六项顺序固定为 `x0 y0 z0 xmax ymax zmax`，且 PML 位于 domain 内部。V2 使用 `20 20 0 20 20 0`，目标、源和接收点之外另保留 10-cell guard。
6. **对象覆盖顺序**：对象命令按出现顺序覆盖既有材料。V2 先填充 bedrock，再覆盖 weathered 和 cover，并对零厚度、越界和材料空洞做静态校验。
7. **扫描步进**：`#src_steps`/`#rx_steps` 仅用于简单源和接收器。V2 控制场景使用理想 Hertzian dipole，不把它用于带几何的商业天线模型。
8. **几何检查**：每个 case 生成单独 `geometry_check_*.in`，支持 `python -m gprMax ... --geometry-only`。
9. **时间轴**：`#time_window: 700e-9` 不表示 501 个 solver step。gprMax 按 CFL 自动生成大量迭代；只有在仿真后才根据 HDF5 根属性 `dt` 重采样到 501 点、1.4 ns。
10. **输出读取**：接收数据从 HDF5 `/rxs/rx1/Ez` 读取，版本、iterations、`dx_dy_dz`、`dt`、source/receiver steps 均写入后处理审计结果。
11. **商业天线限制**：官方 GSSI 400 MHz 天线库要求 0.5/1/2 mm 立方网格，且移动步长必须是网格整数倍。本轮 100 MHz 物理控制不混用该 400 MHz 几何模型；先用理想线源验证几何和标签链。

## 本阶段明确未解决的假设

- `tx_rx_offset_m=0.18` 是可配置的暂定值，必须在获得实际 UAV-GPR 天线几何后校正。
- 当前材料为非色散控制材料，不是最终现场反演结果。
- 当前控制场景均为平地表；地形随飞行高度联动将在 control 通过后进入 pilot。
- 电线、树木、水体、测线边缘和随机外部杂波按用户要求暂缓。
