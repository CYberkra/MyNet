# gprMax 官方证据台账

访问日期：2026-07-11。只使用官方 gprMax 文档与官方仓库。

| 官方来源 | URL | 规则（转述） | 本项目落实 |
|---|---|---|---|
| Input commands | https://gprmax.readthedocs.io/en/latest/input.html | 输入采用 SI；坐标以模型左下角为原点；空间/时间值会映射到离散单元或迭代。 | 0.09 m 步长和 0.18 m Tx/Rx 偏移分别固定为 4/8 个 0.0225 m cell。 |
| Modelling guidance | https://gprmax.readthedocs.io/en/latest/gprmodelling.html | CFL 由网格控制；最小波长通常至少约 10 个 cell；源、接收器和目标应远离吸收边界。 | 以 3×100 MHz 和最大 εr=11.2 做 λ/10 审核；PML 20 cell + guard 20 cell。 |
| Source/receiver steps | https://gprmax.readthedocs.io/en/latest/input.html#src-steps-and-rx-steps | `#src_steps/#rx_steps` 用于多次模型运行时移动简单源/接收器。 | 仅理想 Hertzian 控制使用 `-n 256 --geometry-fixed`；不把该规则外推到移动的复杂天线几何。 |
| PML commands | https://gprmax.readthedocs.io/en/latest/input.html#pml-commands | 六个 PML 值顺序为 x0,y0,z0,xmax,ymax,zmax。 | `[20,20,0,20,20,0]`；2D 不变 z 方向不设 PML。 |
| Basic 2D examples | https://gprmax.readthedocs.io/en/latest/examples_simple_2D.html | 2D 模型为单 cell 不变方向；z 极化 Hertzian dipole 表现为 line source；Ricker 有效高频可到中心频率约 2–3 倍。 | 采用 2D x-y、z 极化、100 MHz Ricker，并以 300 MHz 审核空间离散。 |
| Geometry view | https://gprmax.readthedocs.io/en/latest/input.html#geometry-view | `--geometry-only` 可建立模型并输出 geometry view，而不运行 FDTD。 | 作为 CTRL01–04 的下一项硬门禁；当前环境未能执行。 |
| HDF5 output | https://gprmax.readthedocs.io/en/latest/output.html | 根属性包含版本、Iterations、dx/dy/dz、dt、源/接收步长；接收器分量位于 `/rxs/rxN/`。 | 后处理验证 Iterations、真实 dt、网格、步长、覆盖时长，再重采样至 0–700 ns/501 点。 |
| Official repository | https://github.com/gprMax/gprMax | 官方建议用 Miniconda 和仓库 `conda_env.yml` 建隔离环境，再 build/install；`-n`、`--geometry-only`、`--geometry-fixed` 有明确用途。 | 新 setup 脚本从 `v3.1.7` 和官方 `conda_env.yml` 建 `gprmax317` 环境。 |
| Antenna libraries | https://gprmax.readthedocs.io/en/latest/user_libs_antennas.html | 详细天线模型通常绑定特定网格和频率。 | 当前只声称理想点/线源控制链，不将其解释为真实 100 MHz UAV 天线硬件。 |

## 审计结论

静态命令和后处理契约与上述规则一致。官方 geometry view、真实 CFL/HDF5 和 FDTD 响应仍须在具备 gprMax 的运行环境中验证。
