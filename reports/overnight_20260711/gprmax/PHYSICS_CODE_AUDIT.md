# gprMax V2 物理与代码审计

## 结论

- 修复后静态 P0：**0**；静态 P1：**0**。
- 四个控制场景：**4/4 静态通过**。
- gprMax 实际求解：**环境阻断，未运行**。
- 正式训练放行：**否**。

## 本轮实质修复

1. 将求解请求时窗与训练时轴拆开：solver 701 ns；canonical 0–700 ns、501 点、1.4 ns。HDF5 必须证明最后存储样本覆盖 700 ns，且 `Iterations*dt` 覆盖 701 ns。
2. PML 外额外安全距离从 10 提升到 20 cell，并同步域边界、manifest、合同和测试。
3. `Iterations` 必须与接收器数据时间样本数一致；拒绝短输出和伪完整输出。
4. 平层到时定义为精确水平分层双基地参考；曲面到时改称 columnar layered 搜索参考，不再宣称精确 specular ray truth。
5. CTRL02 正样本与 CTRL04 matched negative 建立双向 provenance，并逐数组验证相同上覆几何。
6. 控制生成器记录 commit、dirty 状态、seed、case spec hash、几何/采集/FDTD 域的不同尺度。
7. gprMax 安装脚本改为官方 release 仓库 `conda_env.yml` 的隔离环境流程。

## 物理链核查

- 256 道、0.09 m，道中心跨度 22.95 m；没有 resize/padding 冒充物理重采样。
- Tx/Rx offset 0.18 m，为 8 个网格；扫描步长为 4 个网格。该 offset 仍是待硬件确认的控制假设。
- 空气、覆盖层、风化层分层到时分别积分；负样本不生成目标曲线。
- full/no-basal/air 三者语义明确：差分是匹配对照响应，不被称为独立纯 target-only 场。
- 几何 box 有序覆盖，不存在零厚度或静态检测到的材料空洞。
- 新 V2 不使用 Line9 曲线、时间或几何统计；所有出现的 `Line9` 文本仅是禁止字段与泄漏检测。

## 尚未验证

- 实际 geometry view；
- CFL dt 与真实 HDF5；
- PML 反射、振铃、相位极性；
- 14 m 中等损耗目标可见性；
- CPU/GPU 资源成本；
- 真实 UAV-GPR 天线结构和 Tx/Rx offset。
