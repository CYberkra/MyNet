# 仿真逼真度十轮研究协议

日期：2026-07-16
冻结基线：`a5f91a1`
物理母模型：`FORMAL06C_SUBTLE_INTERFACE_DEVELOPMENT`

## 目标

在不复制实测像素、不移动标签、不把 Line9 测试信息带入正式参数选择的前提下，寻找一套同时满足以下条件的仿真方案：

1. 基覆界面保持连续、多周期、可见，且严格 full/no-basal 因果对可成立；
2. 背景纹理、局部连续性、振幅变化和频率响应落入多条实测测线的范围，而不是只像 Line9；
3. 每个新增因素均有独立物理或采集含义，可消融、可复现、可用于论文论证；
4. 正式参数只使用 Line3、Line7、LineL1 拟合，Line6 验证；Line9 在候选冻结后才开放诊断，LineX1 不参与选择；
5. 十轮后若没有候选通过视觉和数值双门禁，继续开展定向轮次，不以“做满十次”代替效果达标。

## 统一研究顺序

每轮必须依次完成：

```text
文献/源码证据 -> 单因素假设 -> 静态或低成本试验 -> 盲图
-> 折内数值指标 -> Line6 选择 -> 冻结 -> Line9 诊断 -> 晋级/淘汰
```

物理因素遵循：

```text
geometry-only -> one trace -> native consecutive short B-scan
-> full/no-basal pair -> wider release run
```

后处理/采集因素必须把同一个随机实现同时用于 full/control，并保留 canonical gprMax 输出不变。

## 十轮问题定义

| 轮次 | 唯一主因素 | 核心问题 | 预计求解成本 |
|---|---|---|---:|
| 1 | 有效系统响应 | 点源子波与真实 SFCW/成像链频响差异能解释多少波包差距？ | 无新 FDTD |
| 2 | 平滑增益和时延漂移 | 逐道增益、零时和采样漂移能否产生实测的局部不稳定性而不折断界面？ | 无新 FDTD |
| 3 | 残余共模/处理链 | 不完全背景抑制与低秩相干残留是否是实测背景的重要组成？ | 无新 FDTD |
| 4 | 有色异方差噪声 | 深度相关、横向相关且非高斯的噪声是否优于随机相位高斯场？ | 无新 FDTD |
| 5 | 飞高和姿态 | 逐道飞高、收发几何与姿态变化能解释多少振幅和时间起伏？ | 小型 FDTD + 近似 |
| 6 | 风化带/界面粗糙度 | 多尺度、有限相关长度的粗糙过渡能否增加自然中断而不形成层栈？ | 新 FDTD |
| 7 | 随机体介质 | 各向异性多尺度体散射能否产生实测纹理且保留目标可见性？ | 新 FDTD |
| 8 | 有限相带/透镜 | 稀疏低对比有限地质体能否补充局部事件，而不变成双曲线拼接？ | 新 FDTD |
| 9 | 天线与维度偏差 | 点源、二维近似、极化和有限孔径造成的差距有多大？ | 小型 3D/代理试验 |
| 10 | 正交组合 | 哪组已通过因素以最少复杂度覆盖实测分布并保持因果标签？ | 组合验证 |

## 视觉门禁

固定输出以下视图，任何一项缺失都不能晋级：

1. 原始 signed B-scan；
2. 同一处理合同下的 common-mode suppression + time-power gain；
3. 无标签目标走廊裁剪；
4. 与 Line3、Line7、LineL1、Line6 的等物理宽度盲对照；
5. 候选冻结后的 Line9 诊断；
6. full/control 差分视图（只用于因果归因，不作为“目标可见”的替代品）。

淘汰特征包括：规则长条纹、周期梳状纹、拼接双曲线、孤立方块、整段硬 dropout、标签走廊定向增强、明显过曝目标、只在差分图可见。

## 数值门禁

每轮至少报告：

- target/background RMS；
- target envelope CV 和 dropout；
- 对齐波包相关系数、显著 signed lobe 数、峰频和谱质心；
- 折内与 Line6 的目标谱包络误差；
- 背景横向相关长度、局部谱质心变化、低秩能量和有限事件统计；
- 标签中心漂移；
- 候选复杂度和计算成本。

数值相似不覆盖视觉否决。最终选择采用 Pareto 门禁，不把所有差异压成一个不可解释的总分。

## 论文创新候选

当前优先探索的论文主张是：

> **Causal-pair-preserving, acquisition-conditioned physics-to-measurement simulator**：
> 将可审计 gprMax 地质场、严格 full/control 因果对和折内拟合的采集算子分开建模，训练时同时保留 clean physics 与 degraded measurement 两个视图。

该方向比全图 GAN/CycleGAN 更适合本项目，因为它不允许生成器移动基覆标签，并可对地质、天线、采集和处理因素分别消融。

## 文献与官方能力依据

- gprMax 官方文档：多极 Debye/Lorentz/Drude、分形异质体、粗糙表面、天线模型与 dielectric smoothing。
- Warren et al. (2016), gprMax open-source FDTD framework, DOI `10.1016/j.cpc.2016.08.020`。
- Koyan and Tronicke (2020), realistic multi-scale 3D sedimentary model, DOI `10.1016/j.cageo.2020.104422`。
- Stephan, Allroggen, and Tronicke (2024), convolution-based realistic GPR noise, DOI `10.1002/nsg.12273`。
- Majchrowska et al. (2021), arbitrary complex dielectric properties in gprMax, arXiv `2109.01928`。
- Lambot et al. (2006), off-ground GPR surface roughness and antenna transfer-function calibration, DOI `10.1029/2005WR004416`。
- Warren et al. (2022), realistic FDTD antenna surrogates, DOI `10.1109/TAP.2022.3142335`。

## 产物规则

- 研究脚本、参数、报告、盲图和小型可复现实证纳入 Git；
- 大型 `.out` 与临时 VTI 留在本地 solver cache；VTI 哈希后删除；
- 每轮在 `research_ledger.json` 中登记假设、输入哈希、选择折、结果和决策；
- 未通过轮次保留为负消融，禁止静默覆盖。
