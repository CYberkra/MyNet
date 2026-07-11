# gprMax 官方手册一致性审计（2026-07-11）

## 结论

四个控制场景的 gprMax 输入语法、2D TMz 设置、网格、PML 顺序、源/接收器移动、几何覆盖、匹配反事实关系和理论传播到时均通过静态审计。

原上传 ZIP 存在一个确定的打包完整性问题：文本文件由 LF 变成 CRLF，但 `FILE_SHA256.csv` 和 `control_index.json` 仍保存 LF 版本的哈希，因此官方项目验证器会报告 34 个 hash mismatch。本修正版已统一回 LF，四个场景的哈希和项目静态验证均恢复 PASS。

## 官方规则核对

- 输入单位：空间为 m、时间为 s、频率为 Hz。
- 2D 模型：一个维度必须等于该方向空间步长；本包 z=dz=0.0225 m。
- 2D TMz：使用 z 极化 Hertzian dipole 和 Ez 接收。
- PML 顺序：x0, y0, z0, xmax, ymax, zmax；本包为 20,20,0,20,20,0。
- PML 位于域内；首个 Tx、末个 Rx 和天线顶边均额外保留 20 个网格，超过官方至少 15 个网格建议。
- `#src_steps`/`#rx_steps` 用于简单源和接收器的多次运行移动；0.09 m=4 个网格，`-n 256 --geometry-fixed` 合法。
- 所有坐标、层界面、Tx/Rx 偏移和模型尺寸均为 0.0225 m 的整数倍。
- 100 MHz Ricker 按 3 倍中心频率作为显著高频上限时，最大介电常数 11.2 中仍约有 13.3 cells/wavelength，满足 lambda/10 下限。

## 配对关系

- CTRL01/02/03：`full_scene` 与 `no_basal_contrast_control` 除基覆界面下方材料外完全一致；`full-control` 仅产生 bedrock→weathered 的反事实差异。
- CTRL04：与 CTRL02 的 no-basal 场景在材料、几何、飞高、Tx/Rx 和采样上匹配，是基覆目标的 confirmed-negative 控制候选。
- `air_reference` 不包含任何地下物体；仅用于源波形、空气直达耦合和早期响应诊断，不能视为严格的 A 分量 Maxwell 真值。

## 需要保留的限制

1. 这些是 2D 线源控制场景。它们适合验证坐标、到时、配对差分和处理链，但不能单独证明 14 m 目标在真实 3D UAV-GPR 中可见。
2. `reference_arrival_time_ns` 是传播时间，不包含 gprMax Ricker 的固有峰值延迟。100 MHz 时延迟为 14.1421356 ns；真实 `visible_phase_time_ns` 必须由求解后的 full-control 差分提取。
3. 目前仅静态审计通过；必须依次完成 `--geometry-only`、CTRL01 小规模 FDTD、256 道完整运行、HDF5 合并和 postprocess 后，才能称为运行验收通过。
4. 所有 control 继续保持 `formal_training_allowed=false`。

## 官方来源

- https://gprmax.readthedocs.io/en/latest/input.html
- https://gprmax.readthedocs.io/en/latest/gprmodelling.html
- https://github.com/gprMax/gprMax/releases/tag/v.3.1.7
- https://raw.githubusercontent.com/gprMax/gprMax/master/gprMax/waveforms.py
