# 营山无人机 GPR 原始 CSV 数据格式

## 权威列定义

源数据文件前 4 行依次记录：

1. `Number of Samples`：每道 A-Scan 采样点数；
2. `Time windows (ns)`：观测时窗，单位 ns；
3. `Number of Traces`：测线总道数；
4. `Trace interval (m)`：声明道间距，单位 m。

第 5 行开始按 trace-major 顺序存储数据。每一道连续占用 `Number of Samples` 行，五列含义固定为：

| 列 | 字段 | 单位 |
|---|---|---|
| 1 | 经度 `longitude` | degree |
| 2 | 纬度 `latitude` | degree |
| 3 | 地表高程 `ground_elevation_m` | m |
| 4 | 雷达反射波振幅 `radar_reflection_amplitude` | 原始幅值 |
| 5 | 飞行高度 `flight_height_agl_m` | m |

同一道内部的经度、纬度、地表高程和飞行高度应保持不变；第 4 列随时间采样点变化。

## Canonical 数据约定

`scripts/import_yingshan_raw_csv.py` 直接从原始 ZIP 生成 `lines/*.npz`：

- 原始波形来自第 4 列；
- `raw_full_normalized` 使用整条测线绝对幅值 P99 归一化；
- 天线绝对高程定义为 `ground_elevation_m + flight_height_agl_m`；
- 水平轴同时保留声明道间距和 GNSS 累计距离；
- 标签、状态和权重暂时来自已审计的重叠窗口缓存；
- 原始 ZIP、CSV member 和所有 SHA256 均写入数据合同。

窗口反拼接脚本仅保留为灾难恢复工具，默认禁止覆盖原始 CSV canonical 全测线。
