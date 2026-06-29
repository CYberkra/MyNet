# REAL_CSV_SERVICE_AUDIT - v0.7.22

## 本轮目标

将高级工程界面“实测/弱监督”页签中的真实 UAV-GPR CSV 预览与 QC 导出协调逻辑，从 `gui/main_window.py` 下沉到服务层：

```text
src/uavgpr_simlab/services/real_csv_service.py
```

本轮只迁移协调逻辑，不改变 `core.real_data` 中 CSV 解析、背景扣除、robust normalize、指数增益、SNR 或 PNG/NPZ 导出的算法语义。

## 新增服务

### `load_real_csv_preview(path, max_traces)`

职责：

1. 调用 `read_uavgpr_csv()` 读取真实 CSV；
2. 调用 `subtract_mean_background()` 做均值背景扣除；
3. 调用 `robust_normalize()` 生成 GUI 预览数据；
4. 返回 `RealCsvPreview`，其中包含：
   - `normalized_bscan`
   - `raw_bscan`
   - `info`
   - `time_window_ns`

### `export_real_csv_qc(path, workspace, max_traces, make_baselines=True)`

职责：

1. 按高级界面原有约定构造输出目录：

```text
<workspace>/real_csv_qc/<csv_stem>/
```

2. 调用 `convert_real_csv()` 生成 NPZ、PNG 和 `qc_report.json`。

## `main_window.py` 保留职责

`main_window.py` 仍负责：

- 用户点击回调；
- 错误弹窗；
- Matplotlib 画布刷新；
- `last_preview` 状态保存；
- 后台 worker 调度；
- f-k 图显示触发。

## 未改变事项

- 未改变真实 CSV 文件格式解析；
- 未改变均值背景扣除；
- 未改变 robust normalize；
- 未改变 f-k 预览输入数据；
- 未改变 QC 导出目录结构；
- 未改变 NPZ / PNG / JSON 输出内容；
- 未改变默认 `max_traces`；
- 未改动正式 gprMax 任务运行语义。

## 验证

新增/更新自测覆盖：

```text
load_real_csv_preview(sample_csv, max_traces=16)
export_real_csv_qc(sample_csv, tmp, max_traces=16)
```

断言内容：

```text
preview.normalized_bscan.shape == [501, 16]
preview.info["shape_samples_x_traces"] == [501, 16]
real_uavgpr_bscan_preview.npz 存在
qc_report.json 由 core.real_data.convert_real_csv 生成
```

## 风险

- P0：暂无。
- P1：暂无，未改变 CSV 算法语义。
- P2：真实 CSV 相关的 f-k 绘图仍由画布组件完成，当前边界合理。
- P3：后续可以继续补充更多异常 CSV 文件的错误提示映射。
