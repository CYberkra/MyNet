# ADVANCED_WORKERS_AUDIT - v0.7.20

## 本轮目标

将高级工程界面中的后台 worker 类从 `main_window.py` 拆出，降低主窗口文件对队列运行细节的承载压力。

本轮只迁移类定义，不改变正式 gprMax 调用语义。

## 新增模块

```text
src/uavgpr_simlab/gui/advanced_workers.py
```

该模块当前包含：

```text
GenericWorker      通用后台任务 worker
LiveQueueWorker    高级队列实时运行 worker
```

## 迁移边界

### 已迁移

- `GenericWorker` 的通用后台执行与异常信号。
- `LiveQueueWorker` 的队列运行线程类定义。
- `LiveQueueWorker` 内部的：
  - gprMax 命令执行；
  - stdout 日志转发；
  - 实时 B-scan preview 信号；
  - 进度信号；
  - cancel / terminate；
  - job fingerprint；
  - running / done / failed marker 写入；
  - geometry-only 判断；
  - 完成后的 B-scan 导出。

### 仍保留在 `main_window.py`

- worker 创建时机；
- worker 信号连接；
- 按钮状态恢复；
- 队列表格状态刷新；
- 进度条刷新；
- 实时 B-scan 画布刷新；
- 弹窗和页面状态协调。

## 不变项

本轮不改变：

```text
gprMax 命令构造
conda run 语义
GPU / OpenMP 参数传递
task fingerprint
history marker
running / done / failed 状态
B-scan 后处理
geometry-only 语义
skip completed / force rerun 语义
```

## 验证

本轮要求通过：

```bash
python -m compileall -q src scripts
PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/self_test.py
PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/make_v070_product_screenshots.py
PYTHONPATH=src python scripts/smoke_gprmax_source.py --gprmax-root /mnt/data/_gprmax_v317_read/gprMax-v.3.1.7 --work-dir workspace/gprmax_source_smoke_v0720 --omp-threads 1 --timeout 180
```

自测应确认：

```text
advanced_worker_class: GenericWorker
advanced_live_worker_class: LiveQueueWorker
advanced tabs: 10
easy pages: 6
```

## 风险

### P0

暂无。

### P1

该拆分不能替代 Windows + conda + CUDA + pycuda + GPU 目标机验证。

### P2

`LiveQueueWorker` 仍包含正式运行生命周期、marker 写入和后处理。后续若继续服务化，必须先增加更细粒度的回归测试，尤其是：

- skip completed；
- force rerun；
- geometry-only；
- failed marker；
- running marker 清理；
- B-scan 自动导出。

### P3

当前 `advanced_workers.py` 属于 GUI worker 层，而不是纯服务层；它依赖 PySide6 信号，这是合理边界。

## 后续建议

1. 继续审计高级工程界面剩余页签，优先拆相对独立的真实 CSV / QC / 训练页签。
2. 如需进一步治理队列运行生命周期，应先增加针对 marker 和失败任务的测试样例。
3. 暂不改变正式 gprMax 命令、fingerprint、marker 或 B-scan 后处理。
