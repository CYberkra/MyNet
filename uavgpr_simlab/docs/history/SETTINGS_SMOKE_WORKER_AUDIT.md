# SETTINGS_SMOKE_WORKER_AUDIT - v0.7.14

## 目标

本轮将设置页“最小 CPU 测试”从同步执行改为后台执行。目标是让用户在运行 gprMax 源码最小 CPU smoke test 时，GUI 主线程不再被 `subprocess` 求解过程阻塞。

## 修改范围

- 新增 `src/uavgpr_simlab/gui/easy_workers.py`。
- 新增 `GprMaxSourceSmokeWorker`，通过 Qt signal 返回 smoke test 报告或异常文本。
- `easy_window.py` 负责：
  - 创建 `QThread`；
  - 创建 worker；
  - 禁用按钮并显示“测试中...”；
  - 接收完成 / 失败信号；
  - 恢复按钮状态并清理线程引用。

## 架构边界

后台 worker 只调用：

```text
services/gprmax_smoke_service.py
```

它不直接操作：

```text
core/runner.py
core/job_registry.py
core/history.py
core/postprocess.py
```

因此不会改变正式批量仿真、任务 fingerprint、history marker 或 B-scan 后处理语义。

## 验证

本轮验证项：

```bash
python -m compileall -q src scripts
PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/self_test.py
PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/make_v070_product_screenshots.py
PYTHONPATH=src python scripts/smoke_gprmax_source.py --gprmax-root /mnt/data/_gprmax_v317_read/gprMax-v.3.1.7 --work-dir workspace/gprmax_source_smoke_v0714 --omp-threads 1 --timeout 180
```

另做 Easy GUI 离屏 worker 烟测：设置 gprMax 源码目录后触发按钮回调，等待后台线程结束，确认日志中出现“总体状态：通过”，按钮文本恢复为“最小 CPU 测试”。

## 风险

- P0：暂无。
- P1：仍不能替代 Windows + CUDA + pycuda + GPU 目标机验证。
- P2：当前只有最小 CPU smoke test 使用该 worker；如果后续把更长的真实批量任务接入普通界面，应设计统一任务队列/取消机制。
- P3：按钮目前只支持等待完成，不支持用户主动取消；极小测试通常足够短，后续可视目标机反馈再补取消按钮。
