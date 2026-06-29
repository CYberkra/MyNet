# ADVANCED_QUEUE_SERVICE_AUDIT - v0.7.19

## 本轮目标

把高级工程界面“批量运行”页签中的队列准备逻辑从 `main_window.py` 下沉到无 Qt 依赖的服务层，继续降低高级主窗口的职责密度。

## 新增模块

```text
src/uavgpr_simlab/services/advanced_queue_service.py
```

## 已迁移职责

- 读取 `manifest.csv` 并生成队列列表显示行；
- 限制队列页签最多显示 1500 条 manifest 记录，并返回截断信息；
- 生成 geometry-only / full BAT 文件；
- 将选中 manifest 行转换为 `GprMaxTask`；
- 从 manifest 构造批量 `GprMaxTask`；
- 生成队列任务摘要。

## 仍保留在 `main_window.py` 的职责

以下内容涉及运行生命周期和实时 UI 状态，本轮不迁移：

- `LiveQueueWorker` 启动、取消和信号连接；
- 实时 B-scan 预览刷新；
- 进度条刷新；
- `job_fingerprint`、`job_id` 和 marker 写入；
- gprMax 命令实际执行；
- `.out` 后处理和历史 marker 更新。

## 架构边界

`advanced_queue_service.py` 不导入 PySide6，不直接操作控件，不启动外部进程，不写 marker，不改变 gprMax 命令语义。它只做队列准备与文件生成协调。

## 验证

- `python -m compileall -q src scripts`
- `PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/self_test.py`
- `PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/make_v070_product_screenshots.py`
- `PYTHONPATH=src python scripts/smoke_gprmax_source.py --gprmax-root /mnt/data/_gprmax_v317_read/gprMax-v.3.1.7 --work-dir workspace/gprmax_source_smoke_v0719 --omp-threads 1 --timeout 180`

## 风险

- P0：暂无。
- P1：不涉及真实 Windows/CUDA/GPU 目标机验证。
- P2：`LiveQueueWorker` 仍在 `main_window.py` 中，后续可考虑拆到 `gui/advanced_workers.py` 或运行服务，但必须保持运行语义不变。
- P3：队列页签显示文本仍为工程化表述，适合高级入口。

## 下一步建议

1. 抽 `gui/advanced_workers.py`，迁移 `GenericWorker` 和 `LiveQueueWorker`。
2. 再考虑抽高级模型预览页签 UI。
3. 继续避免改动正式 gprMax 调用、fingerprint、marker 和 B-scan 后处理。
