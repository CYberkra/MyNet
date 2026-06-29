# ADVANCED_QUEUE_TAB_AUDIT - 高级工程界面队列页签拆分审计

## 本轮目标

本轮针对 `src/uavgpr_simlab/gui/main_window.py` 中的高级工程界面“5 批量运行”页签做低风险 UI 拆分。

目标是把队列页签的控件构建移动到独立页面构建器：

```text
src/uavgpr_simlab/gui/advanced_pages/queue_tab.py
```

本轮不改变任何正式任务运行语义。

## 拆分内容

新增：

```text
AdvancedQueueTabWidgets
build_advanced_queue_tab()
```

迁移出的 UI 内容包括：

- manifest 输入框；
- 选择 manifest 按钮；
- 加载任务按钮；
- variant 下拉框；
- 批量数量；
- geometry-only 开关；
- 跳过已完成开关；
- 强制重跑开关；
- 写 geometry-only BAT 按钮；
- 写完整 raw BAT 按钮；
- 运行选中任务按钮；
- 运行前 N 个任务按钮；
- 停止按钮；
- 任务列表；
- 进度条；
- 队列日志框；
- 实时 B-scan 预览画布；
- 预览说明标签。

## 保留在 main_window.py 的逻辑

以下内容刻意保留在 `main_window.py`，避免一次性改变运行语义：

- manifest CSV 读取；
- `QListWidgetItem` 数据绑定；
- BAT 命令生成；
- UI 任务转换为 `GprMaxTask`；
- `LiveQueueWorker` 启动与停止；
- 实时 B-scan 预览刷新；
- 进度条刷新；
- done / failed / cancel 处理；
- job fingerprint；
- history marker；
- gprMax 命令构造；
- conda / GPU / OpenMP 参数传递；
- B-scan 后处理。

## 架构边界

`queue_tab.py` 只负责“构建控件 + 回调挂接”。

它不负责：

- 读取或解释 manifest；
- 构造真实任务；
- 决定运行数量；
- 判断任务是否完成；
- 调用 gprMax；
- 写 marker；
- 生成 `.bat` 内容；
- 读取 `.out`；
- 合并或显示 B-scan 数据。

## 验证方式

本轮验证包括：

```bash
python -m compileall -q src scripts
PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/self_test.py
PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/make_v070_product_screenshots.py
PYTHONPATH=src python scripts/smoke_gprmax_source.py --gprmax-root /mnt/data/_gprmax_v317_read/gprMax-v.3.1.7 --work-dir workspace/gprmax_source_smoke_v0717 --omp-threads 1 --timeout 180
```

自测新增高级队列页签断言：

- 高级界面仍有 10 个页签；
- 队列 variant 数量为 5；
- 默认 variant 为 `raw`；
- 默认批量数量为 1；
- geometry-only 默认启用；
- 跳过已完成默认启用；
- 强制重跑默认关闭；
- 队列日志框只读；
- 进度条初始值为 0；
- 实时预览提示正常挂接。

## 风险评估

### P0

暂无。

### P1

未验证 Windows + conda + CUDA + pycuda + GPU 真实求解链路。

### P2

`main_window.py` 仍保留较多高级业务协调逻辑。下一步建议拆 `advanced_widgets/canvases.py` 或 `services/advanced_queue_service.py`，但应继续避免改变运行语义。

### P3

队列页签按钮文案仍保留工程化表达，适合高级入口，暂不需要按普通用户文案重写。

## 后续建议

1. 抽 `gui/advanced_widgets/canvases.py`，把 `MplCanvas` / `Model3DCanvas` 等高级画布整理出去。
2. 抽 `services/advanced_queue_service.py`，承接 manifest 读取、BAT 生成和 UI 任务构造。
3. 真实目标机验证后，再考虑细化队列页的错误提示和恢复策略。
