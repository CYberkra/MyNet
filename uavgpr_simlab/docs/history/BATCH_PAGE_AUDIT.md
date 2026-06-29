# BATCH_PAGE_AUDIT - v0.7.7 批量仿真页拆分记录

## 本轮目标

将产品化易用界面的“批量仿真”页 UI 构建逻辑从 `easy_window.py` 中拆出，形成独立页面构建器，继续降低主窗口膨胀风险。

本轮只处理 UI 构建和控件引用回传，不改变批量预检、任务构造、去重、运行、实时预览或历史记录语义。

## 新增文件

```text
src/uavgpr_simlab/gui/pages/batch_page.py
```

该文件提供：

```text
BatchPageWidgets
build_batch_page(...)
```

## 拆分内容

`batch_page.py` 现在负责构建：

- 批量页标题和说明；
- manifest 清单输入框；
- variant 标签输入框；
- 最大任务数输入；
- 自动跳过已完成开关；
- 选择模型清单、预检、运行、停止按钮；
- 批次统计卡片；
- 批量流程步骤条；
- 批量任务表格；
- 运行中 B-scan 预览画布；
- 批量运行日志区。

`easy_window.py` 仍负责：

- 当前 manifest 状态；
- 调用 `services/easy_batch_service.py` 生成预检计划；
- 表格数据刷新；
- 构建待运行任务；
- 连接 `LiveQueueWorker`；
- 接收实时 B-scan 和日志；
- 刷新历史结果。

## 未改变内容

本轮没有改变：

- gprMax 调用命令；
- conda / GPU / OpenMP 运行配置；
- `build_batch_plan()` 语义；
- `build_pending_tasks()` 语义；
- fingerprint / done marker / running marker；
- B-scan 后处理；
- 历史记录扫描和删除；
- manifest 字段结构。

## 验证方式

推荐验证：

```bash
python -m compileall -q src scripts
PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/self_test.py
PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/make_v070_product_screenshots.py
```

离屏 GUI 烟测应至少确认：

```text
pages == 6
batch_limit_spin.value() == 60
batch_skip_done.isChecked() == True
batch_table.columnCount() == 8
batch_log.isReadOnly() == True
```

## 风险判断

- P0：未发现。
- P1：真实 gprMax / CUDA / Windows 运行链路仍需目标机验证。
- P2：`easy_window.py` 已继续收敛，但模型预览页、历史页和首页构建仍在主窗口内。
- P3：批量页内部 UI 样式仍有部分依赖全局 QSS，后续可统一到更细粒度控件模块。

## 后续建议

1. 下一轮拆 `gui/pages/history_page.py`，与 `services/easy_history_service.py` 对齐。
2. 再拆 `gui/pages/model_preview_page.py`，降低模型预览逻辑对主窗口的占用。
3. 页面层稳定后，补真实 gprMax smoke test 记录模板。
