# PROJECT_PAGE_AUDIT - v0.7.6 项目管理页拆分记录

## 本轮目标

本轮只拆分产品化易用界面中的“项目管理”页 UI 构建逻辑，避免 `easy_window.py` 继续承担页面表单构造细节。

## 已完成

- 新增 `src/uavgpr_simlab/gui/pages/project_page.py`。
- 新增 `ProjectPageWidgets`，集中返回主窗口后续需要访问的控件引用。
- 新增 `build_project_page()`，负责构建：
  - 工作目录输入框；
  - 仿真计划输入框；
  - 模型数量 spin box；
  - 工作目录/计划选择按钮；
  - 计划预览按钮；
  - 计划 JSON/YAML 预览文本框。
- `easy_window.py` 中的 `_build_project_page()` 改为调用页面构建器，只保留回调绑定、跨页面状态和服务调用。

## 未改变

- 未改变 `services/project_service.py` 的计划解析和模型生成语义。
- 未改变 manifest 结构。
- 未改变模型生成数量、工作目录、计划路径的默认值。
- 未改变 gprMax 调用语义。
- 未改变任务 fingerprint、history marker 或 B-scan 后处理。

## 架构边界

```text
gui/pages/project_page.py
  只负责项目管理页控件构建

services/project_service.py
  负责计划预览和模型生成

gui/easy_window.py
  负责跨页面状态、按钮回调、当前 manifest 同步和消息提示
```

## 验证方式

本轮建议至少执行：

```bash
python -m compileall -q src scripts
PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/self_test.py
PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/make_v070_product_screenshots.py
```

并离屏确认：

- `EasyMainWindow.stack.count() == 6`
- `workspace_edit` 默认指向 `workspace/easy_project`
- `plan_edit` 默认指向 `configs/run_plan_3060_quick.yaml`
- `case_count_spin.value() == 20`
- `project_plan_text` 为只读

## 后续建议

1. 下一步优先拆 `gui/pages/batch_page.py`，因为批量页控件最多，并且已经有 `services/easy_batch_service.py` 可以承接业务协调。
2. 然后拆 `gui/pages/history_page.py`，与 `services/easy_history_service.py` 对齐。
3. 页面层稳定前，不要重写 gprMax 调用链路、任务去重和历史 marker 语义。
