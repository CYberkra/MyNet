# EASY_WIDGETS_AUDIT - UavGPR-SimLab v0.7.3

## 本轮目标

本轮继续收敛 `src/uavgpr_simlab/gui/easy_window.py` 的职责，但不拆页面类、不改变仿真核心、不改变任务 registry、history marker、B-scan 后处理或 gprMax 调用语义。

重点是把易用界面中重复使用的卡片、流程步骤和历史记录卡片渲染逻辑拆到独立 GUI 小控件辅助模块，避免主窗口继续膨胀。

## 新增模块

```text
src/uavgpr_simlab/gui/easy_cards.py
```

该模块负责：

- 页面标题块：`page_header()`；
- 首页/批量页指标卡：`metric_card()`；
- 指标卡数值刷新：`set_metric_value()`；
- 流程步骤卡片：`flow_step()`；
- 模型信息行：`model_info_row()`；
- 历史页顶部模式标签：`mode_tab()`；
- 帮助页步骤标签：`help_step()`；
- 历史列表记录卡：`history_record_card()`。

## 修改边界

`easy_window.py` 现在更接近“页面组合 + 事件响应”：

```text
easy_window.py
  ↓ 调用
gui/easy_cards.py       纯 GUI 小控件构造
gui/easy_ui.py          样式、状态标签、缩略图和表格辅助
services/easy_*.py      批量、历史、首页统计等业务协调
core/                   模型生成、任务注册、历史、后处理等核心逻辑
```

本轮没有把页面整体拆成 `gui/pages/`，是为了避免一次性迁移过多状态变量、信号连接和 Qt 对象生命周期。

## 行数变化

本轮前：

```text
easy_window.py 约 762 行
```

本轮后：

```text
easy_window.py 约 732 行
easy_cards.py 约 151 行
```

虽然行数减少不大，但重复 UI 渲染逻辑已经移出，后续拆页面会更安全。

## 已验证

```bash
python -m compileall -q src scripts
PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/self_test.py
PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/make_v070_product_screenshots.py
```

自测结果：`ok: true`。

离屏 Easy GUI 验证：

```text
version: 0.7.3
display: v0.7
pages: 6
```

## 后续建议

1. 下一步优先拆 `services/project_service.py` 或 `services/environment_service.py`，把项目计划预览、环境保存和环境检查协调逻辑从 GUI 移出。
2. 再考虑拆 `gui/pages/batch_page.py` 与 `gui/pages/history_page.py`。
3. 页面拆分前不要改 gprMax 运行语义，不要改变 marker/fingerprint。 
