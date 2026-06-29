# EASY_SERVICE_LAYER_AUDIT - v0.7.2

## 本轮目标

本轮只做小范围结构治理，不重写页面，不改变 gprMax、任务 registry、历史 marker、B-scan 后处理或模型生成语义。

目标是把 `easy_window.py` 中一部分“业务协调逻辑”抽到 Qt 无关的服务层，让主窗口继续向“只负责显示和触发”的方向收敛。

## 新增服务层

```text
src/uavgpr_simlab/services/__init__.py
src/uavgpr_simlab/services/easy_batch_service.py
src/uavgpr_simlab/services/easy_history_service.py
```

### easy_batch_service.py

职责：

- 解析批量仿真 variant 文本；
- 读取 manifest；
- 生成唯一 case 行；
- 从 manifest 推断 workspace；
- 构建批量预检任务计划；
- 持久化 job plan；
- 为批量表格准备模型缩略图路径；
- 为运行队列构造待运行任务。

该模块仍复用既有核心能力：

```text
core.job_registry
core.runner
core.visual_history
```

未改变任务 fingerprint、done marker、skip completed 或任务构造规则。

### easy_history_service.py

职责：

- 首页项目状态统计；
- 历史记录扫描和筛选；
- 历史记录 marker 读取；
- 历史预览元数据构建；
- 选中历史记录时加载模型标签和 B-scan；
- 导出历史 CSV；
- 删除历史记录。

该模块仍复用既有核心能力：

```text
core.history
core.visual_history
```

未改变历史记录目录、marker 格式、导出 CSV 或删除安全规则。

## easy_window.py 变化

`easy_window.py` 不再直接负责以下逻辑：

- CSV manifest 的底层读取；
- 批量预检记录构造；
- job plan 写入；
- 待运行 task 构造；
- 首页历史状态统计；
- 历史记录扫描、marker 读取和 B-scan 加载；
- 历史导出与删除的 core API 调用。

它仍负责：

- 创建页面和控件；
- 响应按钮；
- 把服务层返回的数据渲染成表格、卡片、画布和弹窗；
- 维护当前 manifest、当前 worker 和页面选择状态。

## 当前边界

本轮没有拆页面类，原因是：

1. 页面间共享状态仍在 `EasyMainWindow` 中，例如 `current_manifest`、`workspace_edit`、`batch_manifest_edit`、`live_worker`；
2. 直接拆 `gui/pages/` 会引入较多信号同步和状态传递风险；
3. 先抽 Qt 无关服务层，可以降低后续页面拆分风险。

## 后续建议

下一步建议继续低风险拆分 GUI 小控件，而不是立刻拆整页：

1. 新增 `gui/easy_cards.py` 或 `gui/widgets/cards.py`，迁移 `card()`、`flow_step()`、历史记录卡片和批量表格行渲染。
2. 再拆 `gui/pages/batch_page.py` 和 `gui/pages/history_page.py`，此时页面只需要调用已稳定的服务层。
3. 真实环境验证前，不建议继续修改 gprMax 调用语义。

## 风险

- P0：本轮未发现。
- P1：真实 Windows + gprMax + CUDA 链路仍未在目标机器验证。
- P2：`easy_window.py` 仍偏大，但职责已经从“UI + 部分业务协调”进一步收敛。
- P3：部分 UI 构建函数仍存在内联样式，后续可继续迁移到 widgets/样式模块。
