# EASY_WINDOW_REFACTOR_AUDIT - UavGPR-SimLab v0.7

## 本轮审计目标

本轮只处理 `src/uavgpr_simlab/gui/easy_window.py` 的可维护性问题，不改变底层仿真、任务 registry、gprMax 调用、历史结果扫描和 B-scan 后处理语义。

## 当前结论

`easy_window.py` 原本同时承担以下职责：

1. 产品化主窗口和导航；
2. 首页、模型预览、批量仿真、历史结果、项目管理、设置帮助六个页面构建；
3. 状态文案、状态标签、缩略图、标题卡片、统一样式；
4. manifest 加载、任务预检、历史扫描、环境检查等 GUI 到 core 的协调逻辑。

这说明 v0.7 的用户流程已经清晰，但主窗口文件仍有继续膨胀风险。后续新增功能不应继续直接堆在 `EasyMainWindow` 中。

## 已完成的低风险拆分

新增：

```text
src/uavgpr_simlab/gui/easy_ui.py
```

该模块承接：

- `EASY_STYLE`：产品化易用界面样式表；
- 状态文案与状态图标；
- 状态 chip；
- 表格单元格写入辅助；
- 缩略图 label；
- 通用 section title；
- 仿真 variant 友好名称。

这样可以把“通用 UI 小部件和样式”从“主窗口页面组织和业务协调”中分离出来，属于低风险结构整理。

## 暂未拆分的内容

暂未拆分六个页面类，原因是：

- 当前 `EasyMainWindow` 内部字段共享较多，例如 manifest、workspace、canvas、表格和设置控件；
- 一次性拆成多个页面类容易引入信号连接、状态同步和生命周期问题；
- 当前阶段优先保证 v0.7 产品化主线稳定。

## 推荐后续拆分顺序

### 第一阶段：继续拆通用 UI 与小控件

候选模块：

```text
src/uavgpr_simlab/gui/widgets/cards.py
src/uavgpr_simlab/gui/widgets/path_inputs.py
src/uavgpr_simlab/gui/widgets/status_widgets.py
```

适合迁移：

- `card()`；
- `flow_step()`；
- 文件/目录选择行；
- 历史记录卡片；
- 批量任务表格行渲染。

### 第二阶段：拆服务层

候选模块：

```text
src/uavgpr_simlab/services/easy_project_service.py
src/uavgpr_simlab/services/easy_batch_service.py
src/uavgpr_simlab/services/easy_history_service.py
```

适合迁移：

- manifest 行读取与唯一 case 提取；
- 首页 summary counts；
- batch plan 生成和 counts 汇总；
- history record 转 preview 数据。

### 第三阶段：拆页面类

候选模块：

```text
src/uavgpr_simlab/gui/pages/home_page.py
src/uavgpr_simlab/gui/pages/model_preview_page.py
src/uavgpr_simlab/gui/pages/batch_page.py
src/uavgpr_simlab/gui/pages/history_page.py
src/uavgpr_simlab/gui/pages/project_page.py
src/uavgpr_simlab/gui/pages/settings_page.py
```

建议等服务层和通用 widgets 稳定后再做，避免页面类之间直接互相访问导致更复杂的耦合。

## 风险分级

- P0：本轮未发现启动级风险。
- P1：本轮未改变真实 gprMax 运行链路，真实 Windows + CUDA + gprMax 仍需目标机验证。
- P2：`easy_window.py` 仍是 700+ 行主窗口，已缓解但未根治。
- P3：部分内联样式仍存在，后续可逐步迁移。

## 验证要求

每次继续拆分后至少执行：

```bash
python -m compileall -q src scripts
PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/self_test.py
PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/make_v070_product_screenshots.py
```
