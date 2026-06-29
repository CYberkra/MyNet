# UavGPR-SimLab v0.8.0-alpha.17 当前架构治理说明

## 处理目标

本轮只处理两个维护性问题：

1. `easy_window.py` 过大，后续继续加入业务逻辑会使主窗口失控。
2. `docs/` 根目录历史审计文档过多，当前开发者很难快速识别正在使用的文档。

## 代码调整

`EasyMainWindow` 现在只保留窗口生命周期、导航、共享状态和少量跨页面辅助函数。页面行为拆到 `src/uavgpr_simlab/gui/controllers/`：

```text
src/uavgpr_simlab/gui/easy_window.py                         203 lines
src/uavgpr_simlab/gui/controllers/home_project_controller.py   88 lines
src/uavgpr_simlab/gui/controllers/model_preview_controller.py 208 lines
src/uavgpr_simlab/gui/controllers/batch_controller.py         325 lines
src/uavgpr_simlab/gui/controllers/history_controller.py       238 lines
src/uavgpr_simlab/gui/controllers/settings_controller.py      202 lines
```

拆分后的边界：

- `easy_window.py`：主窗口初始化、导航、共享 workspace / manifest / env 配置辅助。
- `home_project_controller.py`：首页刷新、项目计划选择和预览。
- `model_preview_controller.py`：模型生成、manifest 加载、2D/3D 预览切换、同步到批量页。
- `batch_controller.py`：批量预检、运行配置、普通批量任务、SceneWorld 全链路任务、GPU/conda 运行参数传递、停止任务。
- `history_controller.py`：历史扫描、dataset/case/variant 树、B-scan 对比、失败定位、重跑 failed、导出和删除。
- `settings_controller.py`：环境保存、诊断、gprMax smoke、ultra tiny 全链路验证、高级工程界面入口。

本轮没有改变页面布局、manifest schema、SceneWorld 五变体数据合同、gprMax 调用语义或 4090 GPU 参数传递。

## 文档调整

`docs/` 根目录只保留当前仍需频繁阅读的文档：

```text
docs/GPRMAX_4090_SETUP.md
docs/GPRMAX_SMOKE_TEST_TEMPLATE.md
docs/REAL_DATA_FORMAT_LINE9.md
docs/UAVGPR_APPLICATION_CONTEXT_FROM_PDF.md
docs/WORKFLOW_FOR_PAPER.md
```

历史审计、阶段性修复说明和旧版 UI 报告已迁移至：

```text
docs/history/
```

`docs/history/README.md` 记录归档说明与文件清单。

## 自动检查

新增：

```text
scripts/check_architecture_guard.py
```

检查内容：

- `easy_window.py` 行数不得超过 350；
- 单个 controller 文件行数不得超过 450；
- `docs/` 根目录不得继续堆放旧审计/报告类 Markdown；
- `docs/history/` 必须存在并带有 README。

当前检查结果：通过。

## 风险与边界

- P0：未发现。
- P1：未改变真实 gprMax / CUDA / 4090 执行链路；目标机 GPU smoke 仍需按 v0.8.0-alpha.16 流程验证。
- P2：主窗口膨胀风险已降低，后续新增页面行为应优先进入 controller / service，不应回填到 `easy_window.py`。
- P3：历史文档已归档，后续每轮新增审计文档若不是当前验收文档，应直接放入 `docs/history/` 或在发布后迁移。
