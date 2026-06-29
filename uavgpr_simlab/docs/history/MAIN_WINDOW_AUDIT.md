# MAIN_WINDOW_AUDIT - 高级工程界面快速审计

## 审计对象

```text
src/uavgpr_simlab/gui/main_window.py
```

当前行数约 1263 行。该文件承载高级工程界面，不是 v0.7 普通用户默认入口，但仍是重要工程工具。

## 当前结论

`main_window.py` 可以继续保留为高级入口，但已经明显偏大。它同时承担：

1. worker 线程；
2. Matplotlib 画布；
3. 3D 模型预览画布；
4. 多个高级页签 UI 构建；
5. 环境检查；
6. 模型生成；
7. 预检与队列运行；
8. 历史扫描、预览、导出、删除；
9. 实测 CSV QC；
10. HPC / 报告 / 训练占位入口。

这不会立即导致 P0/P1，但属于 P2 维护风险。后续不应继续把高级功能直接堆进该主窗口。

## 主要结构观察

### 已识别类

```text
GenericWorker          通用后台线程
LiveQueueWorker        批量队列运行线程，包含较长的 _run_task
MplCanvas              B-scan / FK 画布
Model3DCanvas          3D 模型预览画布
MainWindow             高级工程界面主窗口
```

### 高风险膨胀点

| 区域 | 风险 | 建议 |
|---|---|---|
| `LiveQueueWorker._run_task` | 单函数较长，包含运行、marker、后处理、预览信号 | 后续拆到 `services/advanced_queue_service.py` 或 `core` 层运行协调器 |
| `_build_*_tab` 系列 | 多个页签 UI 仍直接在主窗口构建 | 按页签逐步迁移到 `gui/advanced_pages/` |
| 历史相关方法 | 与易用界面历史服务存在相似职责 | 逐步复用 `services/easy_history_service.py` 或抽通用历史服务 |
| 环境页逻辑 | 与易用界面环境服务有重叠 | 后续接入 `services/environment_service.py` 的格式化诊断输出 |
| 队列运行逻辑 | 与 easy batch 服务边界尚未统一 | 不改 fingerprint/marker 前提下，抽高级队列服务 |

## 建议拆分顺序

不要一次性重构。建议顺序：

```text
1. gui/advanced_pages/env_tab.py
2. gui/advanced_pages/history_tab.py
3. gui/advanced_pages/queue_tab.py
4. gui/advanced_widgets/canvases.py
5. services/advanced_queue_service.py
```

优先拆 UI 构建，再拆服务协调。不要先改运行语义。

## 当前不建议立刻修改的内容

- `job_fingerprint` / `job_id_for` / marker 语义；
- `tasks_from_manifest` 的任务构造；
- `GprMaxRunOptions` 和 `build_gprmax_command`；
- `.out` 合并和 B-scan 后处理；
- 历史记录删除语义；
- 高级界面页签数量和用户入口。

## 风险分级

### P0

暂无确认。

### P1

真实 gprMax + CUDA + conda 环境仍需目标机验证。高级界面的 live queue 不能仅凭离屏自测证明真实运行稳定。

### P2

- `main_window.py` 过大；
- worker、UI、历史、环境、队列协调混在一个文件；
- 与 v0.7 易用界面的服务层存在可复用但尚未统一的逻辑。

### P3

- 高级页签文案和易用界面文案风格不完全一致；
- 部分按钮和提示更偏工程开发者，不适合普通用户，但这是高级入口可接受的边界。

## 下一步建议

先不要大改高级工程界面。建议在真实 gprMax smoke test 模板补齐后，先跑目标机验收；如果运行链路稳定，再拆 `gui/advanced_pages/env_tab.py`，让高级环境页也复用当前环境诊断服务。
