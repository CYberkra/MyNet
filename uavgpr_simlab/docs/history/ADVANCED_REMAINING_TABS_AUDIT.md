# ADVANCED_REMAINING_TABS_AUDIT - v0.7.23

## 本轮目标

一次性完成高级工程界面剩余相对独立页签的 UI 构建器拆分，避免 `main_window.py` 继续承担所有页面布局细节。

## 已拆分模块

```text
src/uavgpr_simlab/gui/advanced_pages/dashboard_tab.py
src/uavgpr_simlab/gui/advanced_pages/generation_tab.py
src/uavgpr_simlab/gui/advanced_pages/model_preview_tab.py
src/uavgpr_simlab/gui/advanced_pages/preflight_tab.py
src/uavgpr_simlab/gui/advanced_pages/qc_tab.py
src/uavgpr_simlab/gui/advanced_pages/train_tab.py
```

## 保持不变的边界

本轮只迁移 UI 构建和控件挂接，不改变以下语义：

- gprMax 命令构造；
- `job_id` / fingerprint / marker；
- manifest 生成与读取；
- 预检去重；
- `.out` 后处理与 ML 产品导出；
- 真实 CSV 解析与 QC 导出；
- 实时 B-scan 预览；
- 历史扫描、导出、删除。

## 拆分后职责

### `main_window.py`

继续负责：

- 跨页状态；
- 按钮回调；
- 服务层调用；
- worker 生命周期；
- 表格数据填充；
- 画布刷新；
- 错误弹窗和状态栏。

### `gui/advanced_pages/*_tab.py`

仅负责：

- 创建控件；
- 设置默认值；
- 连接传入的回调；
- 返回 dataclass 形式的控件引用。

## 验证

本轮新增/增强自测断言：

- 高级工作台摘要只读；
- 仿真计划默认 YAML、case 覆盖值、组件数量；
- 3D 预览表格列数、信息框只读；
- 预检 variants、limit、任务表列数；
- QC 文本框只读和导出按钮；
- 训练页文本只读。

## 后续建议

`main_window.py` 已降至 900 行以内。下一阶段不建议继续大拆页面，而应优先：

1. 补真实 Windows + CUDA + GPU 目标机验收；
2. 为 `LiveQueueWorker` 增加 marker / skip / failed / geometry-only 回归测试；
3. 再考虑把 `LiveQueueWorker` 的运行生命周期服务化。
