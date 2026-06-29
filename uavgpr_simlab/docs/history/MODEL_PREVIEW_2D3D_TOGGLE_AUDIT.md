# MODEL_PREVIEW_2D3D_TOGGLE_AUDIT - v0.7.26

## 目标

响应“点击切换 2D/3D”的使用需求，在产品化易用界面的模型预览页提供明确的视图切换能力。

## 改动范围

- `gui/pages/model_preview_page.py`：新增“3D 视图 / 2D 剖面”切换按钮和 `QStackedWidget` 预览容器。
- `gui/easy_window.py`：新增模型预览模式状态、2D 剖面图刷新、当前模型行解析和 2D 图路径提示。
- `scripts/self_test.py`：新增模型预览切换控件断言。

## 行为

- 默认显示 3D 视图。
- 点击“2D 剖面”后，同一预览区域切换为当前模型的 2D 剖面图。
- 点击“3D 视图”后，同一预览区域切回 3D 画布。
- “打开 2D 剖面图路径”会自动切换到 2D 视图并提示 PNG 路径。
- 如果当前模型没有可读 2D 图，界面显示明确占位提示，不静默失败。

## 未改变事项

- 未改变模型生成逻辑。
- 未改变 manifest 结构。
- 未改变 gprMax 调用语义。
- 未改变任务 fingerprint / marker。
- 未改变 B-scan 后处理。

## 验证

- `python -m compileall -q src scripts`：通过。
- `PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/self_test.py`：通过。
- GUI 离屏烟测：生成 1 个模型后可在模型预览页从 3D 切换到 2D，再切回 3D。
