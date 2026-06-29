# SETTINGS_PAGE_AUDIT - v0.7.5

## 本轮目标

本轮目标是拆分产品化易用界面的“设置与帮助”页 UI 构建逻辑，继续降低 `easy_window.py` 的体积和职责复杂度。

## 已完成

- 新增 `src/uavgpr_simlab/gui/pages/` 页面组件包。
- 新增 `src/uavgpr_simlab/gui/pages/settings_page.py`。
- 新增 `SettingsPageWidgets`，集中返回主窗口需要持有的控件引用。
- 新增 `build_settings_help_page()`，负责构造：
  - gprMax 源码目录输入框；
  - conda 环境名输入框；
  - GPU ID 输入框；
  - OpenMP 线程选择；
  - GPU / conda run 开关；
  - 保存、检查、高级工程界面按钮；
  - 五步帮助说明；
  - 环境检查日志区。
- `easy_window.py` 的 `_build_help_page()` 改为页面组件装配，不再直接构造设置页全部 UI。

## 边界说明

本轮只拆 UI 构建，不改变：

- `.simlab_env` 读写格式；
- `EasyEnvironmentSettings` 字段；
- `run_easy_environment_diagnostics()` 环境诊断逻辑；
- `build_runtime_config_for_easy()` 运行配置组装逻辑；
- gprMax 调用命令；
- 任务去重、marker、B-scan 后处理和历史记录语义。

## 当前收益

- 设置页 UI 与环境服务层形成更清楚的边界。
- 后续优化设置页错误提示、诊断说明或布局时，不需要继续扩大主窗口文件。
- 为继续拆分 `project_page.py`、`batch_page.py`、`history_page.py` 建立了页面组件模式。

## 验证建议

```bash
python -m compileall -q src scripts
PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/self_test.py
PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/make_v070_product_screenshots.py
```

额外建议做离屏设置页烟测：

```bash
PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python - <<'PY'
from PySide6.QtWidgets import QApplication
from uavgpr_simlab.gui.easy_window import EasyMainWindow
app = QApplication.instance() or QApplication([])
win = EasyMainWindow()
win.show(); app.processEvents()
assert win.stack.count() == 6
assert win.help_log.isReadOnly()
assert win.conda_env_edit.text()
print(win.windowTitle())
PY
```

## 后续建议

1. 下一轮优先拆 `gui/pages/project_page.py`，与 `services/project_service.py` 对齐。
2. 然后拆 `gui/pages/batch_page.py` 和 `gui/pages/history_page.py`，但不要一次性移动运行线程和历史删除逻辑。
3. 页面拆分完成后，再处理真实 Windows + gprMax + CUDA smoke test 文档和目标机验收模板。
