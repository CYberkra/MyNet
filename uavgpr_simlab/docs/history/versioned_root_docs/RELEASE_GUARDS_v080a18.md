# UavGPR-SimLab v0.8.0-alpha.18 发布守卫说明

本轮新增四类不依赖 4090 真机的守卫脚本。它们不能替代 Windows/CUDA/PyCUDA/gprMax GPU 验收，但可以在发包前发现版本、入口、页面合同、Windows 脚本和数据包清单问题。

## 固定检查命令

```bash
python -m compileall -q src scripts
PYTHONPATH=src python scripts/check_architecture_guard.py
PYTHONPATH=src python scripts/check_easy_ui_contract.py
PYTHONPATH=src python scripts/check_windows_script_contract.py
PYTHONPATH=src python scripts/check_release_integrity.py
```

## 脚本边界

- `check_architecture_guard.py`：防止 `easy_window.py` 和 controller 重新膨胀，保持 `docs/history/` 归档。
- `check_easy_ui_contract.py`：静态检查 EasyMainWindow 的 mixin 顺序、页面构建顺序、关键 controller 方法、页面 widget dataclass 和按钮回调；如果当前环境安装了 PySide6，会额外 offscreen 实例化主窗口。
- `check_windows_script_contract.py`：静态检查 BAT / PowerShell / runtime bootstrap / 生成数据集 BAT 的调用链、错误码和环境变量合同。
- `check_release_integrity.py`：聚合版本、入口、4090 run plan、`.simlab_env`、workspace skeleton 和 docs 根目录检查。
- `audit_yingshan_real_data_package.py`：读取用户提供的营山真实数据 zip，不把大文件解压进发布包，只生成清单和格式预审报告。

## 仍需真机证明的内容

- PySide6 GUI 在 Windows 目标机的真实启动和截图。
- NVIDIA driver、CUDA Toolkit、MSVC、PyCUDA 与 gprMax `-gpu` 的真实运行。
- 1 case × 5 variant、48 case validation、500 case formal 的真实 `.out`、`.npy`、QC 和历史页显示。
