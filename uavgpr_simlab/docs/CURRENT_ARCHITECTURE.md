# 当前架构治理说明

本文件是当前稳定架构入口。历史版本化说明已归档到 `docs/history/versioned_root_docs/`。

## 分层边界

```text
src/uavgpr_simlab/
├─ app.py                         # GUI 入口选择
├─ cli.py                         # CLI 命令入口
├─ core/                          # 数据模型、manifest、runner、SceneWorld 生成核心
├─ services/                      # 环境诊断、批量任务、历史记录、真实数据、SceneWorld B-scan 服务
├─ simulation/                    # 仿真相关辅助逻辑
├─ io/                            # 输入输出和格式处理
├─ visualization/                 # 图像、B-scan、画布辅助
└─ gui/
   ├─ easy_window.py              # 产品化主窗口壳层
   ├─ controllers/                # 页面行为 controller mixin
   ├─ pages/                      # 页面构建
   ├─ advanced_pages/             # 高级工程界面页签
   └─ advanced_widgets/           # 高级画布和控件
```

## 当前 UI 边界

`easy_window.py` 只作为主窗口壳层，负责窗口初始化、导航、共享状态和跨页面辅助函数。页面行为放在 `gui/controllers/`，页面布局放在 `gui/pages/`，业务协调放在 `services/`。

架构守卫：

```bash
PYTHONPATH=src python scripts/check_architecture_guard.py
```

当前守卫项：

- `easy_window.py` 不超过 350 行；
- 单个 controller 不超过 450 行；
- `docs/` 根目录不出现版本化历史文档或旧 UI 总览图；
- `docs/history/` 归档存在。

## 当前运行时边界

Windows 运行统一通过：

```text
scripts/windows_runtime_bootstrap.bat
```

GUI、验证脚本、生成数据集脚本和 run_all_gprMax 脚本都应使用 bootstrap 选出的 `%PY_RUN%`，不应直接调用系统 `python`。

## 当前文档边界

`docs/` 根目录只保留稳定入口文档。版本化历史、阶段性审计、旧截图和旧报告归档到：

```text
docs/history/versioned_root_docs/
docs/history/real_data/
docs/history/ui_overview_images/
```

新增阶段性报告默认放入 `docs/history/`，不要再使用 `docs/*_v080aXX.md` 作为当前入口。
