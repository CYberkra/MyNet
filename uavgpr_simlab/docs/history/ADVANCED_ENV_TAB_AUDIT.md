# ADVANCED_ENV_TAB_AUDIT - 高级环境页签拆分审计

## 本轮目标

在不改变高级工程界面运行语义的前提下，把 `main_window.py` 中的“1 环境检查”页签 UI 构建逻辑拆到独立页面构建器：

```text
src/uavgpr_simlab/gui/advanced_pages/env_tab.py
```

## 完成内容

1. 新增 `src/uavgpr_simlab/gui/advanced_pages/` 包。
2. 新增 `AdvancedEnvTabWidgets`，集中返回高级环境页签中的关键控件引用。
3. 新增 `build_advanced_env_tab()`，只负责构造环境页签 UI 和绑定主窗口传入的回调。
4. `main_window.py` 保留：
   - 本地环境变量保存；
   - 环境检查 report 写入；
   - gprMax smoke test 命令生成；
   - 4090 安装脚本入口；
   - 运行时配置读取；
   - 高级队列和历史逻辑。

## 未改变内容

本轮没有改变：

```text
gprMax 命令构造
conda run 语义
GPU / OpenMP 参数传递
任务 fingerprint
history marker
B-scan 后处理
队列运行 worker
历史扫描/删除/导出
```

## 架构影响

- 高级界面开始具备 `gui/advanced_pages/` 页面层边界。
- `main_window.py` 从约 1263 行降到约 1246 行。
- 高级环境页签的 UI 构建从主窗口移出，但业务逻辑暂时保持在主窗口内，降低一次性重构风险。

## 验证记录

已执行：

```bash
python -m compileall -q src scripts
PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/self_test.py
PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/make_v070_product_screenshots.py
PYTHONPATH=src python scripts/smoke_gprmax_source.py --gprmax-root /mnt/data/_gprmax_v317_read/gprMax-v.3.1.7 --work-dir workspace/gprmax_source_smoke_v0715 --omp-threads 1 --timeout 180
```

自测新增高级环境页签断言：

```text
advanced tabs: 10
advanced env log readonly: true
advanced env smoke button: 显示 gprMax smoke test 命令
```

## 风险分级

### P0

暂无发现。

### P1

真实 Windows + conda + CUDA + pycuda + GPU 目标机验证仍未完成。

### P2

`main_window.py` 仍偏大，后续应继续按高级页签拆分，建议顺序：

```text
gui/advanced_pages/history_tab.py
gui/advanced_pages/queue_tab.py
gui/advanced_widgets/canvases.py
services/advanced_queue_service.py
```

### P3

高级环境页签仍使用旧式工程文案，适合高级入口；暂不要求与产品化易用界面完全统一。
