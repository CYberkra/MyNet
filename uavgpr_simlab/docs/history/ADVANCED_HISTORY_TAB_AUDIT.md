# ADVANCED_HISTORY_TAB_AUDIT - 高级历史页签拆分审计

## 本轮目标

将高级工程界面 `main_window.py` 中的“6 历史记录”页签 UI 构建逻辑拆分到独立页面组件：

```text
src/uavgpr_simlab/gui/advanced_pages/history_tab.py
```

本轮只迁移 UI 构建和控件引用，不改变历史扫描、缩略图生成、B-scan 加载、导出、删除、自动刷新或 marker 语义。

## 新增模块职责

`history_tab.py` 负责：

- 创建历史页签根页面；
- 创建 workspace 输入和选择按钮；
- 创建状态筛选、缩略图开关、自动刷新开关和最大显示数量；
- 创建刷新、导出、删除和彻底删除按钮；
- 创建历史记录表格；
- 创建模型画布、B-scan 画布和详情文本框；
- 通过 `AdvancedHistoryTabWidgets` 返回主窗口后续需要访问的控件引用。

## 保留在 main_window.py 的职责

以下逻辑刻意保留在 `main_window.py`，避免本轮过度迁移：

- `refresh_history()` 历史扫描与表格填充；
- `_history_auto_refresh_tick()` 自动刷新调度；
- `_history_marker_data()` marker JSON 读取；
- `_set_thumb_cell()` 缩略图单元格填充；
- `preview_history_selected()` 历史详情、模型画布和 B-scan 画布刷新；
- `export_history()` 历史 CSV 导出；
- `delete_selected_history()` 历史记录和输出删除；
- `QTimer` 生命周期。

这些函数和 core 层 `history.py`、`visual_history.py`、`postprocess.py` 仍保持原语义。

## 修改边界

本轮不修改：

- `scan_simulation_history()`；
- `build_history_preview()`；
- `load_bscan_for_history()`；
- `export_history_csv()`；
- `delete_history_record()`；
- marker 文件格式；
- job fingerprint；
- gprMax 运行命令；
- B-scan 后处理逻辑。

## 验证要点

自测新增高级历史页签控件断言：

```text
advanced_history_filter_count: 7
advanced_history_filter_current: 全部
advanced_history_table_columns: 13
advanced_history_log_readonly: true
advanced_history_detail_readonly: true
```

同时完成：

```bash
python -m compileall -q src scripts
PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/self_test.py
PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/make_v070_product_screenshots.py
PYTHONPATH=src python scripts/smoke_gprmax_source.py --gprmax-root /mnt/data/_gprmax_v317_read/gprMax-v.3.1.7 --work-dir workspace/gprmax_source_smoke_v0716 --omp-threads 1 --timeout 180
```

## 当前效果

- `main_window.py` 从约 1246 行降到约 1232 行。
- 高级工程界面已经拆出：
  - `advanced_pages/env_tab.py`
  - `advanced_pages/history_tab.py`
- 历史页签 UI 与历史业务逻辑边界更清楚。

## 风险

- P0：暂无。
- P1：真实 Windows + GPU gprMax 仍需目标机验证。
- P2：`main_window.py` 仍偏大，下一步应继续拆 `queue_tab.py`。
- P3：历史表格填充和缩略图刷新仍在主窗口，后续可在页面层稳定后再拆 helper 或 service。

## 下一步建议

1. 拆 `src/uavgpr_simlab/gui/advanced_pages/queue_tab.py`。
2. 再考虑 `gui/advanced_widgets/canvases.py`，统一高级界面的 Matplotlib / 3D 画布。
3. 页面层稳定后，抽 `services/advanced_queue_service.py`，把高级批量队列逻辑从主窗口移出。
