# HISTORY_PAGE_AUDIT - v0.7.8 历史与结果页拆分记录

## 本轮目标

将产品化易用界面的“历史与结果”页 UI 构建逻辑从 `easy_window.py` 中拆出，形成独立页面构建器，并继续让主窗口只负责状态协调、服务调用和事件处理。

本轮只处理 UI 构建和控件引用回传，不改变历史扫描、B-scan 加载、历史导出、删除记录或 marker 语义。

## 新增文件

```text
src/uavgpr_simlab/gui/pages/history_page.py
```

该文件提供：

```text
HistoryPageWidgets
build_history_page(...)
```

## 拆分内容

`history_page.py` 现在负责构建：

- 历史页标题和说明；
- 状态筛选下拉框；
- 刷新、导出、重新运行、删除按钮；
- 历史记录列表；
- 右侧模型画布；
- 右侧 B-scan 结果画布；
- 历史详情说明标签；
- 历史页顶部模式标签。

`easy_window.py` 仍负责：

- 根据当前 workspace 扫描历史记录；
- 调用 `services/easy_history_service.py` 构建历史详情；
- 将历史记录对象绑定到列表项；
- 根据选择项刷新模型画布和 B-scan 画布；
- 导出历史 CSV；
- 删除历史记录；
- 与批量页之间的“重新运行”页面跳转。

## 未改变内容

本轮没有改变：

- `scan_history_entries()` 语义；
- `build_history_detail()` 语义；
- `export_history_report()` 语义；
- `delete_history_entry()` 语义；
- history marker 文件结构；
- B-scan 加载与显示数据；
- 任务 fingerprint / done marker / running marker；
- gprMax 调用命令和运行配置。

## 验证方式

推荐验证：

```bash
python -m compileall -q src scripts
PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/self_test.py
PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/make_v070_product_screenshots.py
```

离屏 GUI 烟测应至少确认：

```text
pages == 6
history_status_filter.count() == 5
history_status_filter.currentText() == "全部"
history_list.spacing() == 8
history_detail_easy.wordWrap() == True
```

## 风险判断

- P0：未发现。
- P1：真实 gprMax / CUDA / Windows 运行链路仍需目标机验证。
- P2：`easy_window.py` 已继续收敛，但首页和模型预览页构建仍在主窗口内。
- P3：历史页列表项刷新逻辑仍在 `easy_window.py`，这是刻意保留的低风险边界；后续可在页面层稳定后再考虑拆分列表刷新 helper。

## 后续建议

1. 下一轮拆 `gui/pages/model_preview_page.py`，降低模型图库和模型说明 UI 对主窗口的占用。
2. 再拆 `gui/pages/home_page.py`，把首页状态卡和最近 B-scan 预览构建移出。
3. 页面层稳定后，补真实 gprMax smoke test 记录模板。
