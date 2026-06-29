# ADVANCED_REAL_CSV_TAB_AUDIT - v0.7.21

## 本轮目标

将高级工程界面“7 实测/弱监督”页签的 UI 构建逻辑从 `main_window.py` 拆出，降低高级主窗口继续膨胀的风险。

## 修改范围

新增：

```text
src/uavgpr_simlab/gui/advanced_pages/real_csv_tab.py
```

更新：

```text
src/uavgpr_simlab/gui/advanced_pages/__init__.py
src/uavgpr_simlab/gui/main_window.py
scripts/self_test.py
```

## 已迁移内容

- CSV 路径输入框；
- 选择 CSV 按钮；
- 最大道数设置；
- 加载并预览按钮；
- 显示 f-k 图按钮；
- 导出 NPZ/PNG 质控按钮；
- Real CSV B-scan 画布；
- CSV 信息 / QC 输出文本框。

## 刻意保留在 main_window.py 的内容

以下内容本轮没有迁移，避免改变数据处理语义：

- `read_uavgpr_csv()` 调用；
- `subtract_mean_background()` 调用；
- `robust_normalize()` 调用；
- `convert_real_csv()` 调用；
- f-k 预览数据来源；
- CSV 读取失败弹窗；
- 后台导出 worker 调度。

## 架构边界

`real_csv_tab.py` 只负责控件创建和信号绑定。真实 CSV 解析、弱监督数据导出和质控结果生成仍由 `core.real_data` 与 `main_window.py` 回调承接。

## 验证要点

- 高级界面仍保持 10 个页签；
- 默认 CSV 路径仍指向 `sample_data/Line9origin36_first16traces.csv`；
- 最大道数默认仍为 300；
- CSV 信息框只读；
- 加载、f-k、导出按钮文案保持不变；
- Real CSV 处理自测仍通过。

## 风险

- P0：暂无。
- P1：暂无，本轮不改真实 CSV 处理算法和导出语义。
- P2：CSV 处理逻辑仍在 `main_window.py`，后续可考虑抽 `services/real_csv_service.py`。
- P3：高级页签文案仍偏工程化，适合高级入口，暂不调整。
