# HOME_PAGE_AUDIT - v0.7.10

## 本轮目标

将产品化易用界面的“首页” UI 构建逻辑从 `easy_window.py` 拆出到独立页面组件，完成 v0.7 易用界面 6 个主页面构建器的分离。

## 修改边界

新增：

```text
src/uavgpr_simlab/gui/pages/home_page.py
```

该模块只负责构建和返回首页控件：

- 当前项目状态卡；
- 已生成模型数卡；
- 正在仿真任务卡；
- 已完成任务卡；
- 需检查任务卡；
- 最近 B-scan 预览画布；
- “下一步”提示区；
- 首页流程步骤条；
- 跳转到模型预览页和批量仿真页的按钮信号挂接。

## 未改变内容

本轮没有改变：

- 首页统计来源；
- `services/easy_history_service.py` 中的项目摘要统计；
- 历史扫描逻辑；
- 最近/示例 B-scan 刷新逻辑；
- 模型生成、manifest、批量预检、任务运行逻辑；
- gprMax 调用语义；
- 任务 fingerprint / history marker；
- B-scan 后处理。

## 当前职责划分

```text
gui/pages/home_page.py
  只负责首页 UI 构建、静态控件返回和页面跳转信号挂接

gui/easy_window.py
  保留跨页面状态、首页统计刷新、最近 B-scan 画布刷新和服务调用

services/easy_history_service.py
  保留首页项目摘要统计和历史扫描/详情服务
```

## 验证内容

本轮验证应至少包含：

```bash
python -m compileall -q src scripts
PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/self_test.py
PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/make_v070_product_screenshots.py
```

自测新增断言：

- Easy GUI 仍有 6 个页面；
- 首页 B-scan 画布已挂接；
- 首页下一步提示支持自动换行；
- 首页项目指标卡的数值标签已挂接。

## 风险评估

- P0：未发现。
- P1：未触碰真实 gprMax / CUDA / GPU 求解链路。
- P2：`easy_window.py` 已进一步收敛；v0.7 易用界面的 6 个主页面均已有独立页面构建器。后续重点应转向真实环境验证和高级工程界面治理。
- P3：首页统计刷新和示例 B-scan 刷新仍在 `easy_window.py`，这是刻意保留的低风险边界；后续可按需要拆出首页刷新 helper 或 dashboard service。

## 后续建议

产品化易用界面的页面构建器拆分已完成。下一轮建议不要继续做无目标的小拆分，应转向：

1. 补真实 gprMax smoke test 记录模板和 Windows 目标机验收流程；
2. 审视 `main_window.py` 高级工程界面的体积和职责边界；
3. 按真实运行链路完善 conda / CUDA / pycuda 失败诊断文案。
