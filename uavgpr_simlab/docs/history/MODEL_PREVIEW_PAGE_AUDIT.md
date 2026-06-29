# MODEL_PREVIEW_PAGE_AUDIT - v0.7.9

## 本轮目标

将产品化易用界面的“模型预览”页 UI 构建逻辑从 `easy_window.py` 拆出到独立页面组件，继续降低主窗口体积和后续维护风险。

## 修改边界

新增：

```text
src/uavgpr_simlab/gui/pages/model_preview_page.py
```

该模块只负责构建和返回模型预览页控件：

- 模型清单输入框；
- 选择已有模型清单按钮；
- 加载模型图库按钮；
- 生成一批模型按钮；
- 左侧模型图库列表；
- 中间 3D 模型预览画布；
- 右侧模型信息标签；
- 查看 3D 大图、查看剖面缩略图、加入批量仿真按钮。

## 未改变内容

本轮没有改变：

- 模型生成服务；
- manifest 结构；
- manifest 读取逻辑；
- 预览 PNG 生成逻辑；
- `Model3DCanvas.show_label_json()` 的 3D label 解析；
- 加入批量仿真的同步语义；
- gprMax 调用语义；
- 任务 fingerprint / history marker；
- B-scan 后处理。

## 当前职责划分

```text
gui/pages/model_preview_page.py
  只负责模型预览页 UI 构建和信号挂接

gui/easy_window.py
  保留跨页面状态、manifest 加载、模型预览刷新、批量页同步和服务调用

services/project_service.py
  保留计划预览和模型批次生成协调

core/scenario.py / core/visual_history.py
  保留模型生成与模型预览图渲染
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
- 模型清单输入框正常挂接；
- 模型图库列表宽度和图标尺寸正常；
- 模型信息标签数量为 5；
- 3D 模型画布已挂接。

## 风险评估

- P0：未发现。
- P1：未触碰真实 gprMax / CUDA / GPU 求解链路。
- P2：`easy_window.py` 继续收敛，但首页 UI 仍在主窗口内。
- P3：模型图库列表刷新逻辑仍在 `easy_window.py`，这是刻意保留的低风险边界；后续可在页面层稳定后再继续拆列表刷新 helper。

## 后续建议

下一轮建议拆 `gui/pages/home_page.py`，把首页状态卡、最近 B-scan 预览和下一步提示从 `easy_window.py` 中移出。完成后，产品化易用界面的 6 个主页面均有独立页面构建器。
