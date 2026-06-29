# ADVANCED_CANVASES_AUDIT - v0.7.18

## 本轮目标

将高级工程界面中的 Matplotlib 画布类从 `main_window.py` 拆出，降低主窗口文件职责密度，为后续队列服务层和更多高级页签治理做准备。

## 修改范围

新增：

```text
src/uavgpr_simlab/gui/advanced_widgets/
src/uavgpr_simlab/gui/advanced_widgets/canvases.py
```

迁移内容：

- `MplCanvas`
  - B-scan 显示；
  - f-k 幅值预览；
  - 灰度归一化显示窗口。
- `Model3DCanvas`
  - 空状态显示；
  - `label_json` 读取；
  - 地表、基覆界面和 UAV 高度 3D/2.5D 预览。

## 保留在 main_window.py 的内容

`main_window.py` 仍保留：

- 高级界面整体窗口状态；
- 高级页签回调；
- 队列 worker 启停；
- 历史扫描和详情刷新；
- B-scan 数据来源协调；
- gprMax 命令构造和 marker/fingerprint 调用。

## 明确未改内容

本轮没有改变：

- B-scan 后处理算法；
- 3D 预览数据结构；
- `label_json` 字段语义；
- 正式 gprMax 运行命令；
- conda / GPU / OpenMP 参数传递；
- 任务 fingerprint；
- history marker；
- 队列运行流程。

## 验证记录

已执行：

```bash
python -m compileall -q src scripts
PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/self_test.py
PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/make_v070_product_screenshots.py
PYTHONPATH=src python scripts/smoke_gprmax_source.py --gprmax-root /mnt/data/_gprmax_v317_read/gprMax-v.3.1.7 --work-dir workspace/gprmax_source_smoke_v0718 --omp-threads 1 --timeout 180
```

自测新增断言：

- `queue_canvas` 是 `MplCanvas`；
- `history_bscan_canvas` 是 `MplCanvas`；
- `model3d_canvas` 是 `Model3DCanvas`；
- 高级界面仍保持 10 个页签；
- 易用界面仍保持 6 个页面。

## 风险评估

- P0：未发现。
- P1：未涉及真实 Windows/CUDA/GPU 链路，仍需目标机验证。
- P2：`main_window.py` 仍偏大，但画布类已独立，可继续抽 `services/advanced_queue_service.py`。
- P3：画布类仍直接使用 Matplotlib，当前符合项目依赖；后续若引入更复杂交互再考虑细分。
