# UavGPR-SimLab v0.8.0-alpha.14 联动完善审计

## 本轮目标

本轮不改变 SceneWorld 物理语义，不扩充 case 数量，专门完善以下链路：

1. 运行结果 → 历史记录页：dataset / case / variant 树状浏览、多图对比、失败定位；
2. 失败 → 恢复 / 续跑：自动跳过已完成、只重跑 failed、强制重跑；
3. 停止任务：SceneWorld GUI worker 可请求终止当前 gprMax 子进程。

## 已完成

### 1. 历史记录页升级

历史页新增 dataset / case / variant 树状浏览。SceneWorld 结果不再只以扁平卡片展示，而是按以下层级组织：

```text
dataset
  case_id / family
    raw
    target_only
    background_only
    clutter_only
    air_only
    clutter_gt
```

用户选中 variant 后，右侧可查看模型、B-scan、QC 路径、manifest 路径、失败原因。

### 2. 多图对比

历史页新增 B-scan 视图选择：

```text
当前记录
raw
target_only
background_only
clutter_only
air_only
clutter_gt
对比：raw / target / clutter_gt
```

`MplCanvas` 新增 `show_bscan_grid()`，支持 raw / target / clutter_gt 横向对比。

### 3. 失败定位

历史页新增“失败定位 / 打开路径”按钮。它会显示：

```text
case_id
variant
manifest
case_dir
bscan_npy
qc_report_json
bscan_error
variant_qc
```

避免用户只能从日志里手工查失败位置。

### 4. 只重跑 failed

历史页新增“只重跑 failed”入口。选中 SceneWorld 记录后会自动切到批量仿真页，并设置：

```text
manifest = 当前数据集 manifest
只重跑 failed = true
自动跳过已完成 = true
强制重跑 = false
```

批量仿真页新增：

```text
自动跳过已完成
只重跑 failed
强制重跑
```

三种策略不再硬编码在按钮回调中。

### 5. 断点续跑 / 跳过已完成

`run_sceneworld_bscan_from_manifest()` 增加：

```python
skip_completed: bool = True
rerun_failed_only: bool = False
```

当 `skip_completed=True` 时，已有 finite 且 manifest 标记 success 的 aligned B-scan 会被跳过。

当 `rerun_failed_only=True` 时，仅运行 manifest 中 `bscan_status=failed` 的记录。

### 6. 停止当前 gprMax 子进程

`SceneWorldFullChainWorker` 增加 `cancel()`，内部使用 `threading.Event`。

`sceneworld_bscan_service._run_one_gprmax()` 从 `subprocess.run()` 改为 `subprocess.Popen()` 轮询。GUI 点击“停止”时会：

```text
请求 cancel_event
terminate 当前 gprMax 子进程
必要时 kill
保留已完成输出与报告
```

## 未改变

- 不改变 gprMax `.in` 物理语义；
- 不改变 `constant_level` 飞行模式说明；
- 不把 25-run smoke 伪装成训练数据；
- 不放宽 pilot/formal 的严格 QC 边界；
- 不新增依赖。

## 风险

- 停止功能依赖 OS 对子进程 terminate / kill 的响应；Windows 目标机仍需实测。
- 历史页树状浏览已接入 SceneWorld manifest / QC，但 GUI 离屏运行在当前 sandbox 缺 PySide6，仍需目标机验证视觉布局。
- “只重跑 failed”依赖 manifest 中 variant 级 `bscan_status`，旧数据集如果仍是 case 级状态，需要先用新 runner 跑一轮或刷新 manifest。

## 下一步

1. 用户在 Windows 目标机用 v0.8.0-alpha.14 重新跑 25-run smoke；
2. 验证历史页能看到 dataset/case/variant 树；
3. 故意停止一次任务，确认当前 gprMax 子进程可终止；
4. 制造 failed 后测试“只重跑 failed”。
