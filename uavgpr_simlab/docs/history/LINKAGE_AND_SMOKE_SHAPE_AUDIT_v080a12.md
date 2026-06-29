# v0.8.0-alpha.12 链路联动与 25-run shape 中断修复审计

## 背景

用户在 Windows GUI 中运行 25-run smoke 时，`case_000001/raw` 与 `target_only` 已完成 gprMax 求解并生成 `.out`，但 runner 报告：

```text
shape_mismatch_without_resample: got [1909, 72], expected [501, 72]
```

这说明环境与 gprMax 调用已经成立，失败点不是 `.in` 或 `.out` 缺失，而是 gprMax FDTD 原生时间采样数与 SceneWorld manifest 中面向 ML 网格的 `time_axis_ns.npy` 长度不一致。

## 结论

- ultra tiny：仍作为最小链路验证，允许显式重采样。
- 25-run smoke：定位为五类场景链路验证，允许显式重采样到 manifest 网格；报告必须记录 native shape、target shape、resampled 状态，不能静默处理。
- pilot/formal：保持严格 QC，不默认自动重采样。若需要 ML 训练网格，应走后续明确的数据导出/对齐步骤。

## 修复内容

1. `smoke_25run` profile 改为 `chain_resample` QC 模式，允许显式对齐到 manifest 目标网格。
2. `scripts/run_all_gprmax.py --allow-resample` 帮助说明改为链路验证用途，而非仅 ultra tiny。
3. 数据集内 `logs/run_all_gprmax.bat` 对 smoke 也显式传入 `--allow-resample`。
4. `bscan_qc_report.json` 增加：
   - `alignment_policy`
   - `resampled_variants`
   - `native_shapes`
   - `training_ready`
   - 每个 variant 的 `raw_extract_path` / `aligned_shape`
5. `dataset_summary.json` 增加：
   - `qc_mode`
   - `allow_resample`
   - `training_ready`
   - `training_ready_cases`
   - `resampled_cases`

## 风险边界

- 本轮不把 25-run smoke 的显式重采样输出标记为训练可用。
- 本轮不改变 gprMax `.in` 文件物理含义。
- 本轮不改变 pilot/formal 的默认严格 QC 策略。
- 本轮不把 1909 点原生输出直接丢弃，保留在 `outputs/<variant>_bscan_extracted_raw.npy`。

## 验收建议

在 Windows 上运行：

```text
批量仿真 → 运行配置：五类场景 smoke：25-run → 应用运行配置 → 开始运行统一任务
```

预期：

- 25 个 variant 均完成 gprMax 求解；
- 原生输出 shape 记录为类似 `[1909, 72]`；
- 对齐输出 shape 为 `[501, 72]`；
- `clutter_gt_bscan.npy` 生成；
- `dataset_summary.json` 中 `status=success`，但 `training_ready=false`，因为该 smoke 是链路验证而非训练数据。
