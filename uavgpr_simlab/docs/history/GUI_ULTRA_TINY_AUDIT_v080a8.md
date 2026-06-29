# GUI_ULTRA_TINY_AUDIT_v080a8

## 背景

用户在 v0.8.0-alpha.7 运行 ultra tiny 后上传 `bscan_qc_report.json` 和 `dataset_summary.json`。报告显示五个 variant 的 `outputs/*.npy` 均存在且 shape 为 `61 × 2`，但全部为 NaN placeholder，`finite_count=0`，最终 `raw_minus_target_computable=false`，`clutter_gt_generated=false`。

## 结论

这不是普通 B-scan 数值异常，而是 `.out → .npy` 替换链路没有成功完成。v0.8.0-alpha.7 的最终 QC 只能看到 placeholder 仍为 NaN，因此报 `nan_or_inf_present`，但没有把 gprMax 运行失败、`.out` 缺失、shape mismatch 或解析失败充分暴露到 GUI。

## v0.8.0-alpha.8 修复

1. 设置与帮助页新增 “运行 ultra tiny 全链路验证”按钮。
2. 新增 GUI 后台 worker：`SceneWorldUltraTinyWorker`。
3. 服务层 `run_sceneworld_bscan_from_manifest()` 新增 `progress_callback`，向 GUI 实时输出 case、variant、stdout tail、状态。
4. `scripts/run_all_gprmax.py` 新增 `--allow-resample`。
5. ultra tiny BAT 默认启用 `--allow-resample`，并保留原始提取 B-scan：`outputs/<variant>_bscan_extracted_raw.npy`。
6. 正式 smoke/pilot 默认仍不启用 `--allow-resample`。
7. gprMax returncode 非零时写入明确错误，例如 `gprMax_returncode_1`。
8. `bscan_qc_report.json` 中补充 `gprmax_run_ok`、`gprmax_error`、`merged_shape`、`out_files`、`resampled` 等诊断字段。

## 边界

- 本轮不改变 SceneWorld 地质模型语义。
- 本轮不改变正式 smoke/pilot 的严格 QC 要求。
- 本轮不承诺当前 sandbox 完成 Windows 实机 gprMax 5-variant 求解；该验证仍需用户目标机运行。
