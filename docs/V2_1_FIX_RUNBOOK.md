# PGDA-CSNet v2.1-fix Agent Runbook

## 0. 目标

本包不是重新发明一个大模型，而是在当前 `master` 的 `GprMambaSep` 基础上做 **Route 2：G-assisted CurveMamba / CurvePicker**。

核心变化：

```text
当前 master：raw -> A/S/G -> G feature -> mask/center/presence -> DP
v2.1-fix： raw -> shared encoder + A/S/G auxiliary
                    -> shared + G + raw-local fused task feature
                    -> curve_logits / presence / global_no_target / aux mask
                    -> P(t|trace) + DP
```

## 1. 已改文件

```text
pgdacsnet/model_gprmambasep.py
pgdacsnet/model_interfaces.py
pgdacsnet/model_raw_unet.py
scripts/losses_gprmambasep.py
scripts/eval_full_line.py
scripts/audit_component_arrays.py
scripts/plot_gprmambasep_diagnostics.py
configs/gpu_mixed_v2_1_gprmambasep_lite_line9holdout_reviewval.json
configs/gpu_mixed_v2_1_curvegassist_line9holdout_6g.json
configs/gpu_mixed_v2_1_curvegassist_line9holdout_12g.json
tests/test_curvegassist_smoke.py
```

## 2. 静态验证

```bash
python -m py_compile \
  pgdacsnet/model_interfaces.py \
  pgdacsnet/model_gprmambasep.py \
  pgdacsnet/model_raw_unet.py \
  scripts/losses_gprmambasep.py \
  scripts/eval_full_line.py \
  scripts/audit_component_arrays.py \
  scripts/plot_gprmambasep_diagnostics.py

python -m pytest tests/test_curvegassist_smoke.py -q
```

我已在当前容器通过：`1 passed`。第二轮审计还额外跑过 `find pgdacsnet scripts tests -name "*.py" -print0 | xargs -0 python -m py_compile`，全部通过。

## 3. 先不要直接大训练，先跑三件事

### 3.1 复查 component 数组覆盖率

```bash
python scripts/audit_component_arrays.py \
  --data-root data_corrected_v1_4_terrain_direction \
  --out reports/component_array_coverage_real.csv

python scripts/audit_component_arrays.py \
  --data-root data/simulation_pretrain_v1 \
  --out reports/component_array_coverage_sim.csv
```

判定：如果 `G_target` / `Y_target_without_G` 覆盖率低，Route 1（继续 A/S/G 主线并补 component supervision）不能作为当前主线。

### 3.2 用 review-val 规范版复现当前 GprMambaSep

```bash
python scripts/train_raw_only.py \
  configs/gpu_mixed_v2_1_gprmambasep_lite_line9holdout_reviewval.json
```

注意：这个配置把 `LineX1` 从 review 改为 validation，Line9 仍然严格 holdout。

### 3.3 训练 6G 版 G-assisted CurveMamba

```bash
python scripts/train_raw_only.py \
  configs/gpu_mixed_v2_1_curvegassist_line9holdout_6g.json
```

12G 显存用：

```bash
python scripts/train_raw_only.py \
  configs/gpu_mixed_v2_1_curvegassist_line9holdout_12g.json
```

## 4. Line9 holdout 评估

6G 版：

```bash
python scripts/eval_full_line.py \
  --run-dirs outputs/run_curvegassist_lite_mixed_v2_1_line9holdout_6g \
  --line Line9 \
  --checkpoint best \
  --out-dir outputs/eval_curvegassist_lite_line9holdout_6g \
  --search-min-ns 320 \
  --search-max-ns 560 \
  --presence-thr 0.45 \
  --path-prob-thr 0.20 \
  --dp-max-jump 8 \
  --dp-smooth-weight 0.08
```

12G 版把 run-dir/out-dir 换成：

```text
outputs/run_curvegassist_small_mixed_v2_1_line9holdout_12g
outputs/eval_curvegassist_small_line9holdout_12g
```

说明：`eval_full_line.py` 已支持 `curve_logits`。默认当模型有 `curve_logits` 时优先使用 `softmax(curve_logits, dim=time)` 作为 DP 路径概率，并在 metrics 中记录 `curve_source=curve_logits_dp`。要强制使用旧 mask，请加：

```bash
--disable-curve-logits
```

## 5. 生成六类诊断图

```bash
python scripts/plot_gprmambasep_diagnostics.py \
  --line Line9 \
  --data-root data_corrected_v1_4_terrain_direction \
  --eval-dir outputs/eval_curvegassist_lite_line9holdout_6g \
  --out-dir outputs/diagnostics_curvegassist_lite_line9_6g
```

若先跑了 separation eval，也可以加：

```bash
--sep-dir outputs/eval_gprmambasep_sep_line9
```

输出：

```text
01_raw_gt_dp_overlay.png
02_ghat_gt_overlay.png        # sep-dir 有 G_hat 时生成
03_mask_center_curve_heatmap.png
04_presence_trace.png
05_asg_residual_panels.png    # sep-dir 有 A/S/G 时生成
06_false_path_casebook.pdf
```

## 6. 结果判据

必须同时看 pre-DP 与 post-DP：

```text
mean_center_mae_ns       # pre-DP，上游 curve/center 是否真的准
DP Center MAE            # post-DP，最终连续拾取结果
final_pick_rate          # 不能明显低于当前 0.56 左右
presence_false_pick_rate_no_pick
```

阶段性目标：

```text
1) 先让 mean_center_mae_ns 明显低于当前 66.72 ns；
2) 再让 DP Center MAE 从 25.24 ns 压到 8 ns 以下；
3) 最后再尝试接近或超过 P0-3 的 3.268 ns。
```

## 7. 止损规则

```text
若 Route 2 的 pre-DP MAE 没有改善：检查 curve loss、task_feature_mode、eval metrics 中 `curve_source` 是否为 `curve_logits_dp`。
若 Route 2 的 post-DP 仍 > 8 ns 且 no-pick FPR 没下降：冻结为研究分支，不替换 P0-3。
若 G_hat 与 GT band 仍完全无关：A/S/G 只保留为辅助可解释分支，不再作为主任务路径。
```

## 8. 变更摘要

- 新增 `v2_1_curvegassist_lite` 架构别名。
- 新增 `task_feature_mode="g_assisted"`。
- 新增 `curve_logits`、`global_no_target_logits`、`uncertainty_logits` 输出接口。
- 新增 trace-wise curve distribution loss。
- 新增 global no-target loss。
- eval 默认优先使用 `curve_logits` 的 softmax 概率进入 DP。
- 新增 component arrays 覆盖率审计脚本。
- 新增六类诊断图脚本。

## 9. 第三轮审计补充：uncertainty head

12G 配置保留 `enable_uncertainty_head=true`，但现在已经补上低权重监督：

```json
"enable_uncertainty_head": true,
"loss": {
  "uncertainty_weight": 0.02
}
```

该 loss 不是独立真值不确定性监督，而是基于 curve center error 的 heteroscedastic NLL。用途是让 uncertainty 输出获得有效梯度，作为弱校准/诊断项；不要把它作为最终主指标。

6G 配置仍关闭 uncertainty head，以减少显存与不稳定因素。

验收时额外看：

```text
train_uncertainty_nll
val_uncertainty_nll
```

若 uncertainty loss 明显导致主指标变差，先把 `uncertainty_weight` 置 0 或关闭 `enable_uncertainty_head`，不要影响 Route-2 主线判断。

## 10. 第四轮审计补充：诊断图路径一致性

`plot_gprmambasep_diagnostics.py` 现在会优先读取 `eval_full_line.py` 写出的：

```text
*_pred_centerline.csv
```

并使用其中的：

```text
dp_center_sample
mean_center_sample
gt_center_sample
```

来画 DP/mean/GT 线。这样诊断图与 metrics CSV 使用的是同一条实际解码路径，不再用 `*_path_softmask.npy` 的 center-of-mass 近似 DP 路径。
