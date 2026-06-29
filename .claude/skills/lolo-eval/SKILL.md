---
name: lolo-eval
description: 一键 LOLO-CV 集成评估——自动找到 3-seed run dirs、运行 ensemble 评估、对比 baseline。Use when user says "评估 Line9", "run LOLO eval", "集成评估".
---
# lolo-eval: LOLO-CV 集成评估

## 使用方式
```
/lolo-eval Line9
/lolo-eval Line3 --baseline p0-3
```

## 参数
- `line`: 要评估的测线名称（Line3/Line6/Line7/Line9/LineL1）
- `--baseline`: 对比的 baseline 名称（默认 p0-3）

## 评估流程

### Step 1: 定位 run directories
自动搜索匹配的输出目录：
```bash
ls outputs/ | grep "loo_${line}_seed"
```
期望找到 3 个：seed1901, seed1902, seed1903

验证每个目录下 `checkpoint_last.pt` 或 `checkpoint_best.pt` 存在。

### Step 2: 检查训练完整性
读取每个 seed 的 checkpoint epoch：
```python
torch.load('checkpoint_best.pt', weights_only=False)['epoch']
```
如果任一 seed epoch < 80，警告该 seed 训练不完整。

### Step 3: 运行 Ensemble 评估
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" scripts/eval_full_line.py \
  --line <line> \
  --run-dirs outputs/run_gpu_v3_pilot_mixed_loo_<line>_seed1901 \
             outputs/run_gpu_v3_pilot_mixed_loo_<line>_seed1902 \
             outputs/run_gpu_v3_pilot_mixed_loo_<line>_seed1903 \
  --out-dir outputs/eval_v3_pilot_mixed_loo_<line>_3seed_ensemble \
  --dp-breakable --center-fusion-weight 1.0
```

### Step 4: 解析结果
读取 `<out-dir>/<line>_full_metrics.csv`，提取关键指标：
- DP Center MAE
- Pick Rate
- Mean Center MAE
- IoU@0.2 / IoU@0.3 / IoU@0.5

### Step 5: 与 Baseline 对比
当前已知 baseline (P0-3):
- MAE: 3.268
- Pick Rate: 0.562

输出对比表：
```
📊 LOLO-CV 评估结果: Line9
━━━━━━━━━━━━━━━━━━━━━━━━━
指标          P0-3 Baseline  LOLO 3-Seed    增益
DP Center MAE    3.268        X.XXX       ↓ XX%
Pick Rate        0.562        X.XXX       ↑ XX%
Mean Center MAE  -            X.XXX       -
IoU@0.3          -            X.XXX       -
```

### Step 6: 生成可视化（可选）
如果有 preview 目录，展示评估 B-scan 图像。
