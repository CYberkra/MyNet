---
name: lolo-cv-full
description: 完整 LOLO-CV 流水线——5折×3种子串行训练+评估+汇总。When all 5 LOLO-CV folds need training and evaluation.
---

# LOLO-CV Full Pipeline

一键执行完整的 Leave-One-Line-Out 交叉验证流水线。

## 使用方式
```
/workflow lolo-cv-full
```

## 参数
通过 `args` 传入 JSON：
```json
{
  "lines": ["Line3", "Line6", "Line7", "Line9", "LineL1"],
  "seeds": [1901, 1902, 1903],
  "config_base": "configs/gpu_train_v3_pilot_mixed.json",
  "epochs": 80
}
```

## 流程

### Phase 1: 配置生成
为每个 line × seed 组合生成 config 文件。
使用 `scripts/make_v3_pilot_mixed_loo_configs.py`（如已存在）。

### Phase 2: 串行训练
按 line 分组，每条线的 3 个 seed 串行训练（避免 GPU 冲突）：
```
For each line in [Line3, Line6, Line7, Line9, LineL1]:
  For each seed in [1901, 1902, 1903]:
    1. 验证 config
    2. 检查 GPU 空闲
    3. 启动训练 (train_raw_only.py)
    4. 等待完成 (监控 checkpoint_last.pt)
    5. 验证训练结果
    6. 继续下一个 seed
  3-seed 训练完成 → 运行该 line 的集成评估
```

### Phase 3: 集成评估
每条线训练完成后，运行 3-seed ensemble 评估：
```bash
python scripts/eval_full_line.py \
  --line <line> \
  --run-dirs <seed1901_dir> <seed1902_dir> <seed1903_dir> \
  --out-dir outputs/eval_loo_<line>_3seed \
  --dp-breakable --center-fusion-weight 1.0
```

### Phase 4: 跨折汇总
收集 5 条线的评估指标，计算：
- 各指标的 mean ± std
- 与 baseline 对比
- 最佳/最差 fold

### Phase 5: 生成报告
使用 `pgda-paper-report` skill 生成中文总结报告。

## 预计时间
5 lines × 3 seeds × 80 epochs × ~7s/epoch = ~140 分钟（串行）

## 中断恢复
如中途 GPU TDR 或其他原因中断，可重新启动 workflow：
- 自动检测已完成的 fold/seed（检查 checkpoint）
- 跳过已完成的，从上次中断处继续
