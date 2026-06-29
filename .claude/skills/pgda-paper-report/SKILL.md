---
name: pgda-paper-report
description: Turn PGDA-CSNet experiment metrics, reports, checkpoints, and preview outputs into a concise Chinese paper or group-meeting style report.
---

# PGDA Paper Report

Use this skill when the user asks to summarize experiment results, prepare a paper-method report, make a group meeting update, or decide whether a model/result should be promoted.

## Inputs to look for

- `reports/CURRENT_MODEL_STATE.md`
- experiment report Markdown files
- metrics CSV/JSON files
- checkpoint paths and hashes
- evaluation output folders
- preview images or QC figures
- config JSON for the run

## Required checks

1. Identify the experiment:
   - version name
   - checkpoint/run directory
   - data root
   - train/validation/test split
2. Compare to the frozen baseline when relevant:
   - frozen v1.9D seed-1902
   - default valid-line average
   - Line9 holdout result
3. Check promotion discipline:
   - LineX1 must be review-only
   - do not promote by training-line performance alone
   - distinguish internal Line9 holdout from strict zero-material cross-line validation
4. Summarize metrics:
   - MAE ns
   - pick rate / coverage
   - false pick or rejection behavior
   - confidence/abstention if present
5. Classify the result:
   - baseline
   - promoted candidate
   - ablation only
   - negative result
   - diagnostic only
6. State limitations honestly.

## Output format

Use Chinese. Prefer this structure:

```markdown
## 实验结论

一句话结论。

## 关键结果

| 项目 | 数值/结论 |
|---|---:|

## 和 frozen v1.9D 的关系

## 是否建议提升为主模型

## 风险与限制

## 下一步
```

Keep the report concise unless the user asks for a full paper-style draft.
