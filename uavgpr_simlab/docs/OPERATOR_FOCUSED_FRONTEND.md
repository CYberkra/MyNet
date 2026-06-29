# Operator-focused frontend simplification

## Purpose

UavGPR-SimLab is primarily used by one expert operator.  The front page should
therefore prioritize the routine workflow instead of exposing every runtime and
engineering detail by default.

## Current batch-page rule

The visible front page keeps only the routine actions:

```text
导入数据集骨架 → 迁移/修复路径 → 预检 → 一键开始 → 停止
```

The following information remains available, but is hidden by default under
`运行细节/高级诊断`:

- runtime profile selector;
- manifest path;
- variant tag list;
- max task limit;
- skip / failed-only / force-rerun switches;
- runtime profile / GPU / Python / gprMax summary;
- case / variant run queue;
- failure aggregation panel;
- raw task log.

## Design boundary

Do not remove diagnostics.  Move them behind an explicit details control.  This
keeps the interface usable for daily work while preserving enough evidence to
fix gprMax, PyCUDA, CUDA, model and post-processing failures.

## Future UI rule

New front-page controls must satisfy at least one of these conditions:

1. The operator clicks it during a normal run.
2. It summarizes current dataset state.
3. It shows current or recent B-scan output.

Everything else belongs in settings, history, or the advanced diagnostics panel.
