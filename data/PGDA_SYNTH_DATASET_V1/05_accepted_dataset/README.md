# 05_accepted_dataset/ — 正式可训练数据

## 用途

只有通过 QC（GREEN_ACCEPTED）的 case 才复制到这里。直接从 `accepted_dataset/` 导出训练版本。

## 结构

```
{family}/
├── {subtype}/
│   ├── {CASE_ID}/
│   │   ├── input/
│   │   │   └── raw_bscan.npy
│   │   ├── label/
│   │   │   ├── interface_mask_bscan.npy
│   │   │   ├── interface_mask_wide_bscan.npy
│   │   │   ├── y_soft_501x128.npy
│   │   │   ├── target_visible_phase_time_ns.npy
│   │   │   └── target_geom_time_ns.npy
│   │   ├── metadata/
│   │   │   ├── scene_world.json
│   │   │   ├── design_metrics.csv
│   │   │   └── qc_report.json
│   │   └── preview/
│   │       ├── qc_target_zoom.png
│   │       └── geometry_preview.png
│   └── ...
├── accepted_manifest.csv
└── README.md
```

## 训练输入原则

- 训练主输入：`raw_bscan.npy`（原始，不预处理 t²/t⁴/AGC）
- 预处理在训练 pipeline 中完成，不在数据层面做
- 不接受没有 QC 记录的 case

## 当前分类

| 子目录 | 说明 |
|--------|------|
| line9_style/flat/ | 平地表 Line9 风格 |
| line9_style/terrain/ | 起伏地表 Line9 风格 |
| line9_style/mixed/ | 混合地表 Line9 风格 |
| generic_smooth/ | 通用平滑模型 |
| weak_cover/ | 弱覆盖层模型 |
| shallow_perturbation/ | 浅层扰动模型 |
