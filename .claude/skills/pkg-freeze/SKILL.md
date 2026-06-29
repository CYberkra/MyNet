---
name: pkg-freeze
description: 成果冻结打包——收集 checkpoint/metrics/config/图表，生成标准目录结构和 manifest。Use when user says "打包", "freeze", "导出", "成果包".
---
# pkg-freeze: 成果冻结打包

## 使用方式
```
/pkg-freeze v3_pilot_mixed
/pkg-freeze loo_Line9 --include-checkpoints
```

## 流程

### Step 1: 收集产物
从 `outputs/` 和相关目录收集：
- `checkpoint_best.pt` + `checkpoint_last.pt`
- `history.json`
- `*_full_metrics.csv`
- `used_config.json`
- `previews/` 目录
- 生成的图表

### Step 2: 组织目录结构
```
PGDA_CSNet_<version>_<date>/
├── README.md              # 版本说明
├── manifest.json          # 文件清单 + 哈希
├── configs/
│   └── used_config.json
├── checkpoints/
│   ├── checkpoint_best.pt
│   └── checkpoint_last.pt
├── metrics/
│   └── *_full_metrics.csv
├── history/
│   └── history.json
├── previews/
│   └── *.png
└── scripts/
    └── train_raw_only.py  # 训练脚本副本
```

### Step 3: 生成 manifest.json
```json
{
  "version": "v3_pilot_mixed",
  "created": "2026-06-28",
  "config_hash": "sha256:...",
  "metrics": {"mae": 3.268, "pick_rate": 0.562},
  "files": [...]
}
```

### Step 4: 自检
自动调用 `pgda-transfer-validate` skill 验证包完整性。

### Step 5: 压缩
```bash
zip -r PGDA_CSNet_<version>_<date>.zip PGDA_CSNet_<version>_<date>/
```
