# Simulation Data Audit Manifest

## Audit Info
- **Date**: 2026-07-10
- **Auditor**: 人工 (PGDA-CSNet project lead)
- **Method**: 逐 case 查看几何模型图 + 灰度三面板 B-scan (Raw/Dewow/TimeGain) + 标签覆盖

---

## Usable — 已上传至仓库 (05_accepted_dataset/)

| Case | 深度 | Family | 标签 | 备注 |
|------|:---:|--------|:----:|------|
| LINE9_STYLE_V1_001 | — | flat | usable | |
| LINE9_STYLE_001~010 | 6.6~12.4m | mixed | usable | |
| LINE9_TERRAIN_011~012 | ~17.5m | terrain | usable | 起伏地形 |

## Usable — 已上传 (outputs/v4_quick_test/)

| Case | 深度 | Family | 标签 | 备注 |
|------|:---:|--------|:----:|------|
| V4 Quick Test | 7.0m | flat | usable | V4 low-loss params, SNR=25.9dB |

## Inconclusive — 未上传本体

以下 case 人工判定为"存疑"：B-scan 数据存在但界面微弱或深度较大，
**未上传至仓库**（03_runs/* 在 .gitignore 中排除）。

| Case | 深度 | 界面 SNR | 未上传原因 |
|------|:---:|:--------:|:----------|
| Batch 1 (12 cases) | 6.6~17.5m | -31~-40 dB | 界面微弱，需复判 |
| Batch 3 (20 cases) | 7.9~14.6m | -30~-40 dB | 界面微弱，需复判 |
| Pilot Validation case_000001 | 6.9m | -27 dB (FK) | 勉强可见，但变体全部NaN |
| Pilot Validation case_000003 | 13.2m | -31 dB (FK) | 界面微弱 |

## Rejected — 未上传本体

以下 case 人工判定为"不可用"：界面完全不可见，
**未上传至仓库**。

| Case | 深度 | 界面 SNR | 原因 |
|------|:---:|:--------:|:-----|
| Pilot Validation case_000002 | 16.5m | -89 dB | 深度超限 |
| Pilot Validation case_000004 | 15.0m | -94 dB | 深度超限 |
| Pilot Validation case_000005 | 13.9m | -70 dB | 深度超限 |

---

## 说明

1. 所有 `usable` 标注为人工肉眼判定，非自动 QC
2. 存疑和不可用 case 的原始仿真数据因太大未提交
3. 需要复核存疑 case 时，需在本地从 03_runs/ 重新跑 QC
