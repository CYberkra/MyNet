# Dataset contract V2

本目录是实测与仿真训练数据的唯一治理层。目录名、自动 QC 等级或历史 `accepted` 状态均不构成训练许可。

## 实测数据

6 条 canonical 全测线由原始营山 CSV ZIP 生成。当前标签发布为 `YINGSHAN_V15_FINAL_20260710`：Line9 保持原标签并继续作为 test-only 主锚点；Line3 与 X1 各有一处弱标签重标；Line6 与 X1 各有一处歧义区明确排除监督。V14 标签完整保留用于回滚。

## 测线方向与剖面显示合同

- Canonical 数组、窗口索引、训练和指标始终保持原始 CSV 采集顺序。
- `profile_display_flip` 只能用于可视化/导出。
- 地图轨迹使用 `gnss_cumulative_distance_m`；工程剖面对照使用 `profile_chainage_m`。
- Line3、Line6、Line9、LineX1 的剖面显示相对采集顺序翻转；Line7、LineL1 不翻转。

## 当前放行状态

- 没有确认的实测真负窗口；
- 没有获批的非 Line9-conditioned 正式仿真；
- 正式测线划分已锁定在 `configs/paper_splits_v15_aeropath.json`；
- 历史 V1 与 Batch 3 已由归档分支保存，不再进入当前合同。

V15 最终标签发布完成不等于正式训练放行。只有 `dataset_manifest.json` 明确设置 `formal_training_allowed=true` 且 `python scripts/validate_project_contracts.py --require-formal-ready` 通过后才允许启动正式训练。

## 真负样本审计

`status_code=0` 是唯一可进入实测真负样本人工复核的起点。弱标签、无标签、交叉区 ignore、低信噪比和失败正样本均不代表“没有基覆界面”，不得自动转换为负类。

对任何候选发布先运行：

```powershell
python scripts\audit_real_negative_candidates.py `
  --data-root <checksum-verified-v15-release> `
  --output-dir reports\<task_id>\real_negative_audit
```

该命令只生成候选与阻断结论，不修改 V15，也不会放行训练。只有地质人员确认候选窗口确实不含目标界面、生成新的版本化发布和 manifest 后，才可更新正式训练门。

2026-07-22 checksum-verified V15 audit result: all six canonical lines contain
zero explicit `status_code=0` traces. The 310 zero-weight regions are V15
ignore intervals on Line3, Line6, and LineX1, so they remain ineligible for
negative supervision. The real-negative blocker is therefore a data fact, not
a configuration issue.
