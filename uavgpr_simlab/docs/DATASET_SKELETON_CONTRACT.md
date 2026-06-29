# DATASET_SKELETON_CONTRACT - v0.8.0-alpha.33

UavGPR-SimLab 的长期使用方式是：先设计 ready-to-run 数据集骨架，再导入软件检查并启动 GPU 仿真。

## 最小目录合同

```text
workspace/<dataset>/
├─ datasets/<dataset>_manifest.csv
├─ models/<case_id>/
│  ├─ raw.in
│  ├─ target_only.in
│  ├─ background_only.in
│  ├─ clutter_only.in
│  ├─ air_only.in
│  └─ outputs/
├─ logs/run_all_gprmax.bat
└─ reports/
```

## 标准五变体

```text
raw,target_only,background_only,clutter_only,air_only
```

每个 case 应有完整五变体。运行完成后，软件会从 raw 和 target_only 生成 `clutter_gt_bscan.npy`，用于后续 PGDA-CSNet 数据合同。

## 导入前检查

GUI：批量仿真页 → 导入数据集骨架。

CLI：

```bat
python -m uavgpr_simlab.cli check-dataset-skeleton --manifest "workspace\<dataset>\datasets\<dataset>_manifest.csv" --write-report
python -m uavgpr_simlab.cli run-dashboard --manifest "workspace\<dataset>\datasets\<dataset>_manifest.csv" --write-report
```

检查通过后再启动批量任务。合同不通过时，软件会阻止进入 gprMax，避免 GPU 批量任务成片失败。

## 状态字段

manifest / QC 中的 `bscan_status` 会被解释为：

```text
not_run / pending → 即将运行
running → 正在运行
success / done → 历史完成
failed → 失败待处理
stale_running → 中断待检查
```

这些状态会同时显示在批量页看板、批量表格和历史页树状视图中。
