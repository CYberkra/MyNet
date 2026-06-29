# UavGPR-SimLab v0.3 自动化优化说明

本次优化围绕论文方案中仍需人工参与的三个环节展开：完整流水线封装、钻孔弱监督 soft mask、HPC/3D 仿真调度与论文结果自动汇报。

## 1. 新增命令

### 1.1 一键流水线

```bash
PYTHONPATH=src python -m uavgpr_simlab.cli pipeline --config configs/pipeline_automation_template.yaml
```

默认流程会自动完成：

1. 读取 run plan 并生成 gprMax `.in`、label、interface、mask 与 manifest；
2. 生成 Windows BAT 批处理脚本；
3. 生成 SLURM array 脚本与任务 TSV；
4. 转换实测 CSV，输出 NPZ/PNG/QC 报告；
5. 根据钻孔表生成弱监督 soft mask；
6. 汇总 manifest、QC、soft mask 与文件产物，生成 Markdown 自动报告和论文表格 CSV。

安全起见，流水线不会直接启动昂贵的 gprMax 求解器，只生成可执行脚本和报告。正式运行前请人工确认 `.in` 几何和 HPC 参数。

### 1.2 钻孔 weak label soft mask

```bash
PYTHONPATH=src python -m uavgpr_simlab.cli soft-mask \
  --bscan-npz workspace/paper_automation_demo/real_data/line9_qc/real_uavgpr_bscan_preview.npz \
  --boreholes configs/boreholes_example.csv \
  --out workspace/paper_automation_demo/real_data/line9_soft_mask \
  --line-id Line9 \
  --velocity-m-per-ns 0.10 \
  --trace-sigma 2.5 \
  --time-sigma-ns 20
```

钻孔 CSV 支持字段：`line_id,borehole_id,trace_index,distance_m,x_m,depth_m,uncertainty_m,weight`。其中 `trace_index` 可直接给出；也可以使用 `distance_m` 或 `x_m`，程序会结合 `trace_interval_m` 转换为 trace 位置。

输出：

- `borehole_soft_mask.npy`：训练可直接读取的 mask；
- `borehole_soft_mask.npz`：包含 mask 与元数据；
- `borehole_soft_mask_overlay.png`：mask 与 B-scan 的叠加检查图；
- `borehole_picks_used.csv`：每个钻孔点转换后的 trace/time/sample；
- `soft_mask_report.json`：自动报告读取的摘要。

注意：soft mask 是弱监督/保护带，不应作为逐像素 clean 真值。

### 1.3 HPC / SLURM 调度脚本

```bash
PYTHONPATH=src python -m uavgpr_simlab.cli hpc-script \
  --manifest workspace/paper_automation_demo/datasets/paper_automation_demo_manifest.csv \
  --out-sh workspace/paper_automation_demo/logs/run_gprmax_slurm_array.sh \
  --variants raw,target_only,clutter_only,background_only,air_only \
  --partition gpu \
  --gpus-per-task 1 \
  --cpus-per-task 4 \
  --array-parallelism 4 \
  --postprocess
```

该命令会生成：

- `run_gprmax_slurm_array.sh`：SLURM array 作业脚本；
- `run_gprmax_slurm_array.tasks.tsv`：从 manifest 提取出的任务列表。

若只需本地 Linux/macOS 顺序运行，可添加 `--mode local`。

### 1.4 自动论文报告

```bash
PYTHONPATH=src python -m uavgpr_simlab.cli auto-report --workspace workspace/paper_automation_demo
```

输出：

- `reports/auto_report.md`：实验进度、QC、soft mask、文件产物与待复核事项；
- `reports/auto_report.json`：机器可读版；
- `paper/tables/manifest_summary.csv`；
- `paper/tables/real_qc_summary.csv`；
- `paper/tables/soft_mask_summary.csv`。

## 2. 推荐论文工作流

1. 运行 `pipeline_automation_template.yaml` 的小规模示例；
2. 检查 `models/case_xxxxxx/*.in`、`datasets/*_interface.csv`、`real_data/*/*.png` 和 `soft_mask_overlay.png`；
3. 将 `count` 提高到 pilot/main 规模，并按 2D、2.5D、3D/Hard-case 分批执行；
4. 用 SLURM/BAT 脚本跑 gprMax；
5. 运行 `auto-report` 生成论文表格；
6. 后续 PGDA-CSNet 训练脚本读取 manifest、B-scan 产品和 soft mask。

## 3. 自测

```bash
PYTHONPATH=src python scripts/self_test.py
```

自测现在会在无 PySide6 的代码审查环境中跳过 GUI 启动，但仍测试核心仿真输入生成、实测 CSV 转换、soft mask、gprMax HDF5 后处理、SLURM 脚本和自动报告生成。
