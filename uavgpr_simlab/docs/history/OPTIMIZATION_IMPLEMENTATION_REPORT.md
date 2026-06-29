# UavGPR-SimLab v0.3 优化实施报告

## 已完成优化

### 1. 统一流水线封装

新增：

- `src/uavgpr_simlab/core/pipeline.py`
- CLI 子命令：`pipeline`
- 配置模板：`configs/pipeline_automation_template.yaml`
- 脚本入口：`scripts/Run_Full_Pipeline_Example.bat`、`scripts/run_full_pipeline_example.sh`

能力：自动串联仿真输入生成、Windows BAT、SLURM 脚本、实测 CSV 质控、钻孔 soft mask 和自动报告。默认不直接执行 gprMax 求解器，避免误触发长时间 HPC 任务。

### 2. 钻孔弱监督 soft mask 自动生成

新增：

- `src/uavgpr_simlab/core/softmask.py`
- CLI 子命令：`soft-mask`
- 示例钻孔表：`configs/boreholes_example.csv`

能力：读取钻孔/界面弱标签 CSV 或 JSON，将深度转换为 GPR 双程时间，并在 B-scan 上生成 Gaussian soft mask。输出 `.npy`、`.npz`、叠加检查 PNG、转换后的 picks CSV 和 JSON 报告。

### 3. HPC/3D 批量调度脚本生成

新增：

- `src/uavgpr_simlab/core/hpc.py`
- CLI 子命令：`hpc-script`

能力：从 manifest 自动生成 SLURM array 脚本和任务 TSV，也支持本地 bash 顺序脚本。支持 variants 筛选、GPU/CPU/内存/array 并发参数、geometry-only 和 postprocess 选项。

### 4. 论文结果自动汇报

新增：

- `src/uavgpr_simlab/core/reporting.py`
- CLI 子命令：`auto-report`

能力：自动汇总 manifest、实测 QC、soft mask、文件产物数量和缺失项，生成：

- `reports/auto_report.md`
- `reports/auto_report.json`
- `paper/tables/manifest_summary.csv`
- `paper/tables/real_qc_summary.csv`
- `paper/tables/soft_mask_summary.csv`

### 5. 自测增强

更新：

- `scripts/self_test.py`

能力：在无 PySide6 的代码审查/服务器环境中自动跳过 GUI 启动，但继续测试核心算法和自动化模块。

已通过的自测项：

- 动态 `.in`、labels、manifest 生成；
- 实测 Line9 CSV 解析与 QC 导出；
- 钻孔 soft mask 生成；
- 伪 gprMax HDF5 `.out` 合并和传统基线导出；
- SLURM 脚本生成；
- 自动 Markdown/CSV 报告生成。

## 验证命令

```bash
PYTHONPATH=src python -m uavgpr_simlab.cli --help
PYTHONPATH=src python -m uavgpr_simlab.cli pipeline --config configs/pipeline_automation_template.yaml
PYTHONPATH=src python scripts/self_test.py
PYTHONPATH=src python -m compileall -q src scripts
```

## 仍需人工确认

- 真实 gprMax FDTD 求解器、CUDA、PyCUDA 与集群模块需要在目标 Windows/4090/HPC 环境中验证；本容器没有 CUDA/gprMax，未实际跑 FDTD。
- `velocity_m_per_ns`、钻孔坐标到 trace 的映射、RTM/FWI 下游指标仍需结合实测线元数据校准。
- PGDA-CSNet 主训练代码在当前源码包中仍以配置模板/预留目录为主，尚未在本次优化中补齐完整 PyTorch 训练框架。
