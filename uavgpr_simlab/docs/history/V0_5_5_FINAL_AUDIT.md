# UavGPR-SimLab v0.5.5 最终审计报告

## 审计结论

v0.5.5 是在 v0.5.4 行级审计版基础上的最终审计修复版。本轮采用“发现问题 -> 修复 -> 重新全量测试 -> 再审计”的方式进行。第一轮最终审计发现 4 个需要修复的问题；修复后进行第二轮复测，未再发现阻断批量仿真主流程的问题。

当前可审计范围内，以下链路通过：

- 代码编译与导入
- GUI 离屏启动与 10 个标签页加载
- 数据集生成、labels、manifest、3D 模型预览
- 预检去重、done/failed/running/stale_running 记录
- 历史页模型缩略图与 B-scan 缩略图
- 真实 gprMax CPU smoke 调用
- 自动跳过已完成任务
- 真实生成模型的 gprMax geometry-only 校验
- SLURM/local 脚本生成与 CPU-only no-gpu 参数传递
- 删除历史记录的 workspace 边界保护
- 后处理导出 NPZ/CSV/PNG 与传统基线

## 本轮发现并修复的问题

### 1. 论文主仿真计划中的高介电水体可能导致 gprMax 数值色散硬错误

现象：用真实 gprMax 对 `run_plan_paper_main_500.yaml` 生成的第一条 raw 模型做 geometry-only 检查时，gprMax 报错：

```text
Non-physical wave propagation: Material 'surface_water' has wavelength sampled by 2 cells,
less than required minimum for physical wave propagation.
```

原因：默认 5 cm 网格下，`surface_water eps_r=65-82` 在 100 MHz Ricker 波形的显著频带内不满足 gprMax 最小波长采样要求。真实淡水 eps_r 约 80 在 5 cm 网格大批量场景中需要更细网格，计算量不适合 paper-main 批量生成。

修复：

- 将 `surface_water` 默认范围改为 FDTD-safe surrogate：`eps_r=24-32`。
- 将 `saturated_zone` 默认范围收敛到 `eps_r=18-32`。
- 同步更新 `configs/materials_uavgpr_default.yaml` 的高含水/水体模板。
- 在源码中加入注释：若需要真实 eps_r~80 水体，必须降低 dx 到约 1-2 cm，并只建议用于小窗口 3D/high-fidelity case。

复测：重新生成 paper-main 第一条 raw 模型后，真实 gprMax geometry-only 通过。

### 2. CPU-only SLURM 脚本虽然不申请 GPU，但 safe runner 仍会传递 gprMax `-gpu`

现象：v0.5.4 对 `#SBATCH --gres=gpu:0` 做了修复，但 safe runner 命令仍未带 `--no-gpu`，导致 `run-one` 默认会使用 GPU。

修复：

- `core/hpc.py` 新增 safe runner 的 `use_gpu` 参数。
- `hpc-script` CLI 新增 `--no-gpu`。
- SLURM `gpus_per_task=0` 或显式 `--no-gpu` 时，脚本同时满足：
  - 不写 `#SBATCH --gres=gpu:*`
  - run-one 命令带 `--no-gpu`
  - direct non-safe runner 不拼接 `-gpu`
- pipeline 的 hpc 配置支持 `no_gpu: true`。

复测：生成 CPU-only SLURM 脚本，检查到 `# GPU disabled for this job` 且命令包含 `--no-gpu`。

### 3. 历史删除接口在传入 marker_file 时先读取 marker，再检查删除边界

现象：删除函数的删除动作已经限制在 workspace 内，但如果 CLI 传入外部 marker_file，旧逻辑会先读取该外部文件。

修复：

- `delete_history_record()` 在读取 marker 前先验证 marker 路径是否位于 workspace 内。
- 外部 marker 直接抛出 `PermissionError`。

复测：传入 workspace 外的 marker 文件，函数返回 `PermissionError`，不会读取或删除外部文件。

### 4. GUI 批量运行页“运行选中任务”可能因 variant 过滤导致选中行与实际任务不一致

现象：任务列表展示所有 variant，但旧逻辑按当前 variant 下拉框重新筛选 manifest，再用行号定位任务；如果用户选中的行不是当前 variant，可能跑错任务。

修复：

- `load_manifest()` 对每个 QListWidgetItem 保存完整 manifest row 到 `Qt.UserRole`。
- “运行选中任务”直接从选中 item 的 row data 构造 `GprMaxTask`，不再依赖当前 variant 下拉框和行号。
- “运行前 N 个任务”仍保留按 variant 批量筛选的行为。

复测：GUI 离屏深度测试通过，任务页和历史页正常加载。

## 最终复测清单

```bash
python -m compileall -q src scripts
PYTHONPATH=src python scripts/self_test.py
QT_QPA_PLATFORM=offscreen PYTHONPATH=src python scripts/gui_deep_smoke_test.py
PYTHONPATH=src python -m uavgpr_simlab.cli pipeline --config configs/pipeline_paper_simulation.yaml
PYTHONPATH=src python -m uavgpr_simlab.cli run-one \
  --input-file <paper_main_first_raw.in> \
  --workspace /mnt/data/v055_clean_geom_ws \
  --case-id gengeom \
  --variant raw \
  --n-traces 1 \
  --gprmax-root /mnt/data/gprmax_src/gprMax-v.3.1.7 \
  --no-conda-run \
  --no-gpu \
  --geometry-only
PYTHONPATH=src python -m uavgpr_simlab.cli hpc-script \
  --mode slurm \
  --manifest workspace/paper_main_simulation/datasets/paper_main_simulation_manifest.csv \
  --workspace workspace/paper_main_simulation \
  --out-sh workspace/paper_main_simulation/logs/cpu_test.sh \
  --variants raw \
  --gpus-per-task 0 \
  --no-gpu \
  --gpu-ids "" \
  --max-tasks 1
```

结果：上述检查均通过。

## 仍需用户真实环境确认的项目

当前容器没有 NVIDIA 驱动、CUDA、nvcc，因此没有验证 4090 GPU 性能、显存占用和 PyCUDA/gprMax GPU kernel 编译。CPU 版真实 gprMax 调用、生成模型 geometry-only、历史记录、后处理和 GUI 离屏流程已经通过。建议在 Windows/4090 环境里最后跑：

1. 1 个 case，raw variant，`--geometry-only`。
2. 1 个 case，raw variant，`n_traces=3-5` full simulation。
3. 打开 GUI 历史记录页确认 running/done B-scan 实时预览。
4. 再扩大到 48 case validation。

## 最终建议版本

请使用：`UavGPR-SimLab_v0.5_5_final_audited.zip`。

不要再使用 v0.5.4 及更早版本，因为 v0.5.4 的默认水体材料在 paper-main 5 cm 网格下可能触发 gprMax 非物理波传播错误。
