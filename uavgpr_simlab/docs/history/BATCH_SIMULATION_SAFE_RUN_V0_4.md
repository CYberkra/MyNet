# UavGPR-SimLab v0.4 批量仿真安全运行说明

## 1. 本版解决的问题

v0.3 可以批量生成 gprMax `.in` 和 SLURM/BAT 脚本，但默认运行脚本再次执行时，会重新调用同一个 `.in`。v0.4 加入了“任务注册表 + 输入指纹 + 完成标记”的安全队列机制：

- 每个仿真任务根据 `input_file` 文件内容、`n_traces`、`variant` 计算 SHA256 指纹。
- 成功运行后在 `workspace/jobs/done/<job_id>.json` 写入完成标记。
- 之后再次运行同一批任务时，若指纹一致且已有完成标记，会自动跳过。
- 修改 `.in` 或改变 `n_traces/variant` 后，指纹会变化，不会误跳过。
- `geometry-only` 预检任务使用独立完成标记，不会导致后续正式 FDTD 任务被误跳过。
- 使用 `--force` 或 YAML 中 `force: true` 可主动重跑。

## 2. 推荐批量流程

### 第一步：生成模型和 manifest

```bash
PYTHONPATH=src python -m uavgpr_simlab.cli generate \
  --plan configs/run_plan_3060_quick.yaml \
  --workspace workspace/uavgpr_batch_test \
  --count 10
```

生成结果示例：

```text
workspace/uavgpr_batch_test/<plan_name>/
  models/case_000001/raw.in
  models/case_000001/target_only.in
  datasets/<plan_name>_manifest.csv
  logs/run_all_gprmax.bat
```

### 第二步：先做任务计划检查

```bash
PYTHONPATH=src python -m uavgpr_simlab.cli plan-jobs \
  --manifest workspace/uavgpr_batch_test/<plan_name>/datasets/<plan_name>_manifest.csv \
  --workspace workspace/uavgpr_batch_test/<plan_name> \
  --variants raw,target_only,clutter_only,background_only,air_only \
  --out-csv workspace/uavgpr_batch_test/<plan_name>/jobs/job_plan.csv
```

查看：

```text
workspace/uavgpr_batch_test/<plan_name>/jobs/job_plan.csv
workspace/uavgpr_batch_test/<plan_name>/jobs/job_plan.summary.json
```

`status=pending` 表示需要运行；`status=skipped` 表示已有同指纹完成标记，会跳过。

### 第三步：本机顺序批量运行

Linux/macOS：

```bash
PYTHONPATH=src python -m uavgpr_simlab.cli hpc-script \
  --mode local \
  --manifest workspace/uavgpr_batch_test/<plan_name>/datasets/<plan_name>_manifest.csv \
  --workspace workspace/uavgpr_batch_test/<plan_name> \
  --out-sh workspace/uavgpr_batch_test/<plan_name>/logs/run_gprmax_local_safe.sh \
  --variants raw,target_only,clutter_only,background_only,air_only \
  --gpu-ids 0 \
  --postprocess

bash workspace/uavgpr_batch_test/<plan_name>/logs/run_gprmax_local_safe.sh
```

Windows：生成命令时默认写出的 BAT 已改为安全 runner：

```bat
workspace\uavgpr_batch_test\<plan_name>\logs\run_all_gprmax.bat
```

它内部调用：

```bat
python -m uavgpr_simlab.cli run-one ...
```

因此重跑 BAT 时会先检查 `workspace\jobs\done`，不会重复跑同一模型。

### 第四步：HPC / SLURM 批量运行

```bash
PYTHONPATH=src python -m uavgpr_simlab.cli hpc-script \
  --mode slurm \
  --manifest workspace/uavgpr_batch_test/<plan_name>/datasets/<plan_name>_manifest.csv \
  --workspace workspace/uavgpr_batch_test/<plan_name> \
  --out-sh workspace/uavgpr_batch_test/<plan_name>/logs/run_gprmax_slurm_safe.sh \
  --variants raw,target_only,clutter_only,background_only,air_only \
  --partition gpu \
  --gpus-per-task 1 \
  --cpus-per-task 4 \
  --mem 24G \
  --array-parallelism 4 \
  --gpu-ids 0 \
  --postprocess

sbatch workspace/uavgpr_batch_test/<plan_name>/logs/run_gprmax_slurm_safe.sh
```

生成的 SLURM 脚本默认使用 `run-one` 安全封装。若任务已完成，脚本会打印 `[SKIP]` 或返回 `status=skipped`，不会重新调用 FDTD。

## 3. 需要重跑怎么办？

有三种方式：

1. 修改 run plan 或 `.in`，重新生成后指纹会变化，会被识别为新任务。
2. 命令行加 `--force`：

```bash
PYTHONPATH=src python -m uavgpr_simlab.cli hpc-script ... --force
```

3. 删除对应完成标记：

```text
workspace/<plan_name>/jobs/done/<job_id>.json
```

## 4. GUI 中怎么用

队列页新增：

- `跳过已完成`：默认勾选。再次运行同一批任务时会跳过完成任务。
- `强制重跑`：默认不勾选。只有确认要覆盖重跑时再打开。

推荐 GUI 流程：

1. `1 环境/GPU`：检查 conda/gprMax/GPU。
2. `3 动态生成`：选择 run plan，先生成少量 case。
3. `4 队列/实时`：先勾选 `geometry-only` 跑 1-3 个任务。
4. 几何确认后取消 `geometry-only`，保持 `跳过已完成`，批量运行。
5. 中断后再次运行同一批任务，已完成的会自动跳过。

## 5. 注意事项

- 安全跳过以 `jobs/done/*.json` 作为主依据，不单纯依赖 `.out` 文件是否存在。
- 如果手工删除 `.out` 但没有删除 done 标记，系统仍会认为已完成。需要重跑时删除标记或使用 `--force`。
- gprMax 对 B-scan 会生成多道 `.out`，后处理/合并由后续流程处理；v0.4 的跳过机制不以文件名猜测作为唯一判断依据。
