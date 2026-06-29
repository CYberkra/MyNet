# UavGPR-SimLab v0.5.4 行级深度审计报告

审计对象：`UavGPR-SimLab_v0.5_3_real_gprmax_test` 代码基线  
审计产物：`UavGPR-SimLab_v0.5_4_line_audited`  
审计范围：`src/` 与 `scripts/` 下全部 Python 源码、`configs/*.yaml`、批量仿真/历史记录/GUI 关键路径、真实 gprMax CPU smoke 测试链路。

## 1. 行级审计覆盖

本轮对源码逐文件、逐函数、逐关键分支进行了人工+脚本联合审计，并生成了源码行级清单：

- `docs/audit/source_line_manifest_v0_5_4.csv`：每一行源码的文件名、行号、行哈希、是否空行/注释、文本摘要。
- `docs/audit/source_file_summary_v0_5_4.json`：每个源码文件的行数、非空行、注释行、AST 解析状态。

统计结果：

| 项目 | 数量 |
|---|---:|
| Python 源码文件 | 27 |
| Python 源码总行数 | 5328 |
| 覆盖模块 | CLI、core、GUI、scripts |
| YAML 配置 | 全部通过 `yaml.safe_load` |
| AST/compileall | 全部通过 |

## 2. 已执行的检查

| 检查项 | 结果 | 说明 |
|---|---|---|
| `python -m compileall -q src scripts` | 通过 | 所有 Python 文件语法可编译 |
| `scripts/self_test.py` | 通过 | GUI import、模型生成、CSV、soft mask、HDF5 后处理、HPC/report 全通过 |
| `scripts/gui_deep_smoke_test.py` | 通过 | 离屏 GUI、3D 预览、历史 B-scan 预览、预检页全通过 |
| `pipeline_paper_simulation.yaml` | 通过 | 生成 500 case、2500 个任务、BAT/SLURM/report |
| `plan-jobs` | 通过 | 正确统计 pending/skipped |
| `run-one` 真实 gprMax CPU smoke | 通过 | 调用用户上传的 gprMax 源码，生成 `.out` 并完成 postprocess |
| 重复运行去重 | 通过 | 第二次 `run-one` 返回 skipped |
| 失败后重跑成功 | 通过 | 旧 failed marker 被清理，仅保留 done marker |
| 删除安全边界 | 通过 | 拒绝删除 workspace 外部路径和 workspace 根目录 |
| stale running 判定 | 通过 | 同机死 PID 判 stale；异机近期任务不误判 |

## 3. 发现并修复的问题

### P0-1：pipeline 环境检查会因 `rep.ok` 缺失崩溃

位置：`src/uavgpr_simlab/core/pipeline.py` 调用 `rep.ok`，但 `EnvironmentReport` 原本没有 `ok` 属性。

修复：在 `EnvironmentReport` 增加 `ok` property，并写入 `to_dict()`。

影响：启用 `environment.enabled: true` 的 pipeline 现在可正常执行并记录整体状态。

### P0-2：删除历史记录存在路径前缀误判风险

位置：`src/uavgpr_simlab/core/history.py::_safe_remove_path`

原逻辑使用字符串 `startswith` 判断是否位于 workspace 内。例如 `/tmp/ws_evil` 会被错误视为 `/tmp/ws` 的子路径。

修复：改为 `Path.resolve().relative_to()`，并拒绝删除 workspace 根目录。

影响：历史页“彻底删除选中记录+输出”更安全。

### P1-1：跨 HPC 节点时 running marker 可能误判 stale

位置：`src/uavgpr_simlab/core/history.py::_resolved_status`、`job_registry.py`、`gui/main_window.py`

原逻辑只看 PID。本地 GUI/登录节点无法看到计算节点 PID 时，可能把真实运行中的任务误标为 stale。

修复：running marker 增加 `host`、`supervisor_pid`；只有同 host 才做 PID 存活检查。跨 host 任务按 marker 年龄判断，默认 stale 阈值提升到 24 小时。

影响：历史仿真页对本地/集群混合运行更稳。

### P1-2：失败任务重跑成功后，旧 failed marker 可能残留

位置：`src/uavgpr_simlab/core/job_registry.py`、`src/uavgpr_simlab/gui/main_window.py`

修复：成功写入 done marker 后清理同 job_id 的 failed marker。

影响：历史页不再同时显示同一任务的旧 failed 和新 done。

### P1-3：设置 `gprmax_root` 后，相对 `.in` 路径可能失效

位置：`src/uavgpr_simlab/core/runner.py::options_from_config_task`

真实 gprMax 测试时发现：当 cwd 切到 gprMax 源码目录，命令中的相对 input 文件路径会找不到。

修复：构建 gprMax 命令前将 `task.input_file` 统一解析为绝对路径。

影响：CLI、GUI、SLURM/local 安全 runner 均受益。

### P1-4：模型长度小于移动测线长度时，后续 trace 可能越界

位置：`src/uavgpr_simlab/core/scenario.py::make_scene_geometry`

修复：自动计算 `source_x + tx_rx_offset + (trace_count - 1) * trace_step + margin`，必要时扩展模型长度和 domain_x。

影响：批量生成的 `.in` 更不容易出现源/接收器走出计算域的问题。

### P2-1：pipeline workspace 路径重写存在歧义

位置：`src/uavgpr_simlab/core/pipeline.py::_project_out_path`

原逻辑对 `workspace/logs/x.sh` 会错误丢弃 `logs`。

修复：区分 `workspace/old_project/logs/x.sh` 和 `workspace/logs/x.sh` 两类模板路径。

影响：配置模板切换 workspace 时，BAT/SLURM/report 输出位置更稳定。

### P2-2：soft-mask pipeline 参数可能因 YAML 字符串触发类型错误

位置：`src/uavgpr_simlab/core/pipeline.py`

修复：增加 `_optional_float()`，对 `trace_interval_m` 和 `time_sigma_ns` 做显式转换。

### P2-3：CPU-only SLURM 配置会写出 `#SBATCH --gres=gpu:0`

位置：`src/uavgpr_simlab/core/hpc.py`

修复：`gpus_per_task <= 0` 时不写 GPU gres 指令。

## 4. 真实 gprMax 复测摘要

使用用户提供的 `gprMax-v.3.1.7.zip` 解压源码作为 gprMax root，当前源码内部版本显示为 `3.1.6 Big Smoke`。本轮执行 CPU smoke：

```bash
PYTHONPATH=src:/mnt/data/gprmax_src/gprMax-v.3.1.7 \
python -m uavgpr_simlab.cli run-one \
  --input-file sample_data/gprmax_smoke_Ascan_2D.in \
  --workspace workspace/real_gprmax_line_audit \
  --case-id smoke \
  --variant raw \
  --n-traces 1 \
  --no-conda-run \
  --python-executable python \
  --gprmax-root /mnt/data/gprmax_src/gprMax-v.3.1.7 \
  --no-gpu \
  --postprocess
```

结果：

- gprMax 返回码：0
- `.out` 可读：是
- postprocess 产物：raw、dewow、mean_subtract、gain、svd_clean、svd_clutter、fk_clean、fk_clutter
- 第二次重复执行：正确返回 `skipped`

当前容器仍无 NVIDIA 驱动/CUDA/nvcc，因此 GPU 性能与 `-gpu` 实际求解未验证；CPU gprMax 与 UavGPR-SimLab 包装链路已验证。

## 5. 剩余注意事项

1. 论文级大批量仿真前仍建议按 smoke → validation → 500 case → 3000 case 分层推进。
2. 历史页实时 B-scan 依赖 gprMax 在运行期间释放可读 `.out` 文件；本地真实 smoke 已验证可读，但不同 HPC 文件系统可能存在延迟。
3. 当前工程仍是仿真/自动化/GUI 工具链，完整 PGDA-CSNet 训练闭环仍应作为后续独立开发任务。
4. Windows BAT 中路径带空格的情况已由命令拼接部分处理，但建议项目路径尽量不含中文空格，以减少外部 gprMax/conda 工具链兼容风险。

## 6. 审计结论

v0.5.4 已修复本轮行级审计发现的关键问题。就“批量仿真模型生成、去重、安全运行、实时历史预览、后处理导出、自动报告”这一主流程而言，当前代码可以进入小规模真实仿真 smoke 与 validation 阶段。
