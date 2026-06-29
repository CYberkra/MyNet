# UavGPR-SimLab v0.5.1 深度审计报告

## 审计结论

v0.5.1 已通过本轮可执行审计。当前代码可支撑“论文所需仿真数据生产”的主流程：生成同源 raw / target_only / clutter_only / background_only / air_only gprMax 输入、保存 labels/interface/mask、生成去重任务计划、生成本机或 SLURM 批处理脚本、运行成功后写入 registry 标记、可从 history 页面或 CLI 复盘历史记录。

需要如实说明：本审计环境没有 CUDA/gprMax/PySide6，因此没有在容器内执行真实 FDTD 求解或打开 GUI。已经验证的是 Python 代码编译、CLI、数据生成、manifest、任务去重、历史记录、pipeline、后处理链和脚本生成逻辑。真实求解器联调需要在你的 Windows/4090/gprMax 环境中完成。

## 本轮发现并修复的问题

1. pipeline 模板中如果用户只修改顶层 workspace，commands/hpc/real_csv/soft_mask/report 的硬编码输出路径仍可能写回旧 workspace。v0.5.1 已新增 `_project_out_path()`，会把 `workspace/old_project/...` 自动重定向到当前激活的 workspace。
2. SLURM/local 脚本原先在提交目录下创建 `logs/`，不一定写入项目 workspace。v0.5.1 已在脚本开始处 `cd <workspace>`，保证日志和相对输出路径在项目内。
3. registry runner 在求解器成功但 postprocess 失败时，原逻辑可能把整次仿真标成 failed。v0.5.1 改为：求解器 returncode=0 即写 done marker；postprocess 失败只记录 `postprocess_error`，用户可后续单独补做后处理。

## 已执行验证

- `python -m compileall -q src scripts`
- `PYTHONPATH=src python scripts/self_test.py`
- `generate --plan configs/run_plan_3060_quick.yaml --count 2`
- `plan-jobs --variants raw,target_only,clutter_only,background_only,air_only`
- 用 `/usr/bin/true` 模拟 gprMax 成功，验证 `run-one` 完成标记与二次运行跳过机制
- `history --workspace ...`
- `hpc-script --mode local` 与 `hpc-script --mode slurm`
- `pipeline --config ...`，确认所有输出都落在激活 workspace 内
- 自定义 gprMax input 静态检查：domain/material/source/receiver/box/cylinder 均未发现越界或未知材料

## 推荐正式仿真策略

论文正式仿真不要直接一次跑 500 个 case。建议分三层：

1. smoke：2-3 个 case，geometry-only 检查几何。
2. validation：48 个 case，完整 raw/target/clutter/background/air，确认输出、后处理和分解标签。
3. formal：500 个 case 起步，按 100 或 200 case 分批提交；稳定后扩大到 3000+ case。

## 仍需人工确认

- gprMax/CUDA/驱动/PyCUDA 是否在目标机安装完成。
- 4090 显存和单 case 运行时间是否满足当前 dx、trace_count、time_window。
- 正式论文材料参数是否需要用钻孔、FWI 或样品电性进一步收窄。
- 若使用 HPC，partition 名、CUDA module、conda 初始化方式需按集群实际配置调整。
