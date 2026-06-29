---
name: sim-batch
description: 批量 gprMax 仿真启动——场景生成、仿真执行、自动合并、进度监控。Use when user says "批量仿真", "batch simulation", "跑N个场景", "批量跑仿真".
---
# sim-batch: 批量仿真启动

## 使用方式
```
/sim-batch --count 50 --variant raw,target_only
/sim-batch --manifest workspace/xxx_manifest.csv
/sim-batch --plan configs/run_plan.yaml --workspace workspace/my_run
```

## 参数
- `--count`: 生成场景数
- `--variant`: 仿真变体（raw, target_only, air_only）
- `--manifest`: 现有 manifest CSV 路径
- `--plan`: SimLab plan YAML
- `--workspace`: 工作目录
- `--dry-run`: 只生成不运行

## 流程

### Step 1: GPU 检查
- 检查 GPU 空闲（无训练进程）
- 检查显存可用
- 检查温度（如 >85°C 提示等待）

### Step 2: 准备仿真
**方式 A** (SimLab CLI):
```bash
python -m uavgpr_simlab.cli generate --plan <plan> --workspace <ws> --count <N>
```

**方式 B** (手动 .in 文件):
```bash
python scripts/run_batch_safe_3060.py --manifest <manifest> --variants <variants>
```

### Step 3: 执行仿真
使用 `run-gprmax-sim` skill 的底层逻辑逐场景运行。每个场景：
- 运行 gprMax GPU 仿真
- 合并多道输出
- 清理 .vti 文件
- 记录耗时和 GPU 状态

### Step 4: 进度监控
每完成 N 个场景输出进度：
```
🔄 仿真进度: 35/50 (70%)
   已完成: 35 场景, 平均 45s/场景
   预计剩余: ~12 分钟
   GPU: 78°C, 2100 MiB
```

### Step 5: 完成汇总
- 总场景数、成功/失败数
- 总耗时、平均单场景耗时
- QC 检查结果
