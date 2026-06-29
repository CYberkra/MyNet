---
name: project-status
description: 项目状态快照——当前 baseline、已完成实验、正在运行的训练、待办任务。Use when user says "项目状态", "project status", "现在什么情况", "当前进展".
---
# project-status: 项目状态快照

## 使用方式
```
/project-status
```

## 采集信息

### 1. 当前 Baseline
读取 memory 文件 `p0-3-center-fusion.md` 获取当前最佳 baseline。

### 2. 已完成的实验
扫描 `outputs/` 目录，对每个训练目录检查：
- `checkpoint_last.pt` 存在 → 读取 epoch
- `history.json` 存在 → 读取最终 loss 和最佳 epoch
- `*_full_metrics.csv` 存在 → 读取评估指标

按实验类型分组：
- LOLO-CV: 按 line 分组
- Within-line split: 按 version 分组
- 消融实验: 标注

### 3. 正在运行的训练
```bash
wmic process where "name like '%python%'" get ProcessId,CommandLine
```
识别正在训练的进程，显示：
- Config 名称
- 当前 PID
- 启动时间
- 预估剩余时间

### 4. 待启动的任务
检查 pending tasks 和 plan 文件。

### 5. 数据状态
- 实测数据: `data/营山/` 的测线列表
- 仿真数据: `data/simulation_pretrain_v2/` 的场景数量
- 标注版本: 最新的 data_corrected 版本

## 输出格式
```
📋 PGDA-CSNet 项目状态
━━━━━━━━━━━━━━━━━━━━━━━
📅 2026-06-28

🏆 当前 Baseline: P0-3 Center Fusion
   MAE=3.268, Pick Rate=0.562

📊 实验进度:
  ✅ LOLO Line9 seed1901: 80 epochs (train=0.509)
  🔄 LOLO Line9 seed1902: 运行中 (PID 42520, epoch ~30)
  ⏳ LOLO Line9 seed1903: 待启动
  ⏳ LOLO Line3/6/7/L1 × 3 seeds: 待启动

🖥️ GPU 状态: 84°C, 4021/6144 MiB

📁 数据:
  实测: 6 条线 (L3/L6/L7/L9/L1/X1)
  仿真: 20 Pilot-Mini 场景
  标注: data_corrected_v1_4

📝 下一步:
  1. 等待 seed1902 完成
  2. 启动 seed1903
  3. 3-seed 集成评估 Line9
```
