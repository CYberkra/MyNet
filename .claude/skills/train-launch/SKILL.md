---
name: train-launch
description: 一键安全训练启动——验证 config、检查 GPU 冲突、选择正确 Python、启动并重定向日志、监控首个 epoch。Use when user says "开始训练", "launch training", "启动训练", or provides a training config JSON.
---
# train-launch: 安全训练启动器

## 功能
一条命令完成训练启动全流程，避免常见错误。

## 使用方式
```
/train-launch configs/gpu_train_v3_pilot_mixed_loo_Line9_seed1902.json
```

## 启动流程

### Step 1: Config 预检
读取指定的 JSON config，验证必须字段：
- `data_root` 目录存在
- `train_lines` 非空
- `run_dir` 指定
- `seed` 指定
- `epochs` > 0
- `model_arch` 指定
- 如果有 `sim_data_root`，检查路径存在

### Step 2: GPU 可用性
运行 `nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader`
- 温度 > 90°C: 警告并等 30 秒后重检
- 显存占用 > 4GB/6GB: 警告可能有其他进程占用
- 显存占用 > 5GB: 阻止启动，要求先清理

### Step 3: 重复进程检测
用 `wmic process where "name like '%python%'" get ProcessId,CommandLine` 检查是否有正在运行的训练进程。
- 发现重复: 阻止启动，列出已有进程 PID

### Step 4: 确定 run_dir 并检查冲突
如果 `outputs/<run_dir>/checkpoint_last.pt` 已存在：
- 显示当前 epoch 和 loss
- 问用户：从头训练（覆盖）还是 resume？

### Step 5: 选择 Python 解释器
始终使用: `E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe`

### Step 6: 启动训练
```bash
# 如果从头训练:
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -u scripts/train_raw_only.py <config>

# 如果 resume:
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -u scripts/resume_train.py <config>
```

用 Bash 的 `run_in_background: true` 在后台运行。

### Step 7: 监控首个 Epoch
等待 ~15 秒后检查 `outputs/<run_dir>/checkpoint_last.pt` 是否已更新：
- 文件存在且大小 > 1MB → 成功，报告 PID 和预计完成时间
- 文件不存在 → 报错，打印日志尾部
- 进程退出 → 报错，打印 stderr

## 输出格式
```
🚀 训练已启动
  Config: <config_name>
  PID: <pid>
  Run Dir: <run_dir>
  GPU: <gpu_temp>°C, <vram_used>/<vram_total> MiB
  预计完成: ~<minutes> 分钟
  日志: tail -f <log_path>
```
