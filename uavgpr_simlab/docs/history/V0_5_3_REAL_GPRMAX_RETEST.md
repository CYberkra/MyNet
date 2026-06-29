# UavGPR-SimLab v0.5.3 real gprMax retest

## 测试环境

- gprMax source: `/mnt/data/gprmax_src/gprMax-v.3.1.7`
- Python: 3.13.5
- GUI: PySide6 6.11.1, offscreen mode
- CPU only: 当前容器没有 NVIDIA driver / CUDA / nvcc，因此 GPU 模式未测试。

## gprMax 安装/构建

用户上传的 `gprMax-v.3.1.7.zip` 已解压并构建 Cython 扩展。由于当前容器使用 Python 3.13 且只有 4 GiB RAM，为了完成 smoke test，将 gprMax `setup.py` 中 Linux 编译参数临时从 `-O3 -march=native` 调整为 `-O0`。这只影响本容器测试速度，不建议作为正式 Windows/4090 环境设置。

验证命令：

```bash
PYTHONPATH=/mnt/data/gprmax_src/gprMax-v.3.1.7 python -m gprMax --help
```

结果：gprMax CLI 可正常启动，支持 `-n`, `--geometry-only`, `--geometry-fixed`, `--write-processed`, `-gpu` 等参数。

## 实测结果

### 1. gprMax 最小 FDTD smoke

输入：`/mnt/data/tiny_gprmax.in`

结果：成功生成 `/mnt/data/tiny_gprmax.out`，return code 0。

### 2. UavGPR `run-one` 真实调用 gprMax

命令通过 `uavgpr_simlab.cli run-one` 调用真实 gprMax，完成后写入 done marker。

结果：

- `status = done`
- `returncode = 0`
- 生成 gprMax `.out`
- `history-preview` 可读取完成任务并渲染 B-scan 预览

### 3. 真实 gprMax 运行中的实时 B-scan

启动 20 trace 的真实 gprMax B-scan 任务，在任务仍处于 running 时调用：

```bash
PYTHONPATH=src:/mnt/data/gprmax_src/gprMax-v.3.1.7 \
python -m uavgpr_simlab.cli history-preview \
  --workspace /mnt/data/uavgpr_live_real2 \
  --status running \
  --limit 5 \
  --time-window-ns 20
```

结果：

- 识别到 `running` 任务 1 条
- 当时已有 5 道 `.out` 可读
- 实时合成 B-scan 形状为 `425x5`
- 成功生成 `*_bscan.png`

这说明“历史仿真页实时显示正在跑的每一道输出并合成 B-scan”的核心机制可以和真实 gprMax 输出配合工作。

### 4. GUI 离屏深度测试

命令：

```bash
QT_QPA_PLATFORM=offscreen \
PYTHONPATH=src:/mnt/data/gprmax_src/gprMax-v.3.1.7 \
python scripts/gui_deep_smoke_test.py
```

结果：通过。覆盖：

- 10 个主标签页存在
- 3D 预览页可绘制模型
- 预检去重页可生成任务表
- 历史记录页可显示模型缩略图
- 历史记录页可显示 done B-scan
- 历史记录页可显示 running B-scan

## 本次发现并修复的边界问题

测试中发现：如果外部强制杀掉 `run-one` 包装进程，旧版本可能留下 `jobs/running/*.json`，历史页会误认为任务仍在运行。

v0.5.3 修复：

- running marker 写入 `supervisor_pid`
- 历史扫描时检查 supervisor/pid 是否仍存在
- 若进程不存在，则显示为 `stale_running`
- CLI 与 GUI 均支持筛选 `stale_running`

这不会影响正常 done/failed 逻辑，只增强异常中断后的可读性。

## 仍需在你的 Windows/4090 环境确认

- GPU 模式 `-gpu 0` 的真实运行速度和显存占用
- CUDA/PyCUDA 是否正确安装
- 大模型 500/3000 case 的实际吞吐
- Windows 文件锁定情况下 `.out` 是否总能边跑边读；当前 Linux 容器中可以边跑边读。
