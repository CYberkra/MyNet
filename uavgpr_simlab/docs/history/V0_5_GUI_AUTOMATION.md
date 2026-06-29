# UavGPR-SimLab v0.5 GUI 自动化重构说明

## 目标

v0.5 将界面从“功能页堆叠”重构为“项目工作台”式流程，让用户打开软件后按顺序完成：

1. 环境检查
2. 仿真计划生成
3. 3D/2.5D 模型预览
4. 运行前预检与去重
5. 批量运行
6. 历史记录复盘/删除
7. 实测弱监督与结果报告

核心原则是：**可读、自动、可恢复、可删除、可追溯**。

## 新增/重构标签页

### 0 工作台

提供清晰入口卡片：环境检查、仿真计划、3D 预览、预检去重、批量运行、历史记录。新用户不需要记住复杂脚本顺序。

### 3 3D预览

从 `labels.json` 读取：

- 地表曲线
- 基覆界面曲线
- UAV 飞行高度轨迹
- case_id、界面均深、坡度等元数据

该页面在正式 FDTD 前就能做模型质量检查，避免长时间跑完才发现几何异常。

### 4 预检去重

根据 manifest 生成任务表，自动显示：

- 总任务数
- 待运行数量
- 将跳过数量
- 每个任务的 job_id、fingerprint、原因和输入文件

去重依据：`.in` 文件内容 SHA256 + variant + trace 数。成功任务会写入 `jobs/done/*.json`，再次运行会自动跳过。

### 5 批量运行

保留原实时队列能力，同时默认启用：

- 跳过已完成
- geometry-only 预检
- 实时日志
- B-scan 实时预览
- 成功后写 done marker
- 失败后可在历史页复盘

### 6 历史记录

扫描当前 workspace 下的 `jobs/done` 和 `jobs/failed`，显示全部历史仿真：

| 字段 | 说明 |
|---|---|
| 状态 | done / failed |
| 时间 | 完成或失败记录时间 |
| case_id | 场景编号 |
| variant | raw / target_only / clutter_only 等 |
| n | trace 数 |
| geometry | 是否 geometry-only |
| return | 返回码 |
| job_id | 去重任务 ID |
| input | 输入 .in 文件 |
| marker | 历史记录 JSON |

支持操作：

- 刷新历史
- 导出 CSV
- 删除选中历史记录
- 彻底删除选中记录及输出目录

删除操作限制在当前 workspace 内，避免误删外部文件。

## 新增 CLI

```bash
PYTHONPATH=src python -m uavgpr_simlab.cli history \
  --workspace workspace/<project> \
  --limit 50

PYTHONPATH=src python -m uavgpr_simlab.cli history \
  --workspace workspace/<project> \
  --export-csv workspace/<project>/reports/simulation_history.csv

PYTHONPATH=src python -m uavgpr_simlab.cli delete-history \
  --workspace workspace/<project> \
  --marker-file workspace/<project>/jobs/done/<job_id>.json \
  --delete-outputs
```

## 推荐使用方式

### 本机 3060/4090 小批量

1. 打开 GUI：`run_gui.bat`
2. 进入 `1 环境检查`，保存 gprMax 环境路径。
3. 进入 `2 仿真计划`，先生成 1-3 个 case。
4. 进入 `3 3D预览`，检查模型是否合理。
5. 进入 `4 预检去重`，确认待运行数量。
6. 进入 `5 批量运行`，先勾选 geometry-only 跑预检。
7. 取消 geometry-only，正式运行。
8. 进入 `6 历史记录` 查看、导出或删除历史。

### HPC/SLURM

仍可使用 v0.4 的安全 SLURM 脚本生成方式；v0.5 历史页可复盘本地同步回来的 `jobs/done` / `jobs/failed` 记录。

## 主要新增文件

```text
src/uavgpr_simlab/core/history.py
src/uavgpr_simlab/gui/main_window.py  # 大幅重构
```

## 自测结果

在当前容器中完成：

```bash
PYTHONPATH=src python -m uavgpr_simlab.cli --help
PYTHONPATH=src python -m uavgpr_simlab.cli generate --plan configs/run_plan_3060_quick.yaml --workspace /tmp/testws --count 2
PYTHONPATH=src python -m uavgpr_simlab.cli plan-jobs --manifest ... --workspace ... --variants raw,target_only
PYTHONPATH=src python -m uavgpr_simlab.cli history --workspace ...
python -m compileall -q src scripts
```

当前容器没有 PySide6、CUDA 和 gprMax，因此没有实际打开 GUI 或运行 FDTD；GUI 代码已通过 Python 语法编译，真实运行请在已安装 PySide6/gprMax 的 Windows 或工作站环境中执行。
