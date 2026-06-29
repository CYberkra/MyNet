# Workspace 迁移与路径重定位

目标：一台电脑设计数据集骨架，另一台电脑导入后继续运行，不因为 manifest、QC、history 中残留旧电脑绝对路径而失败。

推荐原则：

```text
软件版本目录可以换；
D:\UavGPR_Runtime 长期保留；
workspace 内部路径尽量相对化；
运行前先检查/修复路径，再做 dataset contract 和 GPU runtime 验证。
```

## 命令行用法

先 dry-run：

```powershell
python -m uavgpr_simlab.cli relocate-workspace --manifest "workspace\<dataset>\datasets\<dataset>_manifest.csv"
```

如果报告中显示可自动修复，再写入：

```powershell
python -m uavgpr_simlab.cli relocate-workspace --manifest "workspace\<dataset>\datasets\<dataset>_manifest.csv" --apply
```

如果知道旧电脑的数据集根目录，可显式传入：

```powershell
python -m uavgpr_simlab.cli relocate-workspace `
  --manifest "workspace\<dataset>\datasets\<dataset>_manifest.csv" `
  --old-root "E:\old_project\workspace\<dataset>" `
  --apply
```

也可以直接运行脚本：

```powershell
python scripts\check_workspace_relocation.py --manifest "workspace\<dataset>\datasets\<dataset>_manifest.csv" --apply
```

## GUI 用法

批量仿真页新增：

```text
迁移/修复路径
```

使用顺序：

```text
导入数据集骨架 → 迁移/修复路径 → 预检任务 → 开始运行统一任务
```

按钮会先 dry-run，确认能自动修复后才写入；写入前会把被修改文件备份到：

```text
workspace\<dataset>\reports\relocation_backups\<timestamp>\
```

## 修复范围

自动处理：

- manifest 中的路径列；
- `models/**/*.json`；
- `jobs/**/*.json`；
- `configs/*.yaml/yml`；
- `logs/*.bat` 中可判定为 workspace 内部的旧路径。

不会处理：

- `gprMax` 源码目录，例如 `E:\gprMax\...`；
- CUDA、Miniconda、PyCUDA、Visual Studio Build Tools 路径；
- `reports/*.json` 派生诊断报告，这些报告可重新生成。

## 输出报告

默认写入：

```text
workspace\<dataset>\reports\workspace_relocation_report.json
```

字段包括：

- `absolute_path_count`
- `change_count`
- `changed_file_count`
- `findings`
- `changes`
- `dataset_contract_ok`
- `run_dashboard_json`

## 推荐跨电脑流程

```text
电脑 A：设计骨架 / 小规模验证
复制 workspace\<dataset> 到电脑 B
电脑 B：relocate-workspace --apply
电脑 B：check-dataset-skeleton
电脑 B：Verify_Current_GPU_Runtime
电脑 B：GUI 批量页一键运行
```
