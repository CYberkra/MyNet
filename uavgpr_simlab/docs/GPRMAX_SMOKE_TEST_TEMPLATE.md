# GPRMAX_SMOKE_TEST_TEMPLATE - UavGPR-SimLab 真实目标机验收模板

> 用途：用于 Windows + conda + CUDA / CPU 目标机上记录真实 gprMax 链路是否可用。该模板不是自动测试脚本；它用于人工验收和问题复现记录。

## 1. 验收基本信息

| 项目 | 记录 |
|---|---|
| 验收日期 |  |
| 验收人员 |  |
| 软件版本 | UavGPR-SimLab v0.7.11 / 显示版本 v0.7 |
| 项目包路径 |  |
| gprMax 源码目录 |  |
| conda 环境名 |  |
| 是否使用 GPU | 是 / 否 |
| GPU ID |  |
| OpenMP 线程 |  |
| Windows 版本 |  |
| NVIDIA 驱动版本 |  |
| CUDA Toolkit 版本 |  |
| Python 版本 |  |
| gprMax 实际版本 |  |

注意：如果 gprMax 源码压缩包名和源码内 `_version.py` 不一致，应以实际运行环境中的 `python -m gprMax` 或源码 `gprMax/_version.py` 为准。

## 2. 前置检查

在项目根目录执行：

```bat
run_gui.bat
```

或在命令行执行：

```bat
set PYTHONPATH=src
python -m uavgpr_simlab.app
```

记录结果：

```text
GUI 是否启动：
默认是否进入产品化易用界面：
窗口标题是否为 v0.7：
报错信息：
```

## 3. 设置页环境检查

在“设置与帮助”页填写：

```text
gprMax 源码目录：
conda 环境名：
GPU ID：
OpenMP 线程：
是否使用 GPU：
是否使用 conda run：
```

点击“保存设置”，再点击“检查环境”。记录：

```text
总体状态：
gprMax 源码结构是否有效：
检测到的 gprMax 源码版本：
Conda command：
Conda env：
gprMax import：
NVIDIA SMI：
CUDA nvcc：
PySide6：
报告路径：
```

若失败，复制设置页诊断文本中的“需要处理的检查项”。

## 4. 命令行基础验证

在目标 conda 环境中执行：

```bat
conda activate gprMax
python -c "import sys; print(sys.executable)"
python -c "import gprMax; print('gprMax import ok')"
python -m gprMax --help
```

记录：

```text
conda activate 是否成功：
gprMax import 是否成功：
python -m gprMax --help 是否成功：
报错信息：
```

## 5. 软件自测

在项目根目录执行：

```bat
set PYTHONPATH=src
set MPLBACKEND=Agg
python scripts\self_test.py
```

记录：

```text
SELF_TEST_REPORT.json 是否生成：
结果是否 ok=true：
失败项：
```

## 6. 最小模型生成验证

在 GUI 中执行：

```text
项目管理 → 设置工作目录 → 设置仿真计划 → 模型数量设为 1 或 2 → 生成模型
模型预览 → 加载模型图库 → 检查缩略图、3D 预览、模型信息
```

记录：

```text
manifest 路径：
生成模型数：
模型预览是否正常：
3D 预览是否正常：
报错信息：
```

## 7. geometry-only / 预检验证

在“批量仿真”页执行：

```text
加入批量仿真 → 预检任务
```

建议先选择极小任务数，例如 1。

记录：

```text
待运行任务数：
已完成自动跳过数：
任务表是否显示 input_file、case_id、variant、status：
job_plan.json 是否生成：
报错信息：
```

## 8. CPU full raw 单任务验证

建议先关闭 GPU，只运行 1 个 raw 任务。

记录：

```text
任务 case_id：
variant：raw
是否生成 .out：
.out 文件数量：
是否生成 B-scan npz/png：
历史页是否出现记录：
首页最近 B-scan 是否刷新：
耗时：
报错信息：
```

## 9. GPU 单任务验证

启用 GPU 后，只运行 1 个 raw 任务。

记录：

```text
GPU ID：
nvidia-smi 是否显示进程：
是否生成 .out：
是否生成 B-scan：
耗时：
相比 CPU 是否加速：
报错信息：
```

## 10. 小批量验证

建议运行 2-3 个模型的 raw 任务，不要一开始跑完整批量。

记录：

```text
任务总数：
成功数：
失败数：
跳过数：
历史记录数：
导出 CSV 是否成功：
重跑是否安全：
删除记录是否只删除预期内容：
```

## 11. 验收结论

| 项目 | 结论 | 说明 |
|---|---|---|
| GUI 启动 | 通过 / 失败 |  |
| 环境检查 | 通过 / 失败 |  |
| gprMax 导入 | 通过 / 失败 |  |
| 模型生成 | 通过 / 失败 |  |
| 预检任务 | 通过 / 失败 |  |
| CPU 单任务 | 通过 / 失败 |  |
| GPU 单任务 | 通过 / 失败 / 未测 |  |
| 小批量 | 通过 / 失败 / 未测 |  |
| B-scan 生成 | 通过 / 失败 |  |
| 历史导出/重跑/删除 | 通过 / 失败 |  |

最终结论：

```text
通过 / 有条件通过 / 不通过
```

未解决问题：

```text
1.
2.
3.
```
