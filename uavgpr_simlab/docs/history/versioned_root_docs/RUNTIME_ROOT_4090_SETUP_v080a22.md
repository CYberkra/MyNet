# v0.8.0-alpha.22：外部 gprMax 持久运行环境说明

## 目标

从本版开始，UavGPR-SimLab 发布包不再内置 `gprMax-v.3.1.7.zip`。gprMax 源码、Miniconda、conda 环境、PyCUDA、日志和持久配置应放在长期 RuntimeRoot 或用户指定目录中，后续软件版本只读取该位置。

推荐目录：

```text
D:\UavGPR_Runtime\
├─ miniconda3\
├─ conda_envs\gprMax\
├─ gprMax\gprMax-v.3.1.7\
├─ downloads\
├─ logs\
└─ uavgpr_runtime.env
```

## 推荐命令

如果 gprMax 已放在 RuntimeRoot：

```bat
setup_gprmax_4090_windows.bat -RuntimeRoot "D:\UavGPR_Runtime" -GprMaxDir "D:\UavGPR_Runtime\gprMax\gprMax-v.3.1.7" -ForceRecreateEnv
```

如果 gprMax 放在其他目录，把 `-GprMaxDir` 改成你的实际路径：

```bat
setup_gprmax_4090_windows.bat -RuntimeRoot "D:\UavGPR_Runtime" -GprMaxDir "D:\Your\Path\gprMax-v.3.1.7" -ForceRecreateEnv
```

完成后运行：

```bat
scripts\Verify_4090_GPRMAX_GPU.bat
run_gui.bat
```

## 规则

- 默认不再从发布包查找 gprMax zip。
- 默认不自动联网 clone gprMax；未找到源码会直接报错并提示 `-GprMaxDir`。
- 如果仍有本地 zip，可显式传入 `-GprMaxZip "D:\path\gprMax-v.3.1.7.zip"`。
- 只有显式传入 `-AllowCloneGprMax` 时才会联网 clone。
- 运行成功后会写入 `D:\UavGPR_Runtime\uavgpr_runtime.env`，后续版本复用该文件。

## 验收标准

`logs\check_4090_gprmax_gpu_report.json` 中应为：

```json
"ok": true
```

批量运行日志中不应出现：

```text
python=E:\python\python.exe
```

应指向 RuntimeRoot 环境或显式 conda prefix。
