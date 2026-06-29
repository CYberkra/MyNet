# RuntimeRoot 4090 Setup Notes v0.8.0-alpha.23

本版继续采用外部持久 gprMax 方案，发布包不再携带 `gprMax-v.3.1.7.zip`。

## 修复点

1. Windows 机器没有 `E:` 盘时，PowerShell 不再因候选路径 `E:\...` 抛出 `DriveNotFoundException`。脚本会安静跳过不存在的盘符。
2. Miniconda 下载阶段更稳：先设置 TLS 1.2，再依次尝试 `Invoke-WebRequest`、`curl.exe`、BITS。
3. 如果 RuntimeRoot 下没有 Miniconda，且下载仍失败，脚本默认允许复用已安装的 conda 作为“控制器”。实际 gprMax / PyCUDA 环境仍创建在：

```text
D:\UavGPR_Runtime\conda_envs\gprMax
```

这可以避免把 PyCUDA 装进系统 Python 或用户目录 Python。

## 推荐运行

如果你的 gprMax 已经固定放好，例如：

```text
D:\UavGPR_Runtime\gprMax\gprMax-v.3.1.7
```

运行：

```bat
setup_gprmax_4090_windows.bat -RuntimeRoot "D:\UavGPR_Runtime" -GprMaxDir "D:\UavGPR_Runtime\gprMax\gprMax-v.3.1.7" -ForceRecreateEnv
```

如果你坚持完全不复用任何现有 conda，并希望 Miniconda 下载失败时直接中止：

```bat
setup_gprmax_4090_windows.bat -RuntimeRoot "D:\UavGPR_Runtime" -GprMaxDir "D:\UavGPR_Runtime\gprMax\gprMax-v.3.1.7" -NoExternalCondaFallback
```

## 当前边界

CUDA Toolkit 和 Visual Studio Build Tools 仍是 Windows 系统级组件，不会放进 RuntimeRoot。
