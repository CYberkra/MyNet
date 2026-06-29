# 4090 Windows 一键配置说明

入口脚本：`setup_gprmax_4090_windows.bat`

脚本会尝试执行：

1. 查找 Miniconda/conda；找不到时尝试 winget 安装 Miniconda。
2. 检查 `nvidia-smi`、`nvcc`。
3. 检查或安装 Git。
4. 尝试安装 Visual Studio 2022 Build Tools C++ 工作负载。
5. 克隆或更新 gprMax 到 `%USERPROFILE%\gprMax`。
6. 使用 gprMax 官方 `conda_env.yml` 创建/更新 `gprMax` 环境。
7. `python setup.py build` 与 `python setup.py install`。
8. 安装 GUI 依赖、editable 安装 UavGPR-SimLab、安装 PyCUDA。
9. 运行 CPU/GPU smoke test。

日志位置：`logs/setup_4090_windows.log`

Windows 上 PyCUDA/CUDA/Visual Studio Build Tools 的组合经常受系统状态影响。若脚本中断，先看日志最后 80 行，再检查 CUDA Toolkit、NVIDIA Driver、Build Tools 是否匹配。
