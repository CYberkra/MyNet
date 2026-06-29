# RuntimeRoot 集中环境配置说明

RuntimeRoot 用于把长期运行环境集中到一个可管理目录中，避免每个软件版本都携带或重建 gprMax、Python 和 PyCUDA。

## 推荐目录

```text
D:\UavGPR_Runtime\
├─ miniconda3\                  # 脚本安装或定位的 Miniconda
├─ conda_envs\
│  └─ uavgpr_gprmax_py310_gpu\  # UavGPR/gprMax/PyCUDA 运行环境
├─ downloads\                   # 安装包缓存
├─ logs\                        # setup 和验证日志
└─ uavgpr_runtime.env            # 当前激活运行配置
```

## gprMax 管理

gprMax 不再随 UavGPR-SimLab 发布包发送。推荐固定在：

```text
D:\UavGPR_Runtime\gprMax\gprMax-v.3.1.7
```

也可以通过 `-GprMaxDir` 指向其他长期目录。

## 配置命令

```powershell
.\setup_uavgpr_gpu_runtime_windows.bat -RuntimeRoot "D:\UavGPR_Runtime" -GprMaxDir "你的gprMax源码目录" -ForceRecreateEnv
```

脚本会写入：

```text
D:\UavGPR_Runtime\uavgpr_runtime.env
项目根目录\.simlab_env
```

新版本软件启动时优先读取这些配置，不再依赖系统 PATH 中的 Python 或 conda。
