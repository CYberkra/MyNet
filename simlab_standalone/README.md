# SimLab Standalone

独立 gprMax 仿真运行环境。

## 环境

- **Python**: `E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe`
- **gprMax**: v3.1.6, 已安装 PyCUDA (GPU 支持)
- **GPU**: NVIDIA GeForce RTX 3060 Laptop 6 GB
- **MSVC**: `E:\sisual stdio 2022\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64`

## 使用方法

### 几何检查（快速验证 .in 文件）
```bash
E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe run_geometry.py <path/to/scene.in>
```

### GPU 仿真
```bash
E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe run_gpu.py <path/to/scene.in> [n_traces] [output_dir]
```

### 生成 SceneWorld 数据集
```bash
E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe -m uavgpr_simlab.cli generate \
  --plan configs/run_plan_3060_pilot_v1.yaml \
  --workspace workspace/my_run \
  --count 20
```

## 配置文件

主配置: `D:\Claude\PGDA-CSNet\uavgpr_simlab\configs\run_plan_3060_pilot_v1.yaml`
默认配置: `D:\Claude\PGDA-CSNet\uavgpr_simlab\configs\default_app.yaml`
