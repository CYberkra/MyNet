# GPRMAX_LOCAL_SOURCE_SMOKE_RESULT - v0.8.0-alpha.2

## 结论

当前 sandbox 已使用用户提供的 gprMax 源码完成最小 CPU smoke test。

```text
结果：通过
源码目录：/mnt/data/_gprmax_v317_read/gprMax-v.3.1.7
检测版本：3.1.6
编译扩展：11/11
输出目录：workspace/gprmax_source_smoke_v080a2
```

## 运行命令

```bash
PYTHONPATH=src python scripts/smoke_gprmax_source.py \
  --gprmax-root /mnt/data/_gprmax_v317_read/gprMax-v.3.1.7 \
  --work-dir workspace/gprmax_source_smoke_v080a2 \
  --omp-threads 1 \
  --timeout 180
```

## 验证项目

- `inspect gprMax source`：通过；
- `python -m gprMax --help`：通过；
- `python -m gprMax tiny_Ascan_2D.in -n 1`：通过；
- HDF5 `.out` 检查：通过。

## 输出摘要

```json
{
  "ok": true,
  "compiled_extensions": "11/11",
  "output_file": "workspace/gprmax_source_smoke_v080a2/tiny_Ascan_2D.out",
  "output_size": 13920,
  "iterations": 23,
  "rxs": ["rx1"]
}
```

## 边界说明

该结果证明当前 Linux sandbox、当前 Python 和用户提供的 gprMax 源码可以完成极小 CPU A-scan 求解并生成可读 HDF5 `.out`。它不能替代 Windows + conda + CUDA + pycuda + GPU 目标机验证。
