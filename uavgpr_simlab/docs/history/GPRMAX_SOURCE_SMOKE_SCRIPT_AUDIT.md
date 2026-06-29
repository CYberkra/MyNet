# GPRMAX_SOURCE_SMOKE_SCRIPT_AUDIT

## 本轮目的

用户已把 gprMax 源码作为项目辅助材料提供。本轮新增一个可复用脚本，用于在当前 Python 环境或目标 conda 环境中对 gprMax 源码树执行最小 CPU smoke test。

## 新增脚本

```text
scripts/smoke_gprmax_source.py
```

脚本职责：

1. 检查 gprMax 源码树结构；
2. 可选执行 `python setup.py build_ext --inplace`；
3. 验证 `python -m gprMax --help`；
4. 写入极小 `tiny_Ascan_2D.in`；
5. 执行 CPU 单任务 `python -m gprMax tiny_Ascan_2D.in -n 1`；
6. 检查生成的 HDF5 `.out`；
7. 输出 JSON 报告。

## 边界

该脚本只做最小 CPU 求解验证，不做：

- GPU 调用；
- pycuda 编译；
- Windows conda 环境创建；
- UavGPR-SimLab 批量队列真实运行；
- B-scan 后处理链路验证。

这些仍应按 `docs/GPRMAX_SMOKE_TEST_TEMPLATE.md` 在目标机器执行。

## 对架构的影响

- 不改变 GUI 页面；
- 不改变 gprMax 任务调用语义；
- 不改变 fingerprint / marker；
- 不改变 B-scan 后处理；
- 仅增强诊断能力和可复现验收能力。
