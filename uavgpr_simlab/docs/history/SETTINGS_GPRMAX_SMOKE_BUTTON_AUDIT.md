# 设置页 gprMax 最小 CPU 测试按钮接入审计

## 目标

将上一轮已验证的 gprMax 源码最小 CPU smoke test 接入产品化易用界面的“设置与帮助”页，使用户在填写 gprMax 源码目录后，可以从 GUI 内直接执行极小 A-scan CPU 验证。

## 本轮修改

- 新增 `src/uavgpr_simlab/services/gprmax_smoke_service.py`：
  - `run_gprmax_source_smoke()`：执行源码结构检查、`python -m gprMax --help`、极小 CPU A-scan、HDF5 `.out` 检查，并写出 JSON 报告。
  - `format_gprmax_source_smoke_report()`：生成设置页可读的中文摘要。
  - `write_tiny_cpu_input()`：集中维护极小 smoke 输入文件。
- 修改 `scripts/smoke_gprmax_source.py`：
  - 从独立实现改为 CLI 包装脚本，复用服务层逻辑。
- 修改 `src/uavgpr_simlab/gui/pages/settings_page.py`：
  - 新增“最小 CPU 测试”按钮。
  - 页面构建器仍只负责 widget 构建，不直接运行 gprMax。
- 修改 `src/uavgpr_simlab/gui/easy_window.py`：
  - 新增 `run_easy_gprmax_source_smoke()` 回调。
  - 读取当前设置页的 gprMax 源码目录、OpenMP 线程和工作目录。
  - 运行结果显示在设置页日志区，包含中文摘要和原始 JSON。

## 架构边界

该按钮属于环境诊断辅助，不属于正式批量仿真入口。

不会改变：

- `core/runner.py` 的 gprMax 调用语义；
- 批量任务构造；
- fingerprint；
- done / failed marker；
- B-scan 后处理；
- GPU 参数传递；
- conda run 语义。

GUI 设置页默认不自动执行 `python setup.py build_ext --inplace`，避免无提示修改用户提供的 gprMax 源码树。需要构建时仍通过 CLI 脚本追加 `--build`，或由用户在目标 conda 环境中手动安装/构建。

## 验证

已完成：

```bash
python -m compileall -q src scripts
PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/self_test.py
PYTHONPATH=src python scripts/smoke_gprmax_source.py --gprmax-root /mnt/data/_gprmax_v317_read/gprMax-v.3.1.7 --work-dir workspace/gprmax_source_smoke_v0713 --omp-threads 1 --timeout 180
```

实际 gprMax 源码 smoke 结果：

```text
ok: true
gprMax source tree: true
detected gprMax version: 3.1.6
compiled extensions: 11/11
HDF5 output: tiny_Ascan_2D.out
Iterations: 23
rxs: rx1
```

## 风险

- P0：未发现。
- P1：该按钮只证明当前 Python + 本地源码的最小 CPU 求解，不能证明 Windows + CUDA + pycuda + GPU 批量求解。
- P2：按钮回调目前同步执行，极小测试通常很快；如果未来扩展为长测试，应改成 `QThread`/worker，避免阻塞 GUI。
- P3：真实目标机失败信息可继续补充更细的中文错误映射。
