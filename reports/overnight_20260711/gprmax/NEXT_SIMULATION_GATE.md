# 下一仿真门禁

1. 在隔离环境安装官方 gprMax 3.1.7，并记录 Python/compiler/CUDA/driver。
2. 跑官方最小 smoke test。
3. 对 CTRL01–04 执行 `--geometry-only`，人工核对天线、地层、PML 和 matched pair。
4. 只运行 CTRL01 的单道/少量道；核对 HDF5 属性和理论到时。
5. CTRL01 通过后才运行完整 256 道。
6. CTRL01 运行/相位/边界门禁全部通过后，再依次运行 CTRL02、CTRL03、CTRL04。
7. CTRL04 只有在实际输出通过并人工确认后，才是可晋级的仿真真负候选；今晚仍不放行训练。
8. 不生成 24-case pilot，直到四个控制场景完成正式运行审计。
