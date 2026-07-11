# gprMax V2 与项目静态测试结果

- 四个控制场景静态 preflight：**4/4 通过**。
- 完整测试套件：**138 passed，0 failed，0 errors，19.04 s**。
- V15-final：84/84 个受保护 NPZ SHA256 未变化。
- 正式 split：LineL1/Line3/Line7=train，Line6=val，Line9=test，LineX1=exclude。
- 旧仿真：33/33 继续 `legacy_quarantine`、`formal_training_allowed=false`。
- gprMax/FDTD：未运行；环境没有 gprMax。

完整命令和机器可读结果见 `STATIC_TEST_RESULTS.json`。

- CPU 线程限制：完整测试使用 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`，避免容器线程过度订阅。
- 项目合同：从历史 `MyNet.zip` 临时物化 66 个隔离旧仿真的 raw/label 资产后通过；这些大文件不进入精简结果源码包。
