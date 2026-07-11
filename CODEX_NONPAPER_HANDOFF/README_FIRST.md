# Codex 验收入口：d6f9108 之后的非论文工程增量

## 1. 基线与范围

- 用户指定基线交付物：`MyNet_V15正式数据治理增量包_d6f9108.zip`。
- 本验收分支基于最终 Git bundle 中的重建基线提交 `5535eff`；该提交名为 `reconstructed d6f9108 baseline from compact delivery`。
- 原始 `d6f9108` Git 对象不在最终 bundle 内，因此不要把 `5535eff` 当成原始提交哈希。外层交付同时保留原始 d6f9108 ZIP，供文件级对照。
- 本包仅包含数据治理完成之后的非论文工程工作：旧仿真审计、gprMax V2 物理合同、控制场景、生成/验证/运行/后处理脚本、静态测试与门禁。

## 2. 明确排除

本分支故意不包含本轮论文形态工作：

- `pgdacsnet/models/experimental/interface_query_net.py`
- `tests/test_interface_query_net.py`
- `reports/overnight_20260711/network_research/`
- 架构评分、文献地图、投稿故事线、论文命名和候选网络结论

这些内容不应影响本次工程验收。

## 3. 当前真实状态

- 四个 gprMax V2 控制场景静态校验：4/4 通过。
- 旧 33 个 Line9-conditioned 仿真：继续隔离，禁止正式训练。
- V2 控制场景：`formal_training_allowed=false`。
- 当前环境没有实际 gprMax/CUDA 求解结果；geometry-only、FDTD、HDF5、相位和 PML 运行验证均未完成。
- CTRL04 只是“匹配无目标控制候选”，不是已经批准的正式负样本。

## 4. Codex 建议执行顺序

```bash
python -m compileall -q pgdacsnet scripts tests
python scripts/validate_physical_sim_v2.py

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
pytest -q tests/test_simulation_v2.py

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
pytest -q tests/test_simulation_v2_overnight_contract.py

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
pytest -q tests/test_gprmambasep.py
```

完整项目测试还需要另行放入 V15-final 数据和历史 quarantine 资产；本精简包不重复携带这些大文件。

## 5. 重点验收问题

1. solver 时窗与 0–700 ns/501 点训练时轴是否严格分离。
2. PML、guard、域尺寸、源/接收器移动范围是否无越界。
3. 平层双基地分层到时与曲面 columnar reference 的语义是否准确。
4. full/no-basal/air-reference 是否是匹配控制，而非伪造 component truth。
5. CTRL02/CTRL04 的匹配关系是否逐数组成立。
6. HDF5 `dt`、`Iterations`、接收器长度和时间覆盖是否会被严格检查。
7. 负样本导出是否强制零 mask、零 curve/presence 语义。
8. Line9 泄漏扫描、legacy quarantine 和 formal-training gate 是否不可绕过。
9. 未安装 gprMax 时脚本是否明确失败，而非生成伪运行结果。
