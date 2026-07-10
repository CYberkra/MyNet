# PGDA-CSNet 审计修复报告

修复基线：`1818b25`  
修复分支：`audit-fixes-20260710`  
日期：2026-07-10

## 结论

已完成代码侧和配置侧可执行修复。项目现在会主动阻止数据合同不完整、仿真 loader 缺失、Line9 条件数据污染、无真实负样本、错误飞高先验和 split 污染的正式训练；curve distribution 与二维 mask 的评估语义已经拆分。

仓库当前仍保持正式训练冻结。这不是代码故障，而是数据本体仍缺少 `lines/*.npz`、`window_index.csv`、独立于 Line9 的获批仿真、真实负样本和可信逐 trace 飞高字段。修复没有伪造这些数据。

## 已完成修复

### 1. 配置治理

- 44 个 JSON 配置均可解析并具有显式 `run_type`。
- X1/LineX1 被固定为 review-only，不允许进入 train/validation/test。
- formal 配置要求独立 validation；validation/test 重叠被阻止。
- 未实际被 trainer 使用的 `paper_split_file` 字段被移除或禁止。
- 全部现有训练配置显式冻结，并记录 `training_block_reason`。

### 2. 训练与数据合同守卫

- `sim_batch_ratio > 0` 时，仿真目录、索引、样本或 loader 缺失将立即失败。
- Line9-conditioned 仿真不能进入 Line9 strict holdout。
- 同一源 trace、重叠窗口和冲突 split 会在训练前被拒绝。
- presence/no-target 训练要求存在真实负样本；弱标签不会伪装成确定负类。
- arrival prior 要求可靠高度字段或显式 missing-height 策略，不再静默使用 2.4 m。
- shallow suppression 对存在合法浅层目标的 trace 自动跳过。

### 3. 评估语义

- mask sigmoid：仅用于 Dice、IoU、BCE 等二维分割指标。
- curve distribution：仅用于 NLL、期望/argmax 路径误差及 DP 路径指标。
- center map：独立保存和评估。
- presence：独立计算分类指标，并尊重 `presence_valid`。
- 不再默认把 curve distribution 保存为 `pred_softmask.npy`；旧名仅通过显式兼容开关输出。

### 4. 数据治理

建立 `data/dataset_contract_v2/`：

- `dataset_manifest.json`
- `simulation_cases.csv`
- `human_audit_manifest.csv`
- `real_lines.csv`
- `real_windows.csv`
- `split_manifest.csv`
- `label_semantics.json`

当前登记：

- 仿真 case：33
- Line9-conditioned case：33
- accepted 隔离 case：13
- 人工审计记录：39
- 正式训练获批仿真：0
- 完整登记实测 lines/windows：0/0

所有自动 GREEN、目录名和历史 accepted 状态均不能自动授予训练资格。

### 5. accepted 晋级与仿真导出

- accepted 晋级必须有 case-local raw、label、scene/design provenance 和人工决策。
- 禁止从模板复制标签或元数据形成伪追溯。
- 自动 QC grade 不再等同于人工晋级。
- 仿真训练 NPZ 只允许从 `train_allowed=true` 的 manifest 行导出；当前 0 个获批 case 时会显式失败。

### 6. 工程基础

- 新增根级 `pytest.ini`，避免误收集运行脚本和大型工作目录。
- 新增 `requirements.txt` 与 `requirements-dev.txt`。
- 新增项目合同校验脚本和数据合同构建/仿真导出工具。

## 验证结果

- 治理、训练合同、损失和评估语义测试：`28 passed`。
- 模型接口测试：`15 passed`。
- 修复涉及的 Python 文件：`py_compile` 通过。
- 项目合同普通模式：通过，唯一警告为正式训练冻结。
- 项目合同 `--require-formal-ready`：预期失败，证明冻结门禁生效。

`tests/test_gprmambasep.py` 在当前 CPU 执行环境中耗时过长，分组运行超时；超时前已有多个测试通过，没有得到失败断言。此前审计环境记录的完整核心测试为 `110 passed, 1 warning`，本轮不能用当前 CPU 超时替代该结果，也未虚报全量通过。

## 正式训练解除冻结条件

1. 补齐并登记 `lines/*.npz` 与 `window_index.csv`。
2. 建立按测线/源 trace 分组且无重叠的 split manifest。
3. 生成与 Line9 标签和曲线先验无关的仿真数据，并完成可信人工晋级。
4. 增加人工确认的 true-negative 实测或仿真样本。
5. 统一 visible-phase 标签语义并重建 V4。
6. 补充可信逐 trace 飞高/地形字段，或关闭实测 arrival prior。
7. 将 `dataset_manifest.json` 的 `formal_training_allowed` 改为 true 前，运行：

```bash
python scripts/validate_project_contracts.py --require-formal-ready
```

只有该命令通过后，才允许启动正式多 seed 论文训练。
