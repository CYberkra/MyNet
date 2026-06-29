# PGDA-CSNet 未来方向总纲

**版本**: v1.0 (2026-06-27)  
**依据**: 已有审计结论 + 代码核查 + 前序锁定方案  
**阅读后置文档**: `docs/PGDA_CSNET_V1_LOCKED_SCHEME.md`, `docs/PGDA_CSNET_V1_EXECUTION_ROADMAP.md`

---

## 1. 当前项目真实状态诊断

### 1.1 它已经是什么
一个**基于 raw B-scan、实测弱监督、单场地验证的基覆界面拾取系统**。

- 核心已跑通：frozen v1.9D 已给出可用结果
- Label pipeline 在同类中算清晰：`soft_mask` + `status_code` + `label_weight` + `ignore_mask`
- Split 纪律严格：Line9 holdout + LineX1 review-only
- 后处理贡献清晰：fusion + breakable DP
- 已积累大量报告、配置、ablation，可以为论文提供素材
- `.claude` 自动化已初步建立

### 1.2 它还没什么（需要诚实面对）

| 方面 | 现状 | 缺口 |
|------|------|------|
| 泛化性 | 一线 holdout 可用，跨线泛化弱 | 不能宣称 zero-shot cross-line 解决 |
| 标签真值 | 弱监督，非严格真值 | 论文必须清楚表述为 surrogate/operational |
| 数据量 | 单一场地 | 不足以做大数据通用模型 |
| 网络复杂度 | v1.9D 已较复杂 | 没有严谨 ablation 证明每个模块的必要性 |
| 传统 baseline 对比 | 缺少系统性对比 | SVD, RPCA, f-k filter 等对照不可缺 |
| 物理解释性 | 原设计含物理分支但未用 | 当前版本物理解释性不高 |

### 1.3 最危险的风险

1. **标签政策漂移**：v1.4 / V16 / V17 之间的不一致是最大的论文风险
2. **跨线泛化缺失**：审稿人一定会追问 LOO 结果
3. **缺少消融深度**：当前结构复杂但缺少"每个模块到底贡献多少"的证据
4. **缺少传统对照**：没有 SVD/RPCA/f-k filter 等 baseline 的定量对比

---

## 2. 已经确定可行、应该坚持的方向

### 2.1 raw-only 输入策略
正确且不应改动。

- 排除了 BG501 / AGC / processed view 等可能加入的隐式先验
- 论文中最干净、最可复现的输入方案

### 2.2 Line9 locked holdout 纪律
必须坚持且永远不动。

- 这是目前项目里最值钱的实验设计
- 一旦松了，后面所有结果都不再可信

### 2.3 mask + center fusion + breakable DP
合理的后处理设计，v1 应锁定。

### 2.4 按测线分割而非随机 patch 分割
正确。对 GPR 数据来说这是唯一不泄漏的策略。

### 2.5 corrected v1.4 作为主数据集
在单场地条件下，已是最好的折中选择。V16/V17 不应替代它。

---

## 3. 高风险但值得保留观察的方向

### 3.1 confidence / abstention 主线 (v1.11)
**风险等级**: 中等  
**值得观察的原因**: 这直接补当前最大的缺口（错误自信层位拾取）  
**观察条件**:
- 能够在 source lines 上通过 LOO 验证 abstention 策略
- MAE-coverage Pareto 能证明 tradeoff 有利

### 3.2 simulation-supported pretraining
**风险等级**: 高  
**值得观察的原因**: 从长期看这是唯一可能解决数据量不足的路径  
**观察条件**:
- sim→real gap 必须先量化而非盲目预训练
- Pilot-Mini 的 QC 必须先通过

### 3.3 两阶段系统（抑杂增强 → 拾取）
**风险等级**: 高  
**值得观察的原因**: 比端到端单网络更易调试和解释  
**观察条件**:
- 等 v1 baseline 论文完成后再说

---

## 4. 应该停止投入的方向

### 4.1 更大 backbone 搜索（停止）
当前瓶颈不是容量，是跨线泛化和标签不确定性。堆参数不解决问题。

### 4.2 V16 / V17 替代主线（停止）
- V16 造成训练/holdout 政策不一致
- V17 在 locked split 下不能带来有意义改善
- 投入更多资源到这些标签上无意义

### 4.3 Line9 oversampling / line-balanced 路线（停止）
short trial 已出现浅层系统偏置。不应扩到 80 epochs。

### 4.4 robust normalization 作为默认方案（停止）
改善 Line9 但损害五线平均，只做 ablation。

### 4.5 direct terrain / flight metadata 拼接或 FiLM（停止）
v1.8/v1.10 已显示不仅不改善，甚至更差。对 v1 这条路线不成立。

### 4.6 f-k / SVD 等物理分支主线化（暂缓至 v3）
每个都很贵，当前数据条件下收益不清晰。应压缩到 v3 评估。

---

## 5. 代码与工程治理建议

### 5.1 当前工程问题

| 问题 | 严重程度 | 建议 |
|------|---------|------|
| 旧路径残留 | 低（已修） | 定期扫 |
| 大量历史配置混在 configs 中 | 中 | 可清理到 history 子目录 |
| 大量历史报告（324+ 份）混在 reports 中 | 中 | 标记核心报告 vs 历史报告 |
| checkpoint 无统一管理 | 中 | 应建立 checkpoint registry |
| 源码无 git | 低长期 | 可考虑重新初始化 |
| check_configs.py 已重写但未验证所有历史配置 | 低 | 可运行一次 full check |

### 5.2 建议的工程改进（仅对当前最有价值的）

1. **outputs 分类**: core / ablation / diagnostic / smoke 分开存放
2. **checkpoint registry**: 一个 JSON 记录所有重要 checkpoint 及其 hash/指标/来源
3. **清理旧配置**: 将不用的历史实验配置移到 `configs/_archive/`
4. **核心脚本做 `py_compile` 钩子**：已配置，验证通过

### 5.3 不建议做的工程改进（现在不用折腾）

- 不要大规模重构项目结构
- 不要建立统一测试框架（当前阶段测试成本高于收益）
- 不要轻易移动冻结包

---

## 6. 未来 1 个月路线

### 第 1-2 周：固化基线

- [ ] 完整复现 frozen v1.9D Line9 holdout 全量结果（1664–2377）
- [ ] 记录 full metrics + preview + centerline
- [ ] 固化 checkpoint hash / 参数 / 后处理配置

### 第 3-4 周：启动 v1.11

- [ ] 生成 multiple inference views（default + robust + flip）
- [ ] 提取 confidence features
- [ ] 在 source lines 上测试 abstention 规则
- [ ] 输出 MAE-coverage Pareto

---

## 7. 未来 3 个月路线

### 第 5-8 周：v1.11 实验深化

- [ ] 在 Line9 holdout 上验证 abstention 有效性
- [ ] 确认能否减少高置信错层
- [ ] 如果不能，记录原因并转为仅做 reliability reporting

### 第 9-12 周：论文素材准备

- [ ] 主图（B-scan 对比、centerline、Pareto）
- [ ] 主表（指标表、消融表）
- [ ] baseline 对比（SVD / RPCA / f-k / CR-Net / simpler UNet）
- [ ] 定位写初稿

---

## 8. 未来 6-12 个月方向

### 8.1 如果 v1.11 成功
- 把 confidence / abstention 作为 v1 配套方法，合成论文
- 进入 v2：self-supervised target adaptation
- 考虑两阶段系统

### 8.2 如果 v1.11 不成功
- 论文仍然可以发：保守的 raw-only baseline + 诚实报告局限
- 核心贡献改为：
  - 数据建设
  - split 设计
  - 弱监督界面拾取基线
  - 为后续研究提供可复现基线

### 8.3 长期战略
- 等有新场地数据后可验证跨区泛化
- 等仿真 pipeline 成熟后可做 synthetic pretraining
- 等有多模型可做 ensemble
- **不急于做这些，等基础扎实了再扩展**

---

## 9. 最终优先级清单（从最高到最低）

| 优先级 | 事项 | 预计时间 |
|--------|------|---------|
| P0 | 完整 Line9 holdout 复现 | 1-2 天 |
| P0 | 主指标固化（valid-line + holdout） | 1 天 |
| P0 | 后处理参数锁定 | 1 天 |
| P1 | 传统 baseline 对比（SVD/RPCA/f-k） | 1 周 |
| P1 | v1.11 confidence/abstention 实验 | 2-3 周 |
| P1 | 主图/主表准备 | 2 周 |
| P2 | 论文初稿定位与撰写 | 3-4 周 |
| P2 | 消融实验（模块必要性验证） | 2 周 |
| P3 | 工程清理（配置存档、outputs 分类） | 间断进行 |
| P4 | 仿真管道就绪后评估 sim pretrain | 待条件成熟 |
| P4 | borehole weak sup 作为增强实验 | v2 |
| P5 | f-k / SVD 等物理分支 | v3 |

---

## 10. 一句话总结

> **PGDA-CSNet 当前最该做的事不是追求技术复杂度，而是把一条线上最扎实的 raw-only 弱监督界面拾取基线跑透，加上诚实可靠的评估与限制说明——从"我能做得更复杂"转向"我能证明我做的对"。**
