# PGDA-CSNet v1 执行路线图

**版本**: v1.0 (2026-06-27)  
**对应方案**: `docs/PGDA_CSNET_V1_LOCKED_SCHEME.md`  
**状态**: 待执行

---

## 1. 路线目标

本路线图服务于已锁定的 v1 方案，目标不是继续发散网络设想，而是：

1. 固化当前 frozen baseline
2. 完成 full holdout 可复现验收
3. 完成论文主结果最小包
4. 进入 v1.11 confidence / abstention 主线

---

## 2. 第一阶段：冻结与复现

### 2.1 必须完成

- [ ] 固定主工程目录与主 checkpoint
- [ ] 固定 `data_corrected_v1_4_terrain_direction`
- [ ] 固定 `Line9` locked holdout 协议
- [ ] 固定 postprocess 参数
- [ ] 固定主结果路径与 SHA256

### 2.2 推荐动作

1. 使用当前迁移后的主工程目录作为工作基线：

```text
D:\Claude\PGDA-CSNet\workspace\transfer_20260627_142748\PGDA-CSNet_transfer_bundle_20260627_142748\PGDA_CSNet_v0_9_6_SEARCH_WINDOW_GUARD
```

2. 运行 baseline smoke：

```bash
bash 01_fast_cpu_check_raw_only.sh
```

3. 完整运行 frozen v1.9D 的 Line9 holdout eval。

4. 记录：
   - checkpoint SHA256
   - eval metrics CSV
   - full preview path
   - postprocess parameter snapshot

---

## 3. 第二阶段：主论文最小结果包

### 3.1 主结果必须包含

- valid-line average 指标
- Line9 locked holdout 指标
- 代表性 B-scan 可视化
- centerline 结果图
- pick / reject 统计
- 和传统 baseline 或较简单深度 baseline 的对比

### 3.2 必须明确分开报告

- measured-data valid-line average
- Line9 locked holdout
- strict zero-material leave-one-line-out

### 3.3 不允许混写

以下内容不能混成“整体性能”：

- 训练线表现
- LineX1 结果
- ensemble upper bound
- robust normalization 单线最优结果

---

## 4. 第三阶段：v1.11 主线（推荐）

### 4.1 核心方向

v1 之后不优先继续换 backbone，而是转向：

> **confidence / abstention-first**

即：

- 当模型高置信时拾取
- 当模型不稳定时拒判/留空
- 用 MAE-coverage Pareto 而不是单阈值做结论

### 4.2 推荐实施内容

1. 生成 frozen v1.9D 的多视角推理：
   - default normalization
   - robust normalization
   - optional flip / scale variants

2. 提取 confidence features：
   - path probability
   - presence probability
   - center / mask 一致性
   - ridge contrast
   - DP jump / curvature
   - 跨视图一致性

3. 用 source lines 做阈值或轻量规则学习。

4. 输出 MAE-coverage Pareto。

### 4.3 目标

v1.11 不一定要降低所有情形下的 MAE，但必须：

- 减少高置信错层
- 给出更可信的拒判机制
- 在 coverage 可控时提升 reliability

---

## 5. 第四阶段：论文组织建议

### 5.1 建议章节顺序

1. 问题背景：低频 UAV-GPR、单场地、杂波强、标签弱
2. 数据与标签：corrected v1.4、Line9 holdout、LineX1 排除
3. 方法：raw-only weakly supervised interface picking
4. 后处理：mask + center fusion + breakable DP
5. 主结果：valid-line + Line9 holdout
6. 局限性：strict LOO 仍弱
7. 下一步：confidence-aware inference

### 5.2 方法主卖点

不是“重建出干净雷达图”，而是：

- raw-only
- 弱监督
- 实测可复现
- holdout 协议严格
- 保守可靠的界面拾取

---

## 6. 当前不做的内容

以下内容不进入 v1 主执行路线：

- f-k branch
- SVD branch
- FiLM geometry conditioning
- terrain / flight metadata 主输入
- clean reconstruction 主目标
- clutter map 主监督
- V16 / V17 替代主数据
- 大规模 backbone 再搜索
- borehole weak sup 主线化
- adversarial DA 主线化

---

## 7. 当前最优先的下一步

按优先级排序：

### Priority 1
- 完整复现 frozen v1.9D 的 full Line9 holdout 结果

### Priority 2
- 固化主图、主表、metrics、checkpoint hash

### Priority 3
- 启动 v1.11 confidence / abstention 路线

### Priority 4
- 仅在 v1.11 做完后，再考虑有限 target adaptation

---

## 8. 一句话执行结论

> **当前阶段不再继续发散新网络，而是围绕 frozen v1.9D + corrected v1.4 + Line9 locked holdout，把一个可信、可复现、可发表的 raw-only 弱监督界面拾取基线做扎实。**
