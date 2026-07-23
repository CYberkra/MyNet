# 仿真数据集视觉审计报告

Date: 2026-07-23
Auditor: Agent visual review
Scope: Raw B-scan, background-suppressed time-power gain, AGC, and causal pair audits

---

## 审计方法

对每个案例查看以下预览图：
- **Raw full-scene Ez**: 原始振幅，保留直达波主导性
- **Background-suppressed + time-power gain (tpower)**: 水平背景抑制 + 时间幂增益，用于结构判读
- **Background-suppressed + AGC**: 自动增益控制，显示更多细节但改变振幅平衡
- **Causal pair audit (smoke1)**: 单道 full/no-basal 严格对，验证因果性
- **Distributed morphology**: 多道分布视图的盲审对比

---

## 一、已发布基准: FORMAL06C

| 预览 | 文件 | 观察 |
|---|---|---|
| distributed32 raw | `FORMAL06C_distributed32_raw_tpower15.png` | 基底反射清晰，连续多周期波形，形态自然 |
| distributed32 agc | `FORMAL06C_distributed32_raw_agc13.png` | 结构清晰，层状背景纹理可见但不干扰目标 |
| smoke1 pair | `positive_pair_spatial_preview.png` | **gate=PASS**, target/background=151.67, dropout=0% |

**结论**: 人工接受的开发基线。多周期黑白波形特征、连续渐变基底包络、无规则水平梳状伪影。

---

## 二、FORMAL06 接口系列消融（D–H）

### FORMAL06D — Independent Mechanism

| 预览 | 观察 |
|---|---|
| pair32 strict audit | **gate=HOLD**, target/background=100.81, 因果性通过 |
| native64 feature tpower | 基底反射存在但形态不如06C锐利，有轻微水平条纹噪声 |
| blind8 full tpower | 8道盲审下基底可见但背景均匀性下降 |

**视觉判断**: pair32 因果证据充分，但 native64 盲审下形态不够理想。native64 被拒合理 — 独立机制改变了波形特征但未能保持06C的清晰度。

### FORMAL06E — Nonlayered Cover ❌ REJECTED

| 预览 | 观察 |
|---|---|
| native64 tpower | **明显缺陷**: 非层状覆盖产生了大量水平层状噪声（平行叶瓣），基底反射区域被严重干扰 |
| smoke1 strict audit | **gate=PASS**, target/background=80.14（因果性通过但比值低于06C） |

**视觉判断**: smoke1 因果对通过说明物理机制正确，但 native64 分布视图中非层状覆盖引入了系统性的水平条纹伪影。盲审拒绝合理。

### FORMAL06F — Single Cap Transition ❌ REJECTED

| 预览 | 观察 |
|---|---|
| native64 tpower | 单帽过渡未能改善形态，基底反射区域弥散，波列不紧凑 |

**视觉判断**: 与06E类似，单帽过渡不是有效的形态修复策略。

### FORMAL06G — Terrain Acquisition ❌ REJECTED

| 预览 | 观察 |
|---|---|
| native64 tpower | 地形起伏引入采集路径相关噪声，基底反射可见但被水平条纹干扰 |
| smoke1 strict audit | **gate=PASS**, target/background=163.58, 但 full-scene local ratio=2.08（显著低于06C的5.46） |

**视觉判断**: 单道因果性强（高T/B比），但地形耦合导致分布视图中信噪比下降。盲审拒绝合理。

### FORMAL06H — Source Temporal ❌ REJECTED

| 预览 | 观察 |
|---|---|
| native64 tpower | 源时间变化引入相位不一致，整个图像有水平条纹，基底反射模糊 |

**视觉判断**: 源波形变化破坏了多周期波形的相干性，导致分布视图中形态劣化。

---

## 三、FORMAL07 地层系列

### FORMAL07B — Weak Aperiodic Background

| 预览 | 观察 |
|---|---|
| blind common32 vs 06C | 与06C极其相似，背景纹理略多但不遮蔽目标 |
| smoke1 audit | 因果对通过 |

**视觉判断**: 单因素消融（弱非周期背景）成功保留了06C的核心形态。07B与06C的 blind32 对比显示：
- 路径/几何相关性 0.99994（几乎相同）
- 目标/邻域背景 RMS 16.74 vs 17.29（略好）
- 7个显著有符号波瓣（与06C相同）

**结论**: 作为 controlled development successor 合格，但明确不能进入 formal training（Line9-conditioned）。

### FORMAL07A — Continuous Stratigraphy ❌ REJECTED（已清理）

| 预览 | 观察 |
|---|---|
| blind8 span audit | 规则层状背景过强，基底反射被淹没 |

---

## 四、FORMAL08 真实感系列

### FORMAL08A — Line9 Realism Background

| 预览 | 观察 |
|---|---|
| distributed32 tpower | 基底反射清晰，但背景纹理与实测Line9差异较大 |
| vs Line9 equal width | 仿真（左下）背景比实测（右下）更结构化/层状 |

**视觉判断**: 真实感背景尝试未成功。仿真背景呈现过多人工层状结构，而实测Line9背景更随机/噪声化。目标反射虽可见但背景不真实。

### FORMAL08B — Deep Background

| 预览 | 观察 |
|---|---|
| distributed32 tpower | 深背景版本，目标主导性反而增加（与意图相反） |

**视觉判断**: 深背景未能抑制目标，反而使目标更突出。失败消融。

---

## 五、IV2 独立系列

### IV2_F01 — Gentle Aperiodic Cover Bedrock

| 预览 | 观察 |
|---|---|
| distributed32 blind tpower | **非常清晰**！基底反射明显，背景抑制效果好，形态干净 |
| distributed32 blind agc | 结构丰富，目标可见性高 |

**视觉判断**: 目前最干净的独立物理先验案例。非Line9条件化，形态接近06C但背景更自然。等待native-256完整对运行。

### IV2_F02 — Formal06C Mechanism Transfer

| 预览 | 观察 |
|---|---|
| raw tpower15 vs agc13 | 形态与06C相似，机制转移验证成功 |

**视觉判断**: 机制转移验证通过，但继承Line9条件化决策，不能进入formal training。

### IV2_F03 — Instrument Band

| 预览 | 观察 |
|---|---|
| blind comparison vs 06C | 形态显著不同，波形更窄更锐利，不像06C的柔和多周期波形 |

**视觉判断**: 硬件频带效应导致波形特征偏离项目偏好的形态。形态学上不够理想，等待native-256完整运行后再评估。

---

## 六、FORMAL09C 有限层系列

### FORMAL09C_P1 — Dense Physical Finite Laminae

| 预览 | 观察 |
|---|---|
| native64 blind tpower | **大量水平层状反射**（来自密集有限层），基底反射在底部可见但被层状反射严重干扰 |

**视觉判断**: 密集物理有限层引入了大量内部层状反射，目标-背景对比度下降。需要评估层状反射是否可作为训练中的干扰项。

### FORMAL09C_P2 — Sparse Irregular Finite Laminae

| 预览 | 观察 |
|---|---|
| native64 blind tpower | 层状反射比P1少，基底反射较清晰，但仍有一些干扰 |

**视觉判断**: 比P1更干净，但仍受有限层反射影响。

---

## 七、SHAPE 系列

### SHAPE02 GEO05 — Aperiodic Multiscale Low

| 预览 | 观察 |
|---|---|
| trace128 pair audit | target/pre-target=41.3 dB，单道因果性极强 |
| span16 audit | 空间预览显示基底形态与06C相似 |

**视觉判断**: 有希望的基底几何变体。128道因果对验证通过，需要进一步评估分布形态。

---

## 八、视觉审计总评

### 8.1 形态质量排序（基于视觉）

| 排名 | 案例 | 状态 | 视觉评价 |
|---|---|---|---|
| 1 | FORMAL06C | ✅ 已发布 | 基准质量，多周期波形清晰，背景干净 |
| 2 | IV2_F01 | ⏳ 等待native256 | 独立物理先验中最干净，形态接近06C |
| 3 | FORMAL07B | ⚠️ successor only | 与06C几乎相同，背景略多但不干扰 |
| 4 | FORMAL06D | ⚠️ pair32 only | pair32形态可接受，native64不够清晰 |
| 5 | FORMAL09C_P2 | ⏳ smoke完成 | 有限层干扰较少，基底可见 |
| 6 | FORMAL08A | ❌ not promoted | 背景不真实，目标虽可见 |
| 7 | FORMAL06G/H | ❌ rejected | native64形态劣化明显 |
| 8 | FORMAL06E/F | ❌ rejected | 平行叶瓣/弥散严重 |
| 9 | FORMAL09C_P1 | ⏳ smoke完成 | 密集层状反射干扰大 |
| 10 | IV2_F03 | ⏳ 等待native256 | 波形过锐，偏离偏好形态 |

### 8.2 关键视觉缺陷模式

| 缺陷模式 | 出现案例 | 描述 |
|---|---|---|
| 平行叶瓣噪声 | 06E | 非层状覆盖产生的水平条纹 |
| 波形弥散 | 06F, 06H | 单帽过渡/源变化导致波列不紧凑 |
| 采集路径噪声 | 06G | 地形耦合引入水平条纹 |
| 背景过结构化 | 08A | 仿真背景比实测更规则/层状 |
| 层状反射干扰 | 09C_P1 | 密集有限层产生大量内部反射 |
| 波形过锐 | IV2_F03 | 硬件频带导致窄波形 |

### 8.3 训练数据前景评估

**最有希望晋升到 training_candidate 的案例**:

1. **IV2_F01** — 独立物理先验，形态干净，等待native-256完整对
2. **SHAPE02 GEO05** — 128道因果对通过，需要分布形态审计
3. **FORMAL09C_P2** — 有限层干扰较少，需要完整对运行

**不应进入训练的案例**:
- 所有 FORMAL06E/F/G/H（已明确拒绝）
- FORMAL08A/B（not promoted，背景问题）
- IV2_F03（形态偏离）

---

## 九、建议

1. **优先完成 IV2_F01 的 native-256 完整对运行** — 当前视觉上最独立的候选
2. **FORMAL06D 保留 pair32 因果证据即可**，native64 部分无需 release_spec
3. **清理决策确认**: 06E/F/G/H 的求解缓存已清理，视觉审计支持该决策
4. **FORMAL09C 系列**需要评估有限层反射是否可作为有效的训练干扰项
