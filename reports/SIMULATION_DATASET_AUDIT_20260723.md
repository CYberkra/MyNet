# MyNet 仿真数据集现状分析与规范化建议

Date: 2026-07-23
Scope: MyNet_master_cleanup / PSGN-CSNet 仿真数据治理

---

## 一、当前状态总览

### 1.1 五层资产架构（规范已定义 vs 实际状态）

| 层级 | 规范路径 | 规范状态 | 实际状态 | 差距 |
|---|---|---|---|---|
| L1 Source | `00_controls/` | Git 版本化 | ✅ 43+ 案例已推送 | 无 |
| L2 Release Spec | `release_specs/` | Git 版本化 | ⚠️ 仅 1 个（FORMAL06C） | 严重缺失 |
| L3 Released Evidence | `02_released_solver_evidence/` | Git + LFS | ⚠️ 仅 FORMAL06C | 严重缺失 |
| L4 Training Release | `02_released_canonical/` | Git + LFS | ❌ 目录不存在 | 完全缺失 |
| L5 Runtime Cache | `01_solver_runs/` | 永不版本化 | ✅ 509MB, 40+ 案例 | 无（但缺乏清理） |

### 1.2 案例注册状态（simulation_cases.csv）

| 类别 | 已注册 | 实际存在 | 未注册 |
|---|---|---|---|
| CTRL 控制案例 | 0 | 4 (CTRL01-04) | 4 |
| FORMAL01 基岩密窗 | 0 | 4 (F0-F3) | 4 |
| FORMAL02 分级基岩 | 0 | 1 (G0) | 1 |
| FORMAL03 源消融 | 0 | 3 (Gabor80/Ricker65/80) | 3 |
| FORMAL04 地质因子 | 0 | 3 (A/B/C) | 3 |
| FORMAL05 平衡纹理 | 0 | 1 | 1 |
| FORMAL06 接口系列 | 3 (06, 06B, 06C) | 4 (06, 06B, 06C, 06D-H) | 5 (06D-H) |
| FORMAL07 地层系列 | 2 (07A, 07B) | 2 | 0 |
| FORMAL08 真实感系列 | 2 (08A, 08B) | 2 | 0 |
| FORMAL09C 有限层 | 0 | 2 (P1, P2) | 2 |
| IV2 Family | 6 (F01-03 ±) | 6 | 0 |
| SHAPE01 基底几何 | 0 | 2 (PILOT, VISUAL_SHORT) | 2 |
| SHAPE02-05 基底形状 | 0 | 7 目录 | 7 |
| N256 原生 256 | 0 | 8 (CV01-04, F01-04, N01-02) | 8 |
| **合计** | **13** | **~50+** | **~37** |

### 1.3 Release Class 分布

| Class | 定义 | 当前案例数 | 备注 |
|---|---|---|---|
| `source_only` | 仅有源定义，无求解输出 | ~20 | CTRL, FORMAL01-05, SHAPE01 等 |
| `rejected_evidence` | 被拒绝，仅保留审计报告 | 3 | FORMAL06, 06B, 07A |
| `development_evidence` | 开发基线，人工接受 | 2 | FORMAL06C, 07B |
| `training_candidate` | 完整运行，待审批 | 0 | **缺失** |
| `formal_release` | 正式训练发布 | 0 | **缺失** |

---

## 二、识别的关键问题

### 问题 1：治理注册表严重不完整

**证据**：`simulation_cases.csv` 仅注册 13 行，但 `00_controls/` 中有 40+ 个案例目录。

**影响**：
- 无法从单一入口了解全部仿真资产状态
- 新团队成员无法快速了解哪些案例可用、哪些不可用
- 无法自动化检查案例完整性

**涉及未注册案例**：
- CTRL01-04（4个基础控制）
- FORMAL01 F0-F3（4个基岩密窗族）
- FORMAL02-05（4个早期实验）
- FORMAL06D-H（5个接口消融）
- FORMAL09C P1/P2（2个有限层）
- SHAPE01-05（7个基底几何系列）
- N256 CV01-04/F01-04/N01-N02（8个原生256试点）

### 问题 2：发布流程执行率低

**证据**：40+ 个有求解输出的案例中，仅 1 个（FORMAL06C）有 `release_spec` 和 `released_solver_evidence`。

**影响**：
- 大量求解输出（509MB）处于"已完成但未审计"状态
- 无法判断哪些求解输出值得保留、哪些应该删除
- 其他机器无法复现已完成的审计结果

**具体案例**：
- FORMAL06D-H：已完成求解，有决策报告，但无 release_spec
- FORMAL07B：已通过代理接受，但无 release_spec
- IV2_F01-03：已完成求解，但无 release_spec
- SHAPE02 GEO01-15 + CAL00-01：已完成求解，但无 release_spec

### 问题 3：Training Release 层完全缺失

**证据**：`02_released_canonical/` 目录不存在。

**影响**：
- 当前没有任何仿真数据可用于正式训练
- 这是论文训练的主要阻塞项
- 即使有 development_evidence，也无法直接转化为训练数据

### 问题 4：01_solver_runs 缺乏生命周期管理

**证据**：509MB 的求解缓存中包含已拒绝案例（如 FORMAL06B、07A）的求解输出。

**影响**：
- 磁盘空间浪费
- 增加新成员的理解成本
- 无法自动清理已过时/已拒绝的求解缓存

### 问题 5：案例 ID 命名空间不一致

**证据**：
- `00_controls/` 中：SHAPE02 作为一个整体目录
- `01_solver_runs/` 中：BS01-04, CAL00-01, GEO01-15 等子目录
- `simulation_cases.csv` 中：完全没有 SHAPE 系列

**影响**：
- 源文件和求解输出之间的映射关系不清晰
- 难以追踪一个案例从源到求解到发布的完整生命周期

---

## 三、规范化建议：仿真数据集做法

### 3.1 核心原则

1. **单一注册表**：`simulation_cases.csv` 是所有仿真案例的唯一权威来源
2. **状态驱动**：每个案例必须有明确的 release_class，状态转换是单调的
3. **发布必填**：任何保留的求解输出必须通过 `release_spec` 定义
4. **训练隔离**：只有 `formal_release` 级别的数据才能进入训练集

### 3.2 推荐的工作流程

```
Step 1: 源定义
  └─> 在 00_controls/ 中创建案例
  └─> 生成 POLICY.json 和 scene_manifest.json
  └─> 注册到 simulation_cases.csv (status = planned_pre_solver)

Step 2: 求解运行
  └─> 在 01_solver_runs/ 中运行 gprMax
  └─> 更新 simulation_cases.csv (status = runtime_completed)

Step 3: 审计门
  └─> 静态审计 -> 因果审计 -> 形态审计 -> 人工审查
  └─> 更新 simulation_cases.csv (status = 对应结果)

Step 4: 发布（仅对通过的案例）
  └─> 创建 release_specs/<case>.json
  └─> 运行 package_gprmax_release.py
  └─> 生成 02_released_solver_evidence/
  └─> 更新 simulation_cases.csv (solver_evidence_released = true)

Step 5: 训练发布（仅对正式案例）
  └─> 从 released evidence 提取 501xN 规范数组
  └─> 存入 02_released_canonical/
  └─> 更新 simulation_cases.csv (train_allowed = true)
```

### 3.3 simulation_cases.csv 字段规范

当前已有字段（应全部保留）：
- `case_id`, `source_group`, `case_path`, `scene_family_id`
- `status`, `line9_conditioned`, `human_decision`
- `train_allowed`, `line9_holdout_allowed`, `negative_semantics`
- `exclusion_reason`, `contract_id`, `target_presence`
- `solver_evidence_released`, `formal_training_allowed`

建议新增字段：
- `release_class`: source_only | rejected_evidence | development_evidence | training_candidate | formal_release
- `solver_run_dir`: 01_solver_runs 中的对应目录名
- `release_spec_path`: release_specs 中的规范路径
- `released_evidence_dir`: 02_released_solver_evidence 中的路径
- `canonical_training_dir`: 02_released_canonical 中的路径
- `created_date`, `last_updated_date`
- `superseded_by`: 如果被后续案例替代，记录替代者 ID

### 3.4 立即行动清单

#### 高优先级（本周）

1. **补齐 simulation_cases.csv 注册**
   - 为所有 00_controls/ 中的案例添加记录
   - 设置正确的 `release_class` 和 `status`
   - 标记已拒绝的案例和原因

2. **为已审计案例创建 release_spec**
   - FORMAL06D/E/F/G/H：已有决策报告，应创建规范
   - IV2_F01/F02/F03：已完成求解，评估是否值得发布
   - SHAPE02 候选通过案例：筛选 GEO 子案例

3. **清理 01_solver_runs/**
   - 删除已拒绝案例的求解输出（如 FORMAL06B、07A）
   - 为保留的案例创建 release_spec 并打包后删除原始求解缓存

#### 中优先级（本月）

4. **创建 02_released_canonical/ 目录结构**
   - 定义规范数组格式（501xN，NPZ 封装）
   - 从已发布的 evidence 中提取训练数据

5. **建立自动化检查脚本**
   - `scripts/audit_simulation_registry.py`：检查注册表完整性
   - `scripts/cleanup_solver_runs.py`：自动清理过期缓存
   - `scripts/validate_release_spec.py`：验证发布规范完整性

#### 长期优先级

6. **推动案例从 development_evidence 晋升到 training_candidate**
   - 需要完成 native-256 完整对运行
   - 需要通过严格的零材料审计
   - 需要独立（非 Line9 条件化）来源证明

---

## 四、具体数据速查

### 4.1 当前 01_solver_runs/ 内容（509MB）

| 案例 | 大小 | Release Class 建议 | 行动 |
|---|---|---|---|
| FORMAL06C | 15M | development_evidence | 已有 release_spec，保留 |
| FORMAL06D | 53M | development_evidence | **需创建 release_spec** |
| FORMAL06E | 17M | rejected_evidence | 盲审被拒，可清理 |
| FORMAL06F | 16M | rejected_evidence | 盲审被拒，可清理 |
| FORMAL06G | 16M | development_evidence | **需创建 release_spec** |
| FORMAL06H | 18M | development_evidence | **需创建 release_spec** |
| FORMAL07A | 7.6M | rejected_evidence | 已被拒，可清理 |
| FORMAL07B | 15M | development_evidence | **需创建 release_spec** |
| FORMAL08A | 12M | rejected_evidence | 保留为消融记录 |
| FORMAL08B | 13M | rejected_evidence | 保留为失败消融记录 |
| FORMAL09C P1/P2 | 18-19M | source_only | 仅 smoke 运行，未完整 |
| IV2_F01 POS/NEG | 19-3.4M | training_candidate | **需评估是否晋升** |
| IV2_F02 POS/NEG | 32-2.1M | development_evidence | 机制转移验证 |
| IV2_F03 POS/NEG | 15-2.2M | training_candidate | **需评估是否晋升** |
| SHAPE02 BS01-04 | 3.7-40M | source_only | 仅 16/32 道稀疏运行 |
| SHAPE02 CAL00-01 | 47-3.9M | calibration_only | 仅参考运行 |
| SHAPE02 GEO01-15 | 2.3-22M | candidate | **需筛选通过案例** |

---

## 五、结论

当前仿真数据治理的**核心瓶颈**是：

1. **发布流程执行率过低**（1/40+ 有 release_spec）
2. **训练发布层完全缺失**（无 02_released_canonical/）
3. **注册表不完整**（simulation_cases.csv 缺 37 个案例）

**最关键的下一步**是：
- 为已通过审计的案例（FORMAL06D, 06G, 06H, 07B, IV2_F01）创建 release_spec
- 将它们的求解输出从 01_solver_runs/ 转移到 02_released_solver_evidence/
- 清理已拒绝案例的求解缓存，释放 50-100MB 空间
- 推动至少一个案例完成 native-256 完整对运行，晋升到 training_candidate

---

## 更新记录

**2026-07-23 后续审计结果（由 agent 执行后更新）**

以下案例在原始审计报告发布后，经 native-64 盲审被明确拒绝，不再适合创建 release_spec：

| 案例 | 原建议 | 实际结果 | 决策报告 |
|---|---|---|---|
| FORMAL06G | 需创建 release_spec | **rejected** — native-64 盲审拒绝 | `reports/formal06g_terrain_acquisition_20260722/FORMAL06G_NATIVE64_DECISION.md` |
| FORMAL06H | 需创建 release_spec | **rejected** — native-64 盲审拒绝 | `reports/formal06h_source_temporal_20260722/FORMAL06H_NATIVE64_DECISION.md` |
| FORMAL07B | 需创建 release_spec | **development_successor only** — 明确不能进入 formal training | `reports/formal07b_weak_aperiodic_background_20260715/FORMAL07B_RUNTIME_AND_MORPHOLOGY_AUDIT.md` |
| FORMAL06D | 需创建 release_spec | **pair32 passed, native64 rejected** — 仅保留 pair32 因果机制证据 | `reports/formal06d_successor_20260722/FORMAL06D_PAIR32_DECISION.md` |

**结论修正**：截至 2026-07-23，唯一已有 release_spec 的案例仍为 **FORMAL06C**。06G/06H 的求解缓存已清理（释放 34MB），06E/06F/07A/08A/08B 的缓存也已清理（释放 65.6MB），累计释放 **99.6MB**。

当前 `simulation_cases.csv` 已注册 35 条记录（含 header 36 行），所有已拒绝案例状态已正确标注。
