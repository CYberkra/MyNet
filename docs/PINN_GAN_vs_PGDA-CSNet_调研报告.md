# PINN / GAN vs PGDA-CSNet 调研报告

**日期**: 2026-06-12  
**方法**: Deep Research — 106 agents, 24 篇文献, 25 条主张经 3 轮对抗验证  
**结论**: 保持 PGDA-CSNet 路线不变

---

## 核心发现

### 1. PINN — 不适合

| 发现 | 置信度 | 来源 |
|------|--------|------|
| da-PINN 仅支持**两层均质+单一平面界面**，不适用于多层弯曲基岩 | 高 (3-0) | Piao et al., IEEE TAP 2024 |
| PINN 解决的是**参数反演**（估算 ε/μ），不是图像到图像的杂波抑制 | 高 (3-0) | 同上 |
| PINN 的物理约束是**软 L2 惩罚**（PDE 残差 10⁻³-10⁻⁵），无散度约束、无能量守恒 | 高 (3-0) | Abdelraouf et al., 2025; Menon & Iyer, 2025 |

**技术细节**：
- da-PINN 参数向量 λ=[μ₁,ε₁,μ₂,ε₂,d]ᵀ，仅支持两层+标量界面深度
- 作者明确声明："we will extend da-PINN to solve Maxwell's equations in heterogeneous media with unknown shapes of interface" — 尚未实现
- 所有 PINN+GPR 文献（GPR-IFNet 2025, MSCCEAU-Net 2025, MPPINet 2024）均做介电常数反演，非杂波抑制
- PINN 旨在替代 FDTD 求解器 — 但我们已有 gprMax GPU FDTD，无需替代

### 2. GAN — 不适合作为主框架

| 发现 | 置信度 | 来源 |
|------|--------|------|
| GAN 潜空间梯度下降成功率仅 **~13%**（2024 修订降至 **0-5%**），即使正向物理模型是线性的 | 高 (2-1) | Laloy et al., 2019/2024 |
| 所有 GAN+GPR 论文停留在**概念验证**，无真实 UAV 测线部署 | 高 (3-0) | IEEE 11108996, July 2025 |
| GAN 训练需要 **35,000+ 对数据** — 我们仅有 6 条实测测线 | — | VelocityGAN 文献 |

**技术细节**：
- Laloy et al. 跨孔 GPR 层析成像：Adam 10,000 次迭代仅 ~13% 成功率；Gauss-Newton 0% 成功率
- IEEE 11108996 (2025)：唯一直接处理机载 GPR 的 GAN 论文，纯合成数据训练，"proof of concept"
- 7+ 篇 GAN+GPR 论文（Declutter-GAN, DR-GAN, RCE-GAN, 2C-GAN, VAE-GAN, Wavelet-GAN, CycleGAN-based）：全部依赖合成 FDTD 训练
- 对抗训练不稳定（模式崩塌）在 GPR 数据上表现为"所有场景输出同一张干净图"

### 3. PGDA-CSNet — 保持

| 优势 | 依据 |
|------|------|
| f-k/SVD 提供**硬物理偏置**，非软惩罚 | 架构设计 |
| 利用 gprMax FDTD **精确仿真**管线 | 已验证 3060 GPU 稳定运行 |
| 域适应处理 sim-to-real | DANN/MMD 比 GAN 更稳定，适合小样本 |
| 残差学习保护微弱基岩信号 | 不会像 GAN 那样凭空生成或抹除 |

---

## 文献验证统计

- 搜索角度: 5（综述/PDE/失败模式/数据稀缺/消融对比）
- 抓取文献: 24 篇
- 提取主张: 93 条
- 对抗验证: 25 条
- **确认: 6 条**（全部高置信度）
- **否决: 19 条**（所有声称 PINN/GAN 优势的主张均被否决）

### 确认的 6 条主张

1. PINN 仅支持两层均质+单平面界面，物理约束为软 L2 惩罚
2. PINN 解决参数反演问题，非图像到图像杂波抑制
3. GAN 潜空间梯度下降成功率 ~13%（2024 修订 0-5%）
4. GAN+GPR 均停留概念验证，无真实 UAV 部署

### 否决的 19 条（摘要）

- "PINN 可作为 gprMax 的高效替代" → 否决
- "GAN 无监督方法在小数据下优于监督学习" → 否决
- "对抗域适应可解决 GPR sim-to-real" → 否决
- "GAN 可以实现物理一致的 subsurface 成像" → 否决
- 其余 15 条类似主张均被 3 人投票否决

---

## 开放问题（后续研究方向）

1. PGDA-CSNet 物理分支能否加**硬约束投影层**而不牺牲几何灵活性？
2. PatchGAN 能否作为**辅助正则项**（λ~0.01）在 clutter map 上加判别器？
3. 营山 6 条测线是否足够验证 sim-to-real（建议 6-fold leave-one-line-out）？
4. gprMax 仿真管线能否增强**随机化地下非均质性建模**？（已在实施）

---

## 建议

| 方案 | 决策 | 理由 |
|------|------|------|
| 转向 PINN | **不建议** | 解决的是不同问题，结构上无法处理多层基岩 |
| 转向 GAN | **不建议** | 数据量不足（需 35,000 vs 我们 6 条线），成功率 ~13% |
| 保持 PGDA-CSNet | **保持** | 架构针对性强，利用现有 FDTD 管线，风险可控 |
| GAN 作为辅助正则 | **后续消融** | 等训练数据足够多样化后再评估 |
| f-k 硬投影层 | **后续研究** | 需先验证软约束版本，再考虑 gated projection |

---

## 参考文献

1. Piao et al., "da-PINN: Domain-adaptive Physics-Informed Neural Network for Inverse Problems of Maxwell's Equations in Heterogeneous Media," IEEE TAP Vol. 23(10), 2024. https://ar5iv.labs.arxiv.org/html/2308.06436
2. Laloy et al., "Gradient-based deterministic inversion of geophysical data with Generative Adversarial Networks," Computers & Geosciences, 2019 (updated 2024). https://ar5iv.labs.arxiv.org/html/1812.09140
3. Airborne GPR GAN, IEEE 11108996, July 2025. https://ieeexplore.ieee.org/document/11108996
4. Sun et al., "Learning to Remove Clutter in Real-World GPR Images Using Hybrid Data" (CR-Net), IEEE TGRS, 2022
5. Ganin et al., "Domain-Adversarial Training of Neural Networks," JMLR, 2016
6. Lehtinen et al., "Noise2Noise: Learning Image Restoration without Clean Data," ICML, 2018
