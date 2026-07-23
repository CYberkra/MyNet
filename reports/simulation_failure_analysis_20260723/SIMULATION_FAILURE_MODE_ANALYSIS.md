# GPR 仿真数据集失败模式分析与质量控制经验总结

**日期**: 2026-07-23  
**分析范围**: FORMAL06E–06H (rejected), IV2_F01 (user-rejected), 以及对照组 FORMAL06C/08A/08B/09C-P2  
**目的**: 总结失败仿真数据的共性问题和根因，建立可自动化的批量质量控制规则

---

## 1. 失败模式总览

| 案例 | 失败模式 | 根因类别 | 严重程度 |
|------|----------|----------|----------|
| **IV2_F01** | 增益后背景几乎空白 | 覆盖层模型过于均质 | 🔴 致命 |
| **FORMAL06E** | 等间距平行条纹（人工感） | 层理生成算法过于规则 | 🔴 致命 |
| **FORMAL06F** | 背景纹理稀疏到近乎空白 | 覆盖层过渡模型过于简单 | 🟡 严重 |
| **FORMAL06G** | 异常强反射 + P99.5增益异常高10倍 | 地形/采集参数设置不当 | 🔴 致命 |
| **FORMAL06H** | 基岩反射模糊发散 | 波源时域设计不当 | 🟡 严重 |

---

## 2. 五大失败模式详解

### 模式 1: 覆盖层过于均质 — "Blank Background"

**典型表现**:
- 背景抑制 + tpower 增益后，0-500ns 区域呈均匀的灰色空白
- 仅有基岩界面处有反射，上方无任何可见纹理
- P99.5 显示增益值与正常案例相近，但背景区域的标准差极低

**受影响案例**: IV2_F01 (最严重), FORMAL06F, 早期 FORMAL01-03 部分参数

**物理根因**:
- 覆盖层介电常数模型使用了单一值或简单线性渐变
- 没有引入随机介电常数扰动来模拟真实地质的含水量/粒度变化
- 或者扰动幅度太小（< 0.5 relative permittivity），低于 GPR 分辨率阈值

**可量化诊断指标**:
```python
def check_blank_background(bscan_tpower, time_ns_window=(50, 400)):
    """
    在背景抑制+增益后的B-scan中，检查指定时间窗内的纹理丰富度。
    """
    t0, t1 = time_ns_window
    region = bscan_tpower[t0:t1, :]  # 排除直达波和基岩反射
    
    # 指标1: 标准差（反映纹理强度）
    std_val = np.std(region)
    if std_val < 0.001:  # 经验阈值
        return "REJECT: Blank background (std={:.4f})".format(std_val)
    
    # 指标2: 空间频率能量（检测是否有水平纹理）
    fft_row = np.fft.rfft(region, axis=0)
    freq_energy = np.sum(np.abs(fft_row[1:10]))  # 低频水平纹理能量
    if freq_energy < 0.01:
        return "REJECT: No horizontal texture (freq_energy={:.4f})".format(freq_energy)
    
    return "PASS"
```

**修复策略**:
- 覆盖层必须引入**多尺度随机介电常数扰动**: 大尺度趋势（深度渐变）+ 中尺度层理（10-50cm）+ 小尺度随机噪声（cm级）
- 扰动幅度应达到 **Δε_r ≈ 1-3** 才能产生可分辨的反射
- 使用非高斯随机场（如截断高斯、指数相关函数）来避免过于平滑的纹理

---

### 模式 2: 纹理过于规则 — "Over-Regular Stripes"

**典型表现**:
- 增益后出现大量完全平行、等间距的水平条纹
- 条纹间距均匀，横向连续性完美，没有任何中断或合并
- 看起来像数字信号处理伪影或人工层理，完全不像自然地质

**受影响案例**: FORMAL06E (最严重), FORMAL09C_P1 (偏此方向)

**物理根因**:
- 层理生成使用了固定的层间距（如每N个网格层一个界面）
- 介电常数变化的横向相关性长度设置过大（>> 测线长度），导致完全横向连续
- 或者使用了确定性函数（如正弦函数）生成层理起伏，缺乏随机性

**可量化诊断指标**:
```python
def check_over_regular_stripes(bscan_tpower, time_ns_window=(50, 400)):
    """
    检测纹理是否过于规则（平行条纹）。
    """
    region = bscan_tpower[time_ns_window[0]:time_ns_window[1], :]
    
    # 指标1: 自相关函数（检测周期性）
    avg_trace = np.mean(region, axis=1)
    autocorr = np.correlate(avg_trace, avg_trace, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    
    # 如果自相关在多个滞后处有明显峰值，说明有周期性
    peaks = find_peaks(autocorr[10:], height=0.3*np.max(autocorr))[0]
    if len(peaks) >= 3:  # 多个等间距峰值
        return "REJECT: Over-regular periodic stripes (n_peaks={})".format(len(peaks))
    
    # 指标2: 横向断裂度（检测层理的横向不连续性）
    # 计算相邻道之间的相关系数分布
    correlations = []
    for i in range(region.shape[1] - 1):
        corr = np.corrcoef(region[:, i], region[:, i+1])[0, 1]
        correlations.append(corr)
    
    # 如果所有相邻道相关系数都>0.95，说明横向过于连续
    if np.min(correlations) > 0.95:
        return "REJECT: Excessive lateral continuity (min_corr={:.3f})".format(np.min(correlations))
    
    return "PASS"
```

**修复策略**:
- 层间距应从**随机分布**中采样（如指数分布、对数正态分布），而非固定值
- 引入**层理中断机制**: 每层有概率在某处"消失"或"合并"，模拟真实沉积中的侵蚀面
- 横向相关性长度应**远小于**单条B-scan的横向范围（如相关性长度 = 5-15m，B-scan长度 = 50-200m）
- 层理厚度应随深度变化（浅层更薄更密，深层更厚更疏）

---

### 模式 3: 基岩反射模糊发散 — "Blurry Basal Reflection"

**典型表现**:
- 基岩界面的反射不是一个清晰的波组（1-3个相位），而是发散成一片模糊的能量
- 波组特征不清晰，难以辨认双程走时
- 可能伴随异常的衍射尾迹

**受影响案例**: FORMAL06H, IV2_F03

**物理根因**:
- **波源脉冲过宽**: 时域波形的时间宽度超过了基岩界面的分辨能力
- **频带不匹配**: 波源中心频率与天线设计频率不匹配，导致脉冲拖尾
- **过渡带过厚**: 覆盖层-基岩过渡带的厚度超过了1/4波长，导致反射能量分散
- **网格分辨率不足**: 空间网格步长大于1/10波长，导致数值色散

**可量化诊断指标**:
```python
def check_basal_reflection_quality(bscan_raw, basal_time_range_ns):
    """
    检查基岩反射的清晰度。
    """
    t0, t1 = basal_time_range_ns
    basal_region = bscan_raw[t0:t1, :]
    
    # 指标1: 峰值锐度（波峰与两侧谷值的比值）
    avg_trace = np.mean(basal_region, axis=1)
    peaks = find_peaks(np.abs(avg_trace), distance=5)[0]
    if len(peaks) == 0:
        return "REJECT: No detectable basal reflection"
    
    # 主峰的半峰宽
    main_peak_idx = peaks[np.argmax(np.abs(avg_trace[peaks]))]
    peak_val = avg_trace[main_peak_idx]
    half_max = np.abs(peak_val) / 2
    # 找到半峰宽
    left = main_peak_idx
    while left > 0 and np.abs(avg_trace[left]) > half_max:
        left -= 1
    right = main_peak_idx
    while right < len(avg_trace)-1 and np.abs(avg_trace[right]) > half_max:
        right += 1
    fwhm = right - left
    
    if fwhm > 15:  # 半峰宽超过15个样本（约21ns @ 0.72 samples/ns）
        return "REJECT: Blurry basal reflection (FWHM={} samples)".format(fwhm)
    
    # 指标2: 反射能量集中度（峰值能量 vs 区域内总能量）
    peak_energy = np.abs(peak_val)**2
    total_energy = np.sum(avg_trace**2)
    if peak_energy / total_energy < 0.3:
        return "REJECT: Diffuse basal reflection (peak_ratio={:.2f})".format(peak_energy/total_energy)
    
    return "PASS"
```

**修复策略**:
- 波源时域波形应使用**Ricker子波或高斯脉冲**，并确保脉冲宽度 < 20ns（对于100MHz天线）
- 覆盖层-基岩过渡带厚度应**小于 λ/4**（约 0.5m @ ε_r=9, f=100MHz）
- 空间网格步长应满足 **dx < λ_min/10**（约 0.075m @ ε_r=9, f=200MHz）
- 如果使用gprMax，检查 PML 边界条件和 CFL 稳定性条件

---

### 模式 4: 采集/地形伪影 — "Acquisition Artifacts"

**典型表现**:
- Raw 面板中出现异常的强振幅事件（如一条横跨所有道的亮线）
- P99.5 显示增益值比正常案例高出一个数量级（如 3.8e-02 vs 正常的 3.8e-03）
- 增益后面板被局部强事件"占据"，其他结构被压缩不可见

**受影响案例**: FORMAL06G (最典型)

**物理根因**:
- **地形起伏未正确处理**: 地表起伏导致直达波到达时间变化，但后处理未做地形校正
- **飞行高度变化**: 无人机载GPR的天线离地高度变化导致耦合不一致
- **近场效应**: 某些道的天线位置过于接近地表或地下结构，产生近场耦合异常
- **数值伪影**: gprMax 中的 PML 反射、网格不匹配等

**可量化诊断指标**:
```python
def check_amplitude_anomalies(bscan_raw, max_acceptable_gain_ratio=3.0):
    """
    检测振幅异常。
    """
    p99 = np.percentile(np.abs(bscan_raw), 99.5)
    p50 = np.percentile(np.abs(bscan_raw), 50)
    
    # 指标1: P99.5 vs 中位数的比值（动态范围合理性）
    dynamic_range = p99 / (p50 + 1e-12)
    
    # 正常案例的动态范围通常在 100-500 之间
    if dynamic_range > 2000:
        return "REJECT: Extreme dynamic range ({:.0f}), likely artifact".format(dynamic_range)
    
    # 指标2: 局部能量集中度（检测单一强事件）
    trace_max = np.max(np.abs(bscan_raw), axis=0)
    global_max = np.max(trace_max)
    median_max = np.median(trace_max)
    if global_max > median_max * max_acceptable_gain_ratio:
        return "REJECT: Local amplitude spike (ratio={:.1f}x median)".format(global_max/median_max)
    
    # 指标3: P99.5增益值合理性（与历史正常案例对比）
    # 正常案例的P99.5 raw增益通常在 100-200 之间
    if p99 > 500:
        return "REJECT: Abnormally high P99.5 raw amplitude ({:.1f})".format(p99)
    
    return "PASS"
```

**修复策略**:
- 地形变化应**平滑**（避免突变），天线离地高度应保持恒定或缓慢变化
- 如果必须模拟地形起伏，应在后处理阶段做**地形校正**（将每道的时间轴平移以补偿地形）
- 检查 gprMax 输入文件中的 PML 厚度（建议 10-20 层）
- 使用 **dx = dy = dz** 的均匀网格，避免网格各向异性

---

### 模式 5: 振幅尺度异常 — "Amplitude Scale Mismatch"

**典型表现**:
- 与同类案例相比，P99.5 显示增益值偏离一个数量级以上
- 这不一定是错误，但如果与其他参数不一致，可能是物理模型参数设置错误

**可量化诊断指标**:
- 建立每个仿真系列的**基准P99.5范围**（如 FORMAL06 系列: 150-200）
- 新案例的P99.5超出该范围 ±50% 时发出警告

---

## 3. 成功案例的共性特征

| 特征 | FORMAL06C | FORMAL08A | FORMAL08B | FORMAL09C-P2 |
|------|-----------|-----------|-----------|--------------|
| 背景纹理 | 微弱但存在 | 丰富不规则 | 丰富+深层纹理 | 稀疏不规则 |
| 纹理规则性 | 低 | 低 | 低 | 低 |
| 基岩清晰度 | 清晰 | 清晰 | 清晰 | 清晰 |
| P99.5 raw | 187 | 187 | 187 | 187 |
| P99.5 tpower | 3.9e-03 | 3.8e-03 | 3.8e-03 | 2.1e-03 |
| 状态 | 可接受 | 推荐 | 推荐 | 有潜力 |

**成功共性**:
1. **基岩反射始终清晰**（FWHM < 15 samples, 峰值能量比 > 30%）
2. **背景纹理存在**（std > 0.001 in 50-400ns window）
3. **纹理不规则**（自相关无周期性峰值，相邻道相关系数有变化）
4. **振幅尺度一致**（P99.5 raw 在 150-200 范围，P99.5 tpower 在 2e-03 ~ 4e-03 范围）

---

## 4. 批量仿真质量控制流水线

建议将以下检查集成到仿真后处理流程中，实现自动筛选：

```
仿真输出 → [原始振幅检查] → [背景纹理检查] → [基岩反射检查] → [规则性检查]
              ↓                    ↓                  ↓                ↓
         P99.5范围校验      std & freq_energy    FWHM & peak_ratio   autocorr & corr
              ↓                    ↓                  ↓                ↓
         PASS / REJECT       PASS / REJECT      PASS / REJECT    PASS / REJECT
              ↓                    ↓                  ↓                ↓
              └────────────────────┴──────────────────┴────────────────┘
                                     ↓
                              全部PASS → 进入数据集
                              任一REJECT → 标记为失败，记录原因
```

### 4.1 推荐的质量控制阈值（基于本批案例分析）

| 检查项 | 通过阈值 | 警告阈值 | 拒绝阈值 |
|--------|----------|----------|----------|
| P99.5 raw 振幅 | 100-250 | 50-100 或 250-400 | < 50 或 > 500 |
| P99.5 tpower 增益 | 1e-03 ~ 6e-03 | 6e-03 ~ 1e-02 | > 1e-02 或 < 5e-04 |
| 背景区域 std | > 5e-04 | 2e-04 ~ 5e-04 | < 2e-04 |
| 背景频率能量 | > 5e-03 | 1e-03 ~ 5e-03 | < 1e-03 |
| 基岩 FWHM | < 12 samples | 12-20 | > 20 |
| 基岩峰值能量比 | > 0.30 | 0.15-0.30 | < 0.15 |
| 相邻道最小相关系数 | < 0.95 | 0.90-0.95 | > 0.98 |
| 自相关周期性峰值 | < 2 | 2-3 | > 3 |
| 动态范围 (P99.5/P50) | 100-800 | 800-1500 | > 2000 |

### 4.2 批量生成策略

**Step 1: 参数空间采样**
- 覆盖层厚度: 均匀采样 3m, 5m, 8m, 12m, 15m, 20m
- 基岩深度: 均匀采样 5m, 8m, 12m, 18m, 25m
- 层理密度: 稀疏(λ/2间距) / 中等(λ/4) / 密集(λ/8)
- 层理规则性: 高( correlated length = line length) / 中( = 1/3 line) / 低( = 1/10 line)
- 介电常数扰动幅度: Δε_r = 0.5, 1.0, 2.0, 3.0

**Step 2: 小规模 smoke 测试**
- 每个参数组合先生成 1-2 条 B-scan
- 应用上述质量控制检查
- 仅通过所有检查的组合进入全量生成

**Step 3: 全量生成**
- 每个通过的组合生成 32-128 条 B-scan
- 每条 B-scan 随机变化: 基岩形态、层理随机种子、局部异常体位置

**Step 4: 最终质量审查**
- 从每个组合中随机抽取 5% 做人工视觉审计
- 统计每个组合的不合格率，淘汰整体不合格率 > 10% 的组合

---

## 5. 关键经验法则

1. **"空白背景 = 失败"**: 如果 tpower 增益后 50-400ns 区域的标准差 < 0.0002，该仿真在物理上不真实，不应进入训练集。

2. **"完美平行线 = 失败"**: 如果层理在横向上的相关系数 > 0.98，说明层理过于规则，缺乏真实地质的随机中断。

3. **"基岩必须可辨"**: 基岩反射的半峰宽如果超过 15 个样本（~20ns），说明波源或过渡带设计有问题。

4. **"P99.5 是早期预警信号"**: 如果 P99.5 增益值与同类案例偏离 > 50%，应先检查物理参数而非直接接受。

5. **"多层次异质性是关键"**: 成功的覆盖层模型必须同时包含 (a) 深度相关的趋势，(b) 中等密度的随机层理，(c) cm 级的小尺度扰动。仅有一层是不够的。

6. **"真实地质从不完美横向连续"**: 任何横向连续性长度 > 50% 测线长度的层理模型都会产生人工感。

---

## 附录：生成的分析文件

| 文件 | 说明 |
|------|------|
| `failure_mode_gallery.png` | 8个案例的 tpower 面板对比（4 rejected + 2 good + 2 acceptable） |
| `failure_vs_success_comparison.png` | 失败模式 vs 成功模式的 4x2 并排对比 |

