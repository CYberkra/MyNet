---
name: sim-qc
description: 仿真质量检查——PML 填充、波形一致性、标签对齐、批量异常检测。Use when user says "仿真QC", "仿真质量", "simulation quality check".
---
# sim-qc: 仿真质量检查

## 使用方式
```
/sim-qc outputs/some_run/
/sim-qc data/simulation_pretrain_v2/
/sim-qc --latest
```

## 检查项目

### 1. B-scan 质量
- 信号幅度范围合理（非全零、非饱和）
- 直达波位置一致
- 地表反射清晰可辨

### 2. PML 边界检查
- 底部 PML 无信号泄漏
- 侧面 PML 填充正确

### 3. 标签对齐
- 基岩面标签与 B-scan 中的反射层对应
- 标签深度范围在场景几何范围内
- 无漂移的标签

### 4. 批量异常检测
- 场景间信号幅度一致性
- 噪声水平异常
- 几何参数变化范围

### 5. 波形验证（如有 source waveform）
- 中心频率正确（100 MHz）
- 无截断或失真

## 输出格式
```
🔍 仿真 QC 报告: simulation_pretrain_v2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
场景数: 60
✅ B-scan 质量: 58/60 通过
⚠️ 异常:
  - case_000023: 信号幅度偏低 (max=0.001, 预期>0.01)
  - case_000041: 标签偏移 3 traces
✅ PML: 无泄漏
✅ 标签对齐: 59/60 通过
📊 信号统计: mean=0.045, std=0.012, range=[0.001, 0.15]
```
