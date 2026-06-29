---
name: train-log-analyzer
description: 训练日志分析——loss 曲线诊断、过拟合检测、异常波动识别、中文报告。Use when user says "分析训练日志", "analyze training", "loss 分析".
---
# train-log-analyzer: 训练日志分析 Agent

> 此 skill 包含一个 subagent 定义，用于深度分析训练过程。

## 使用方式
```
/train-log-analyzer outputs/run_gpu_v3_pilot_mixed_loo_Line9_seed1901
```

## 分析维度

### 1. Loss 曲线形态
从 `history.json` 读取 loss 序列，分析：
- **单调递减** → 🟢 正常收敛
- **震荡下降** → 🟡 学习率可能偏大
- **plateau（>20 epoch 无改善）** → 🟡 提前停止或降低学习率
- **上升** → 🔴 过拟合或标签问题
- **突跳** → 🔴 梯度爆炸或数据异常

### 2. Train vs Val Gap
- gap < 0.3: 🟢 欠拟合/泛化好
- gap 0.3-0.8: 🟢 正常范围
- gap > 0.8: 🟡 轻度过拟合
- gap > 1.5: 🔴 严重过拟合

### 3. 收敛速度
- 前 10 epoch loss 下降 > 50%: 🟢 快速学习
- 前 10 epoch loss 下降 < 10%: 🟡 学习率可能偏小
- epoch 1 loss > 2.0: 初始 loss 偏高，检查数据

### 4. 异常检测
- NaN/Inf loss: 🔴 训练失败
- Loss 突然翻倍: 🔴 梯度爆炸
- 连续多个 epoch loss 完全相同: 🔴 可能卡在局部最小值

### 5. 最佳 Epoch 分析
- 最佳 epoch 在前 20%: 🟡 可能训练过长
- 最佳 epoch 在中间 60%: 🟢 理想
- 最佳 epoch 就是最后一个: 🟡 可以继续训练

## 输出格式
```
📊 训练分析报告: Line9 seed1901
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
收敛状态: 🟢 正常收敛
最终 train_loss: 0.509 (epoch 1: 1.400, ↓63.6%)
最佳 epoch: 6 (val_loss=1.059)
Train-Val Gap: ~1.0 (🟡 轻度过拟合，LOLO-CV 正常)

📈 Loss 曲线诊断:
  - 阶段 1 (ep 1-10): 快速下降，loss 1.40→0.75
  - 阶段 2 (ep 10-50): 稳定收敛，loss 0.75→0.55
  - 阶段 3 (ep 50-80): 轻微波动，loss 0.55→0.51

⚠️ 注意事项:
  - Val loss 在 epoch 6 后上升，说明模型开始过拟合到训练线
  - LOLO-CV 中这是正常现象（训练线与验证线完全不重叠）
  - 建议使用 epoch 6 的 best checkpoint 做最终评估

💡 建议:
  - 如需进一步优化，可尝试增加数据增强力度
  - 或增加训练线的多样性（更多仿真场景）
```
