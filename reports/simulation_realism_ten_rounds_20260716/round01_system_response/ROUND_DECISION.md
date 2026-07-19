# 第 1 轮：有效系统响应

## 决策

```text
PASS_AS_OPTIONAL_SYSTEM_RESPONSE_COMPONENT
REJECT_AS_STANDALONE_REALISM_SOLUTION
SELECT strength_1.00
PROCEED_TO_ROUND_02
```

第 1 轮只从 Line3、Line7、LineL1 的强标签区估计目标波包的等测线权重频谱包络，用 Line6 选择响应强度。估计器不读取实测相位，不复制波形片段，响应限制在 20-180 MHz、最大绝对增益 6 dB，并采用零相位滤波，因此不应移动标签。Line9 在候选 `strength_1.00` 冻结后才打开。

## 数值结果

| 指标 | FORMAL06C | strength 1.00 | 变化 |
|---|---:|---:|---:|
| 折内目标谱形 RMSE | 4.4045 dB | 2.9687 dB | -32.6% |
| Line6 谱形 RMSE | 5.5957 dB | 3.7745 dB | -32.5% |
| 冻结后 Line9 谱形 RMSE | 4.5827 dB | 3.4710 dB | -24.3% |
| target/background RMS | 10.3003 | 9.9787 | -3.1% |
| envelope CV | 0.4688 | 0.4686 | 基本不变 |
| dropout | 0 | 0 | 保持 |
| 显著波瓣数 | 8 | 8 | 保持 |
| 主峰频率 | 79.37 MHz | 88.18 MHz | 向折内实测包络移动 |

## 视觉审计

- 四个盲候选都保留了 FORMAL06C 的连续多周期基覆波包，没有折断、硬 dropout 或标签定向增强。
- `strength_1.00` 的频率形态更接近拟合折和 Line6，但 B-scan 的宏观形态变化很小。
- 与 Line3、Line7、LineL1、Line6 以及冻结后 Line9 对照时，候选仍显著过于平滑、规则、目标占优；实测包含更强的局部相干事件、振幅变化和多尺度背景。
- 因此频率响应是一个真实缺失因素，但不是主要形态差距的来源。

## 论文价值

该因素可作为最终 `physics -> measurement` 采集算子的确定性子模块，并形成“无系统响应/有系统响应”的清晰消融。它不能单独支撑“逼真实测”的主张。

## 下一轮

第 2 轮只加入平滑逐道增益和亚波长时延漂移。参数从拟合折的强标签包络、飞高和标签校正后残差中估计；Line6 选择，Line9 保持关闭直到冻结。
