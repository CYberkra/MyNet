---
name: data-convert
description: 数据格式转换向导——自动选择正确的转换脚本、预检、执行、验证。Use when user says "转换数据", "convert data", "数据格式".
---
# data-convert: 数据格式转换向导

## 使用方式
```
/data-convert
/data-convert --from batch_v1 --to training
/data-convert --from pilot --to pretrain_v2
```

## 可用转换

| 来源 | 目标 | 脚本 | 说明 |
|------|------|------|------|
| batch_v1 gprMax 输出 | training dataset | `scripts/convert_batch_v1_to_training.py` | 旧版仿真批次 → windowed 格式 |
| Pilot gprMax 输出 | training dataset | `scripts/convert_pilot_to_training.py` | Pilot 仿真 → windowed 格式 |
| SimLab SceneWorld | simulation_pretrain | `scripts/convert_sim_to_training.py` | SimLab 输出 → 训练格式 |

## 交互流程

### 1. 确认来源
如果未指定 `--from`，问用户：
- 数据在哪个目录？
- 是什么格式？（gprMax raw output / SimLab SceneWorld / 其他）

### 2. 选择转换脚本
根据来源推荐正确的脚本，显示将执行的命令。

### 3. 预检
- 源目录存在且非空
- 必要的依赖文件（merged.out 等）存在
- 目标目录不存在或确认覆盖

### 4. 执行
运行转换脚本并捕获输出。

### 5. 验证
- 检查输出目录的文件数
- 读取 index CSV 前几行
- 随机加载一个样本验证维度
