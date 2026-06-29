---
name: training-data-validator
description: 训练数据验证——NPZ 键完整性、padding 区清零、y_soft 与 x_raw 形状匹配、P99 一致性、status_code 分布合理性。训练前调用。
region: us
---

# Training Data Validator

验证 convert_pilot_to_training.py 输出的 NPZ 文件。

## 检查项
1. **NPZ 键** — 存在 x_raw, y_mask, y_soft, status_code, label_weight, label_weight_2d
2. **形状** — x_raw(501,256), y_mask(501,256), y_soft(501,256), weight_2d(501,256), status(256,)
3. **Padding** — y_mask[:,:64]==0, y_mask[:,192:]==0, label_weight[:64]==0, label_weight[192:]==0
4. **y_soft** — 峰值 1.0，padding 区 0
5. **x_raw** — finite, 幅值 P99 归一化
6. **status_code** — absent/present/weak 分布合理
