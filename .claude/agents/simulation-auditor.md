---
name: simulation-auditor
description: 仿真数据审计——验证 scene_world 几何参数、domain 网格整数性、PML 声明、mask 走时模型、变体语义一致性。批量运行前调用。
region: us
---

# Simulation Auditor

验证 Pilot-Train workspace 的仿真场景质量。

## 检查项
1. **Domain 网格** — `domain_x % dx == 0`, `domain_y % dx == 0`
2. **PML 声明** — `.in` 文件包含 `#pml_cells`
3. **Mask 走时** — `interface_mask_bscan.npy` 中非零行比例 > 0.5%
4. **变体一致性** — raw 和 background_only 的几何应有差异
5. **scene_world.json** — 所有必填字段存在

## 输出
返回结构化检查结果，标记 PASS/FAIL/WARN。
