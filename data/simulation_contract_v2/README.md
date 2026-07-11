# PGDA Simulation Contract V2

本目录是旧 Line9-conditioned 仿真的替代设计入口。

- `simulation_contract_v2.json`：不可绕过的物理、标签和 provenance 合同。
- `materials_v1.json`：用于 control 阶段的非色散材料集合。
- `control_cases_v1.json`：4 个解析/匹配控制场景。
- 生成命令：`python scripts/generate_physical_sim_v2.py --overwrite`
- 静态验证：`python scripts/validate_physical_sim_v2.py`
- gprMax 运行后：`python scripts/postprocess_physical_sim_v2.py <case_dir>`

所有 control 默认 `formal_training_allowed=false`。只有真实 gprMax 输出、matched-control 可见相位、运行后物理验证和人工审阅全部通过后，才能进入 pilot 审批；control 本身不直接晋级正式训练。

`legacy_quarantine.csv` 冻结旧合同中的全部仿真 case；这些 case 可保留作开发回归、smoke 或压力测试，但不能被 V2 exporter 晋级。
