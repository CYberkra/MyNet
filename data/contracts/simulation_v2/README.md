# PGDA Simulation Contract V2

本目录是旧 Line9-conditioned 仿真的替代设计入口。

- `simulation_contract_v2.json`：不可绕过的物理、标签和 provenance 合同。
- `hardware_measurement_contract_v1.json`：有限天线 3D 局部验证的硬件证据门禁；未采集字段时会明确阻断运行。
- `HARDWARE_MEASUREMENT_AND_3D_PREFLIGHT.md`：现场脉冲采集与首个 3D 局部对照协议。
- `materials_v1.json`：用于 control 阶段的非色散材料集合。
- `control_cases_v1.json`：4 个解析/匹配控制场景。
- 生成命令：`python scripts/generate_physical_sim_v2.py --overwrite`
- 静态验证：`python scripts/validate_physical_sim_v2.py`
- gprMax 运行后：`python scripts/postprocess_physical_sim_v2.py <case_dir>`

所有 control 默认 `formal_training_allowed=false`。只有真实 gprMax 输出、matched-control 可见相位、运行后物理验证和人工审阅全部通过后，才能进入 pilot 审批；control 本身不直接晋级正式训练。

## Independent V2 Family 01

`independent_v2_family01_pilot.json` freezes the first non-Line9-conditioned
positive and exact matched true-negative family. Its generator reads only the
contract and seeded generic priors. It does not read measured arrays or any
FORMAL06/07 development geometry.

As of 2026-07-15, the family has passed static and geometry checks, one-trace
runtime equivalence, an 8-trace full/control causal pilot, and a 32-trace
full-scene blind morphology review. It remains blocked from training until all
native 256-trace required runs, immutable evidence packaging, and independent
human release approval are complete.

`independent_v2_family01_r2_pilot.json` is a separate, pre-solver provenance
rebuild. It retains the original generic physics and seeds but writes every
hash-protected text artifact with canonical LF serialization. It never
overwrites the historical Family 01 assets, which remain audit-blocked until
their recorded/raw hash mismatch is resolved as historical evidence.

`legacy_quarantine.csv` 冻结旧合同中的全部仿真 case；这些 case 可保留作开发回归、smoke 或压力测试，但不能被 V2 exporter 晋级。

旧 V1 case 的唯一去重、标签 override 与 trace-level ignore/weak 状态由
`data/simulation_governance_v1_20260711/` 管理。该 catalog 只引用原始证据，
不复制或重写 raw 波形；它明确禁止历史 `y_soft` 作为曲线目标，且所有 V1
case 仍禁止进入正式 Line9 holdout 训练。
