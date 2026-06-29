# MODEL_PLAN_SELECTION_AUDIT - v0.7.25

## 背景

用户在 v0.7.24 验证“生成一批模型”和“加载模型图库”后提出：一键生成模型需要支持选择模型配置。

原有易用界面已经有“仿真计划”路径输入，但普通用户不容易知道可选 YAML 有哪些，也不容易判断各配置的规模、道数、网格精度和深度范围。

## 本轮修改

### 新增模型配置发现服务

`src/uavgpr_simlab/services/project_service.py` 新增：

- `ModelPlanPreset`
- `discover_model_plan_presets(root_dir)`

只扫描：

```text
configs/run_plan*.yaml
configs/run_plan*.yml
```

不把材料表、ML 配置、自动化 pipeline 配置混入一键模型生成配置。

### 项目管理页新增模型配置下拉框

`src/uavgpr_simlab/gui/pages/project_page.py` 新增“模型配置”选择框，显示：

```text
plan_name | 默认场景数 | trace_count | dx | domain_depth | 文件名
```

选择后自动回填“仿真计划”路径。

### 主窗口状态协调

`src/uavgpr_simlab/gui/easy_window.py` 新增：

- `self.plan_presets`
- `select_model_plan_preset()`

切换模型配置时会：

- 更新 `plan_edit`
- 清空旧 manifest 路径
- 清空批量页 manifest 路径
- 自动刷新计划预览
- 在状态栏提示当前选择的模型配置

模型生成仍调用原服务：

```text
generate_model_batch(plan_edit, workspace_edit, case_count)
```

## 边界

本轮不改变：

- `core.scenario.generate_cases()` 语义
- run_plan YAML 字段含义
- manifest 结构
- gprMax 调用方式
- fingerprint / marker
- B-scan 后处理

## 验证

已验证：

- `python -m compileall -q src scripts`
- `PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/self_test.py`
- `PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/make_v070_product_screenshots.py`
- `PYTHONPATH=src python scripts/smoke_gprmax_source.py --gprmax-root /mnt/data/_gprmax_v317_read/gprMax-v.3.1.7 --work-dir workspace/gprmax_source_smoke_v0725 --omp-threads 1 --timeout 180`

自测新增断言：

- 易用项目页至少发现 1 个模型配置
- 当前模型配置下拉框有有效路径
- 服务层能发现 `run_plan_3060_quick.yaml`
- 服务层只暴露 `run_plan*` YAML

## 后续建议

- 根据真实使用经验补充更多业务化配置名称，例如“快速预览”“4090 高精度验证”“论文主数据集”。
- 后续可在模型预览页顶部只读显示当前配置名，进一步降低用户找不到配置来源的问题。
