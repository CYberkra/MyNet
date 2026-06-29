# GPRMAX_SOURCE_DIAGNOSTICS_AUDIT - v0.7.4

## 本轮目标

用户提供了 `gprMax-v.3.1.7.zip` 作为辅助材料。本轮只读审视该源码包，并把可稳定复用的“源码目录结构诊断”固化到 UavGPR-SimLab 服务层中。

本轮不修改 gprMax 源码，不复制 gprMax 到项目正式目录，也不改变 UavGPR-SimLab 对 gprMax 的命令行调用语义。

## 已确认的 gprMax 源码结构

解压后的源码树包含以下关键文件或目录：

```text
gprMax/__init__.py
gprMax/__main__.py
gprMax/gprMax.py
gprMax/_version.py
setup.py
conda_env.yml
requirements.txt
README.rst
user_models/
tools/
tests/
```

其中 `gprMax/__main__.py` 会调用 `gprMax.gprMax.main()`，因此 UavGPR-SimLab 现有的 `python -m gprMax <input_file>` 调用形式与源码入口匹配。

需要注意：压缩包目录名为 `gprMax-v.3.1.7`，但内部 `gprMax/_version.py` 标识为 `3.1.6`。后续验收时不要仅凭压缩包文件名判断安装版本。

## 新增服务层

### `services/environment_service.py`

职责：

- 读取和保存 `.simlab_env` 中的易用界面环境设置；
- 统一组装实时批量运行所需 `AppConfig`；
- 结构化检查 gprMax 源码目录是否有效；
- 运行环境检查并输出包含 `gprmax_source` 的扩展报告。

有效 gprMax 源码根目录至少应包含：

```text
gprMax/__init__.py
gprMax/__main__.py
gprMax/gprMax.py
setup.py
```

可选但有助于诊断的文件包括：

```text
conda_env.yml
requirements.txt
README.rst
user_models/cylinder_Ascan_2D.in
```

### `services/project_service.py`

职责：

- 读取项目计划 YAML 供项目管理页预览；
- 从计划和工作目录生成模型批次；
- 返回 manifest 和 GUI 可直接使用的行数据。

## GUI 层变化

`easy_window.py` 不再直接承担以下职责：

- `read_simlab_env` / `write_simlab_env`；
- `run_environment_checks` / 环境报告保存；
- `_cfg_from_plan` 到运行配置的组装；
- `load_yaml` 计划预览；
- `generate_cases` 模型批次生成。

这些职责已迁移到服务层，GUI 只负责读取控件值、调用服务和显示结果。

## 验证记录

本轮新增自测项：

```text
Easy project/environment services
```

它会构造一个最小 fake gprMax 源码树，用于验证源码结构诊断、计划预览和运行配置组装，不依赖真实 CUDA 或 gprMax 编译环境。

本轮也对用户提供的 gprMax 源码包执行了只读结构诊断，结果为：

```text
is_source_tree: true
detected_version: 3.1.6
user_model_count: 11
```

## 风险边界

- 结构诊断通过，只表示路径像 gprMax 源码根目录；不表示 gprMax 已完成编译或可在目标 conda 环境导入。
- 当前 Linux sandbox 不能替代 Windows + CUDA + GPU + conda 的真实求解验证。
- 真实 P1 验证仍需在目标机器执行少量 geometry-only 和 full raw 仿真。
