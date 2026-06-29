# MODEL_GALLERY_BUGFIX_AUDIT - v0.7.24

## 背景

用户在 Windows 本地点击“生成一批模型”时报错：

```text
TypeError: 'WindowsPath' object is not iterable
```

直接原因是 `core.scenario.generate_cases()` 返回值为 `(models_dir, manifest_path)`，但 `services/project_service.generate_model_batch()` 将第一个返回值误当成可迭代模型集合处理。

同时，用户点击“加载模型图库”时，如果模型清单输入框为空，旧逻辑会直接静默返回，造成“没反应”的体验问题。

## 修复内容

1. `generate_model_batch()` 改为明确接收 `model_root, manifest`。
2. 模型数量改为从 manifest 中按唯一 `case_id` 统计。
3. 新增 `find_latest_manifest()`，支持从以下位置自动发现模型清单：
   - `<workspace>/datasets/<workspace-name>_manifest.csv`
   - `<workspace>/datasets/*manifest*.csv`
   - `<workspace>/*/datasets/*manifest*.csv`
4. “加载模型图库”在未填写 manifest 时会自动查找最近清单；仍找不到时给出明确中文提示。
5. 生成模型后自动回填模型预览页和批量仿真页的 manifest 路径。
6. 修复 `run_gui.bat` 的 Windows 路径和 `PYTHONPATH` 设置方式，避免双击脚本时把路径误解析成命令。

## 未改变内容

- 不改变模型生成算法。
- 不改变 manifest 字段结构。
- 不改变 gprMax 调用语义。
- 不改变任务 fingerprint / marker / B-scan 后处理。
