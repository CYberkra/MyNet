# UavGPR-SimLab v0.8.0-alpha.9 ultra tiny 几何修复审计

## 背景

用户在 Windows 目标机通过 GUI 运行 `yingshan_sceneworld_ultra_tiny_v080a8`。日志显示：

- `raw` 成功；
- `target_only` 成功；
- `background_only` 成功；
- `air_only` 成功；
- `clutter_only` 失败。

失败来自 gprMax 对 `clutter_only.in` 中一条 `#box` 的几何校验：

```text
#box: 0 9.2965 0 1 9.3465 0.25 silty_clay
```

该盒体在 y 方向厚度约 0.05 m，而 ultra tiny 网格为 0.25 m。gprMax 在网格化后会把这种 sub-cell 几何判定为非法盒体。

## 根因

`scene_variant_writer.py` 中 `clutter_only` 为排除目标界面，只保留了 0.05 m 的 surface proxy：

```python
#box: x0 gy-0.05 0 x1 gy domain_z cover
```

这对 0.25 m 网格不安全。

## 修复

1. 新增 `_safe_box_line()`：
   - 坐标裁剪到 FDTD domain；
   - 保证 x/y/z 坐标严格递增；
   - 对薄于网格的盒体扩展至至少一个网格单元。

2. 新增 `validate_gprmax_box_lines()`：
   - 生成 `.in` 时检查所有 `#box`；
   - 对非法或 sub-cell 盒体提前报错，避免目标机运行时才失败。

3. `clutter_only` surface proxy 改为一格厚度：

```text
surface_thickness = max(dx_m, 0.05)
```

4. 重新生成：

```text
workspace/yingshan_sceneworld_ultra_tiny_v080a9
```

5. GUI 的“运行 ultra tiny 全链路验证”默认使用 v080a9 骨架。

## 不改变内容

- 不改变 gprMax 源码目录自举方式；
- 不改变 5 variant 定义；
- 不改变 `constant_level` 轨迹说明；
- 不改变正式 smoke/pilot 的严格 QC；
- 不把 ultra tiny 当训练数据。

## 验证

已完成：

```bash
python -m compileall -q src scripts
PYTHONPATH=src python scripts/run_all_gprmax.py --help
```

已完成 v080a9 `.in` 预检：

```text
raw.in: 0 problems
target_only.in: 0 problems
background_only.in: 0 problems
clutter_only.in: 0 problems
air_only.in: 0 problems
```

当前 sandbox 未完整运行 gprMax 5 variant；目标机已证明 gprMax 环境可运行，用户需用 v080a9 复测。
