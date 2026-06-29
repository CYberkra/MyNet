# UX_DATASET_RUN_WORKFLOW_AUDIT - v0.8.0-alpha.33

## 目标工作流

长期目标固定为：

```text
设计数据集骨架 → 导入软件 → 合同检查 → 一键启动 GPU 仿真 → 实时看 B-scan / 日志 → 历史复盘 / failed 重跑
```

用户不应在每个软件版本里重新寻找 gprMax、手工改 Python、手工判断哪些 case 跑过。软件应从 manifest / QC / history marker 中自动恢复状态。

## 本轮体验审计结论

### 已实现并继续保留

- 批量页已有统一运行配置、预检、跳过已完成、只重跑 failed、强制重跑。
- SceneWorld manifest 已有五变体合同：raw、target_only、background_only、clutter_only、air_only。
- 历史页已支持 dataset / case / variant 树状查看。
- 双机 GPU runtime 已从软件包中独立出来。

### 本轮发现的体验缺口

1. 导入骨架不够显性：此前用户需要知道 manifest 应该在哪一页选择。
2. “已经跑的、正在跑的、即将跑的”没有在同一个看板里统一解释。
3. 历史页主要偏完成结果，对未运行 pending 行展示不足。
4. 命令行可以检查 skeleton，但缺少一个只看运行状态的 dashboard 命令。
5. GUI 实机若安装 PySide6，会依赖 `__display_version__`，但包内 `__init__.py` 只保留了 `__version__`，这是潜在 P1 启动风险。

## 本轮改进

- 批量页新增“导入数据集骨架”入口。
- 批量页新增数据集运行看板，显示：总任务、即将运行、正在运行、历史完成、失败待处理。
- 导入骨架后自动：设置 workspace、同步模型页、刷新批量预检、刷新历史页。
- 历史页开始展示 `pending` 记录，用户能在一个树里看到未跑、正在跑、已完成和失败。
- 新增 `core/run_dashboard.py`，将运行看板抽离为可测试服务。
- 新增 CLI：`python -m uavgpr_simlab.cli run-dashboard --manifest ... --write-report`。
- 修复 `__display_version__` 缺失风险。

## 后续体验优化建议

1. 增加“数据集骨架向导”：从 YAML plan、真实数据 profile 或外部目录自动生成 manifest skeleton。
2. 批量页增加“暂停后继续”显式按钮，目前继续依赖 skip/completed/failed 合同。
3. 历史页增加右键菜单：打开 case 文件夹、打开 QC JSON、复制失败原因。
4. 运行中 B-scan 可增加最近 3 个 variant 缩略条，便于比较 raw / target / clutter_gt。
5. workspace 迁移工具应进入下一轮：检查绝对路径并批量改为相对路径。
