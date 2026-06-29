# UavGPR-SimLab v0.5.2：历史仿真可视化增强说明

## 本版目标

v0.5.2 针对“历史仿真记录页不够直观”的问题进行了增强：历史页不再只是状态表，而是一个可视化复盘工作台。用户可以直接看到每条任务对应的模型几何缩略图、已完成 B-scan 缩略图；对于正在运行的任务，界面会周期性读取已经写出的 `.out` 道文件，并自动合成为当前可用的 B-scan 预览。

## 新增能力

1. **历史记录可视化表格**
   - 每行显示：状态、模型视图、B-scan、已读 trace 数、时间、case_id、variant、n_traces、geometry 标记、returncode、job_id。
   - 支持筛选：全部、running、done、failed、geometry-only、full simulation。
   - 支持限制最多显示数量，默认 300 条，避免大型论文任务一次性渲染过多缩略图。

2. **模型几何缩略图**
   - 从 `datasets/<case_id>_labels.json` 自动解析地表、基覆界面、UAV 高度。
   - 不需要跑 gprMax 即可生成预览。
   - 缩略图缓存到：`<workspace>/previews/history/`。

3. **已完成 B-scan 缩略图**
   - 优先读取 postprocess 生成的 raw `.npz`。
   - 如果没有 `.npz`，会尝试直接读取对应 `.in` 旁边的 gprMax `.out` 文件。

4. **运行中实时 B-scan**
   - v0.5.2 新增 `jobs/running/*.json` 运行中 marker。
   - GUI 历史页开启“运行中自动刷新”后，会每 2.5 秒刷新选中记录，并每约 8 秒刷新一次表格。
   - 对正在运行的任务，会读取当前已经写出的可用 `.out` 文件，自动合成实时 B-scan。
   - trace 列会显示当前已经读到的道数。

5. **CLI 预览命令**

```bash
PYTHONPATH=src python -m uavgpr_simlab.cli history-preview \
  --workspace workspace/paper_main_simulation \
  --status running \
  --limit 20
```

该命令会返回每条记录的模型预览图、B-scan 预览图、已读 trace 数和路径信息。

6. **安全删除增强**
   - 删除历史记录并勾选删除输出时，会同时清理该 job 的历史缩略图。
   - 删除仍然限制在当前 workspace 内，避免误删外部文件。

## 使用建议

- 正式批量前：先在“3 3D预览”页检查 case 的几何是否合理。
- 批量运行时：在“5 批量运行”页启动任务，同时打开“6 历史记录”页筛选 `running`。
- 运行中：选中某个 running 任务，右侧会显示该任务的模型和当前已合成的 B-scan。
- 跑完后：筛选 `done`，查看模型/B-scan 对应关系；发现异常任务可直接删除记录和输出。

## 限制说明

- 实时 B-scan 依赖 gprMax 在运行过程中陆续写出可读 `.out` 文件。若某些 gprMax 版本或运行模式在任务结束前不释放可读 HDF5 文件，GUI 会显示“等待 .out”，任务结束后再显示完整 B-scan。
- `geometry-only` 不产生波形，因此历史页只显示模型视图，不显示 B-scan。
- 大规模历史记录建议关闭“显示模型/B-scan缩略图”或限制显示数量，以减少刷新开销。
