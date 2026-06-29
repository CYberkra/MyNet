# 面向论文的仿真-实测闭环流程

1. 3060 上运行 quick smoke test：小网格、小道数、geometry-only。
2. 检查 `.in` 文件、`scene_meta.json`、`interface_label.csv` 是否合理。
3. 在 GUI 队列页只跑 `raw` 的少量样本，确认 gprMax 输出。
4. 再跑 `target_only`、`clutter_only`、`background_only`，构建可解释监督标签。
5. 上传/选择实测 CSV，在“实测数据”页生成 NPZ/PNG 和传统方法基线。
6. 4090 上正式跑 500+ 场景，保留随机种子和环境报告。
7. 用 `configs/ml_pgda_csnet.yaml` 进入 PGDA-CSNet 训练。
8. 论文评价必须包含：仿真标签指标、实测 SNR/CNR、钻孔误差、交叉测线一致性、RTM/FWI 下游改善。


## v0.3 自动化推荐补充

- 使用 `configs/pipeline_automation_template.yaml` 作为论文流水线入口，先小规模验证，再放大到 pilot/main/hard-case。
- 钻孔弱监督不再手工画区域，优先维护 `configs/boreholes_example.csv` 同格式的钻孔表，并运行 `soft-mask` 自动生成保护带。
- 正式批量仿真优先从 manifest 生成 SLURM array 脚本，避免手动复制命令导致漏跑或重复跑。
- 每轮实验结束后运行 `auto-report`，把 manifest 统计、实测 QC、soft mask、缺失产物和论文表格统一归档。
