# Codex 审计指令

请以“当前实现可能仍有错误”为假设，审计本分支，但不要修改 V15 标签、正式 split 或旧仿真隔离状态。

重点输出：

1. P0/P1/P2 问题表，附文件、行号、复现步骤和修复建议。
2. 对 gprMax input 的 domain、PML、guard、源/接收器轨迹、材料 box 连续性逐项验证。
3. 独立重算 CTRL01/02 平层双基地分层到时，并检查 CTRL03 曲面 reference 的命名和用途。
4. 验证 solver time、HDF5 `dt`/`Iterations`/rx length、canonical resampling 之间无 off-by-one 或伪覆盖。
5. 验证 CTRL02/CTRL04 的上覆几何、采集几何、材料和数组匹配；确认负样本目标 mask 为零。
6. 检查 full/no-basal/air 差分是否被错误解释为 component ground truth。
7. 检查 Line9 泄漏扫描是否覆盖文件名、manifest、数组来源和统计量。
8. 检查 export 路径能否绕过 `formal_training_allowed=false` 或 `manual_approval_required`。
9. 在无 gprMax 的环境中确认脚本明确报告 blocked，不产生伪 HDF5 或成功状态。
10. 运行 `CODEX_NONPAPER_HANDOFF/README_FIRST.md` 中的测试命令并记录结果。

明确禁止：

- 把静态通过写成 FDTD 已通过。
- 把 CTRL04 写成已批准正式负样本。
- 启动 24-case pilot 或正式训练。
- 引入论文网络、Mamba、query decoder 或架构调研内容。
