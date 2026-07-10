# MyNet 精简源码审计包

该包是本地提交 `7a250a7` 的干净源码快照，包含 origin/master 之后 10 个本地未推送提交的最终文件状态。

## 包含
- `pgdacsnet/` 网络、loss、接口与训练合同代码
- `scripts/` 数据导入、V15 生成、校验、训练/评估脚本
- `configs/` 已冻结和修复的实验配置
- `tests/` 合同、V15、模型接口与评估测试
- `data/dataset_contract_v2/` 轻量治理清单
- `reports/` 关键 MD/CSV/JSON 审计结果（不含大图）
- `docs/`、依赖文件、Claude/Codex 辅助说明
- `AUDIT_HANDOFF/` 本地提交状态、差异统计、源码补丁、审计提示词

## 有意不包含
- `.git`
- 历史仿真大数据和旧 accepted cases
- V14/V15 candidate 重复数据
- V15-final 二进制数据（单独数据包提供）
- 原始 PDF/CSV/剖面证据（单独证据包提供）
- checkpoints、outputs、logs、workspace、缓存和临时图片

## 组合方式
1. 解压本源码包。
2. 将 `MyNet_V15最终标签数据包_7a250a7.zip` 解压到 `MyNet/` 根目录。
3. 原始证据包仅供审计核对，不应放进训练数据目录。

## 状态边界
V15-final 标签已落盘，但 formal training 仍冻结：缺真实负样本、缺非 Line9 条件化获批仿真、正式 split 尚未完全放行。
