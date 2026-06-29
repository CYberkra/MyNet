# ENV_DIAGNOSTICS_AUDIT - 设置页环境诊断文案增强

## 本轮目标

提高“设置与帮助”页环境检查结果的可读性。此前检查结果主要输出 JSON，对普通用户不够直接。本轮新增面向用户的中文诊断摘要，同时保留原始 JSON 报告，便于后续排错和开发审计。

## 变更范围

```text
src/uavgpr_simlab/services/environment_service.py
src/uavgpr_simlab/gui/easy_window.py
scripts/self_test.py
```

## 新增能力

- `format_easy_environment_report()`：把环境检查结果转成中文摘要；
- 常见失败项的建议文案，包括：
  - conda 未找到；
  - conda 环境不存在或不可进入；
  - gprMax 无法导入；
  - nvidia-smi 未找到；
  - nvcc 未找到；
  - PySide6 或其他 Python 模块缺失；
  - gprMax 源码目录结构不完整；
- 设置页日志区现在先显示摘要，再显示原始 JSON。

## 边界

本轮只增强诊断显示，不改变：

- gprMax 命令构造；
- conda run 调用语义；
- GPU / OpenMP 参数传递；
- 任务 fingerprint / marker；
- B-scan 后处理。

## 验证要点

- `format_easy_environment_report()` 能稳定生成包含“诊断摘要”和“目标机 smoke test 建议顺序”的文本；
- 原始 JSON 仍保留；
- `scripts/self_test.py` 已覆盖格式化输出的基本断言。
