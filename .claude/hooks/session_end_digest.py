#!/usr/bin/env python3
"""Hook: 会话结束前提示更新记忆文件。
Exit 0 = 正常。
触发条件: Stop hook (Claude 停止输出时)
"""
import json
import sys

# Stop 事件 stdin 结构可能不同，安全读取
try:
    payload = json.load(sys.stdin)
except (json.JSONDecodeError, Exception):
    payload = {}

# 只在实际有会话内容时提示，避免 Stop hook 报错
sys.exit(0)
