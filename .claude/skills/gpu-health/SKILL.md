---
name: gpu-health
description: GPU 健康检查——温度、显存、CUDA 进程、热降频风险、后台抢占。Use when user asks "GPU状态", "显卡温度", "gpu health", or training keeps crashing.
---
# gpu-health: GPU 健康快诊

## 使用方式
```
/gpu-health
```

## 检查项目

### 1. GPU 基本状态
```bash
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit --format=csv,noheader
```

输出格式化表格：
| 指标 | 值 | 状态 |
|------|-----|------|
| GPU | RTX 3060 Laptop | - |
| 温度 | XX°C | 🟢<80 🟡80-87 🔴>87 |
| 利用率 | XX% | - |
| 显存 | XXXX/6144 MiB | 🟢<4G 🟡4-5G 🔴>5G |
| 功耗 | XX/115W | - |

### 2. CUDA 进程列表
```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```
列出所有使用 GPU 的进程，标记训练进程 vs 后台进程。

### 3. Python 进程排查
```bash
wmic process where "name like '%python%'" get ProcessId,CommandLine,WorkingSetSize
```
识别：
- 🟢 训练进程（train_raw_only.py / resume_train.py）
- 🟡 数据预处理进程
- 🔴 僵尸进程（占用内存但无 CPU）

### 4. 热降频风险评估
- 温度 > 85°C + GPU 利用率 100% → 🔴 极高 TDR 风险
- 温度 > 85°C + 功耗被限制 → 🟡 已在热降频
- 温度 < 80°C → 🟢 安全

### 5. 后台抢占检测
列出非训练但占用 GPU 显存的进程：
- Wallpaper Engine (wallpaper32.exe)
- NVIDIA Overlay
- 浏览器 (msedge/EdgeWebView)
- ChatGPT Desktop
- 其他应用

## 输出
```
🖥️ GPU 健康报告
━━━━━━━━━━━━━━━━
🌡️ 温度: 84°C 🔴 (TDR 风险: 高)
💾 显存: 4021/6144 MiB 🟡 (可用 2123 MiB)
⚡ 功耗: 74/115W
📊 利用率: 98%

🔧 CUDA 进程:
  PID 42920  python.exe (训练)  ~3800 MiB
  PID 1644   WindowsTerminal    ~50 MiB

⚠️ 建议: 温度偏高，建议等待降温后再启动新训练。
   或关闭 Wallpaper Engine / NVIDIA Overlay 释放 ~200 MiB。
```
