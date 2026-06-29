# MANIFEST_PATH_GUARD_AUDIT_v080a15

## 问题来源

用户在 Windows 实机启动批量仿真时遇到：

```text
PermissionError: [Errno 13] Permission denied: '.'
```

调用链为：

```text
EasyMainWindow.run_pending_batch
  -> build_pending_tasks
  -> tasks_from_manifest
  -> manifest_csv.open(...)
```

根因是批量页“清单”输入框为空时，`Path("")` 会解析为当前目录 `.`。代码只检查 `exists()`，目录同样存在，于是目录路径被传入 CSV 打开逻辑，在 Windows 上表现为 `PermissionError`。

## 修改内容

1. `services/easy_batch_service.py`
   - 新增 `require_manifest_file()`，集中校验 manifest 路径非空、存在且必须是文件。
   - `read_manifest_rows()` 对缺失或目录路径返回空列表，避免普通统计/预览路径触发异常。
   - `build_batch_plan()` 和 `build_pending_tasks()` 复用统一校验入口。

2. `gui/easy_window.py`
   - `load_model_manifest()` 增加 `is_file()` 校验。
   - `refresh_batch_plan()` 对空输入、缺失文件、目录路径给出明确提示。
   - `run_pending_batch()` 对 manifest 路径做前置校验，并同步 `current_manifest`。

3. `core/runner.py`
   - `tasks_from_manifest()` 和 `manifest_input_files()` 增加 `exists()` / `is_file()` 校验。

## 验证

```bash
python -m compileall -q src scripts
PYTHONPATH=src python - <<'PY'
from uavgpr_simlab.services.easy_batch_service import require_manifest_file, build_pending_tasks, read_manifest_rows
from uavgpr_simlab.core.runner import tasks_from_manifest
for fn in (lambda: require_manifest_file('.'), lambda: build_pending_tasks('.', variants=['raw'], max_tasks=1), lambda: tasks_from_manifest('.', variants=['raw'], limit=1)):
    try:
        fn()
    except Exception as exc:
        print(type(exc).__name__, exc)
print(read_manifest_rows('.'))
PY
```

验证结果：目录路径会变成明确的 `IsADirectoryError`，GUI 入口会提前转换为“无清单”提示，不再进入 gprMax 运行链路。

## 影响范围

- 不改变 manifest schema。
- 不改变模型生成、SceneWorld、gprMax 调用、B-scan 后处理或 QC 语义。
- 只增强路径输入防护和错误提示。
