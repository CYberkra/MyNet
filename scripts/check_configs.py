from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
BAD = 0
WARN = 0

LINE6_ALLOWED_DATA_ROOTS = {
    "data_corrected_v1_4_terrain_direction",
    "data_audited_v16_20260627",
    "data_audited_v17_line9_consistent",
}


def bad(msg):
    global BAD
    print(msg)
    BAD += 1


def warn(msg):
    global WARN
    print(msg)
    WARN += 1


for p in sorted((ROOT / "configs").glob("*.json")):
    try:
        cfg = json.load(open(p, encoding="utf-8"))
    except Exception as exc:
        bad(f"CONFIG_BAD {p.name}: invalid json: {exc}")
        continue

    if not isinstance(cfg, dict):
        warn(f"CONFIG_SKIP {p.name}: non-object helper JSON")
        continue

    if "train_lines" not in cfg and "run_dir" not in cfg and "data_root" not in cfg:
        warn(f"CONFIG_SKIP {p.name}: helper JSON")
        continue

    tr = set(cfg.get("train_lines", []))
    va = set(cfg.get("val_lines", []))
    te = set(cfg.get("test_lines", []))
    data_root = cfg.get("data_root")

    if "LineX1" in (tr | va):
        bad(f"CONFIG_BAD {p.name}: LineX1 is review-only and cannot be train/val")

    if "Line6" in (tr | va):
        if data_root not in LINE6_ALLOWED_DATA_ROOTS and cfg.get("allow_line6_training_after_correction") is not True:
            bad(f"CONFIG_BAD {p.name}: Line6 requires an audited/corrected data_root")

    if "Line9" in tr:
        train_range = cfg.get("train_trace_ranges", {}).get("Line9")
        test_range = cfg.get("test_trace_ranges", {}).get("Line9")
        is_locked_holdout = "line9holdout" in p.name.lower() or "line9holdout" in str(cfg.get("version", "")).lower()
        if train_range is None or test_range is None:
            warn(f"CONFIG_WARN {p.name}: Line9 training has no explicit locked trace ranges")
        elif is_locked_holdout and (train_range != [0, 1407] or test_range != [1664, 2377]):
            bad(f"CONFIG_BAD {p.name}: locked Line9 holdout config must preserve the guard band")

    if tr & va:
        experiment_kind = " ".join(str(cfg.get(k, "")) for k in ("version", "note", "acceptance_note")).lower()
        if "fewshot" in experiment_kind or "few-shot" in experiment_kind or "smoke" in experiment_kind:
            warn(f"CONFIG_WARN {p.name}: intentional train/val overlap {sorted(tr & va)}")
        else:
            warn(f"CONFIG_WARN {p.name}: train/val overlap {sorted(tr & va)}")

    if data_root:
        data_path = ROOT / data_root
        if not data_path.exists():
            warn(f"CONFIG_WARN {p.name}: data_root not present in transfer tree: {data_root}")

    if cfg.get("run_dir", "").strip() == "" and cfg.get("epochs") is not None and p.name not in {
        "fast_cpu_check.json",
        "v1_9_candidate_base.json",
    }:
        bad(f"CONFIG_BAD {p.name}: missing run_dir")

print("CONFIG_GUARDRAILS_OK" if BAD == 0 else f"CONFIG_BAD_TOTAL {BAD}")
if WARN:
    print(f"CONFIG_WARN_TOTAL {WARN}")
sys.exit(1 if BAD else 0)
