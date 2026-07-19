#!/usr/bin/env python3
"""Make non-promotable short-window visual copies of shape pilot cases."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from generate_independent_v2_family01 import write_checksums


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data" / "simulations" / "v2" / "00_controls" / "SHAPE01_BASAL_GEOMETRY_PILOT"
DEFAULT_OUTPUT = ROOT / "data" / "simulations" / "v2" / "00_controls" / "SHAPE01_VISUAL_SHORT"


def make_copy(family: str, output_root: Path) -> Path:
    source = SOURCE_ROOT / family / f"{family}_POS"
    target = output_root / family / f"{family}_POS"
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    for input_path in target.glob("*.in"):
        text = input_path.read_text(encoding="ascii")
        text = text.replace("#time_window: 7.5e-07", "#time_window: 5e-07")
        input_path.write_text(text, encoding="ascii")
    manifest_path = target / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["lifecycle_state"] = "short_visual_diagnostic_only"
    manifest["formal_training_allowed"] = False
    manifest["promotion_allowed"] = False
    manifest["training_block_reason"] = "short 500 ns sparse visual diagnostic; not canonical and not release eligible"
    manifest["grid"]["solver_time_window_ns"] = 500.0
    manifest["grid"]["protected_time_window_ns"] = 450.0
    manifest["grid"]["canonical_output_samples"] = 501
    manifest["next_gate"] = "visual morphology only; regenerate canonical full pair before any promotion"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    write_checksums(target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--family", action="append", default=["BS02_BROAD_RISE", "BS04_GENTLE_MULTISCALE"])
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    paths = [str(make_copy(family, output_root)) for family in args.family]
    print(json.dumps({"formal_training_allowed": False, "cases": paths}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
