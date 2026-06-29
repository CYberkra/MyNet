from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uavgpr_simlab.core.dataset_contract import validate_dataset_skeleton


def _split(text: str) -> list[str]:
    return [x.strip() for x in str(text).replace(";", ",").split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a UavGPR-SimLab dataset skeleton/manifest before importing or batch running it.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--variants", default="raw,target_only,background_only,clutter_only,air_only")
    ap.add_argument("--allow-absolute-paths", action="store_true")
    ap.add_argument("--write-report", action="store_true")
    args = ap.parse_args()
    rep = validate_dataset_skeleton(
        args.manifest,
        expected_variants=_split(args.variants),
        require_relative_paths=not args.allow_absolute_paths,
        write_report=args.write_report,
    )
    print(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2))
    return 0 if rep.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
