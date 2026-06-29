from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uavgpr_simlab.core.workspace_relocator import relocate_workspace_paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Inspect/fix absolute paths after moving a UavGPR-SimLab workspace")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--old-root", default="")
    ap.add_argument("--old-roots", default="")
    ap.add_argument("--new-root", default="")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--keep-absolute", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args(argv)
    roots = [x.strip() for x in args.old_roots.replace(";", ",").split(",") if x.strip()]
    rep = relocate_workspace_paths(
        args.manifest,
        old_root=args.old_root or None,
        old_roots=roots,
        new_root=args.new_root or None,
        to_relative=not args.keep_absolute,
        dry_run=not args.apply,
        write_report=True,
        make_backup=not args.no_backup,
        validate_after=not args.no_validate,
    )
    print(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2))
    return 0 if rep.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
