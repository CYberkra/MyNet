from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch UavGPR-SimLab GUI")
    parser.add_argument("config", nargs="?", default=None, help="Optional YAML/JSON config path")
    parser.add_argument("--advanced", action="store_true", help="Launch the advanced engineering tabbed interface instead of the productized v0.7 easy interface")
    args = parser.parse_args()
    if args.advanced:
        from uavgpr_simlab.gui.main_window import run_app
        return run_app(Path(args.config) if args.config else None)
    from uavgpr_simlab.gui.easy_window import run_easy_app
    return run_easy_app(Path(args.config) if args.config else None)


if __name__ == "__main__":
    raise SystemExit(main())
