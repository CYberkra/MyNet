from __future__ import annotations

import argparse
import json
from uavgpr_simlab.services.sceneworld_bscan_service import check_sceneworld_case_package


def main() -> int:
    ap = argparse.ArgumentParser(description="Check SceneWorld case folder completeness.")
    ap.add_argument('--workspace', required=True)
    ap.add_argument('--manifest', default=None)
    ns = ap.parse_args()
    rep = check_sceneworld_case_package(ns.workspace, manifest_csv=ns.manifest)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0 if rep.get('ok') else 2


if __name__ == '__main__':
    raise SystemExit(main())
