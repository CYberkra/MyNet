from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

COMPONENT_KEYS = {
    "Y_air": ("Y_air", "y_air", "air_only", "air_only_bscan"),
    "Y_target_without_G": ("Y_target_without_G", "y_target_without_g"),
    "X_clean": ("X_clean", "x_clean", "clean", "clean_bscan", "x_target_clean"),
    "G_target": ("G_target", "g_target"),
    "Y_full_component": ("Y_full_component", "y_full_component"),
}


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def first_present(files: set[str], aliases: tuple[str, ...]) -> str:
    for a in aliases:
        if a in files:
            return a
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit optional A/S/G component arrays in training NPZ windows.")
    ap.add_argument("--data-root", default="data_corrected_v1_4_terrain_direction")
    ap.add_argument("--out", default="reports/component_array_coverage.csv")
    ap.add_argument("--max-files", type=int, default=0)
    args = ap.parse_args()

    data_root = resolve(args.data_root)
    windows = data_root / "windows"
    if not windows.exists():
        raise FileNotFoundError(f"windows directory not found: {windows}")
    rows = []
    npz_files = sorted(windows.glob("*.npz"))
    if args.max_files > 0:
        npz_files = npz_files[: args.max_files]
    totals = {k: 0 for k in COMPONENT_KEYS}
    finite_ok = {k: 0 for k in COMPONENT_KEYS}
    nonzero = {k: 0 for k in COMPONENT_KEYS}

    for fp in npz_files:
        try:
            z = np.load(fp, allow_pickle=False)
            files = set(z.files)
            row = {"sample_id": fp.stem}
            for key, aliases in COMPONENT_KEYS.items():
                hit = first_present(files, aliases)
                row[f"{key}_alias"] = hit
                has = bool(hit)
                row[f"{key}_present"] = int(has)
                if has:
                    totals[key] += 1
                    arr = np.asarray(z[hit])
                    ok = bool(np.isfinite(arr).all())
                    nz = bool(np.nanmax(np.abs(arr)) > 0) if arr.size else False
                    finite_ok[key] += int(ok)
                    nonzero[key] += int(nz)
                    row[f"{key}_shape"] = "x".join(map(str, arr.shape))
                    row[f"{key}_finite"] = int(ok)
                    row[f"{key}_nonzero"] = int(nz)
                else:
                    row[f"{key}_shape"] = ""
                    row[f"{key}_finite"] = 0
                    row[f"{key}_nonzero"] = 0
            rows.append(row)
        except Exception as exc:
            rows.append({"sample_id": fp.stem, "error": repr(exc)})

    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id"]
    for key in COMPONENT_KEYS:
        fieldnames += [f"{key}_present", f"{key}_alias", f"{key}_shape", f"{key}_finite", f"{key}_nonzero"]
    fieldnames += ["error"]
    with out.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)

    n = len(npz_files)
    print(f"Audited {n} npz windows under {windows}")
    for key in COMPONENT_KEYS:
        print(
            f"{key}: present={totals[key]}/{n} "
            f"finite={finite_ok[key]}/{n} nonzero={nonzero[key]}/{n}"
        )
    print(out)


if __name__ == "__main__":
    main()
