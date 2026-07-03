from pathlib import Path
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "reports" / "full_label_reaudit_terrain_direction"
BY_TRACE_DIR = AUDIT_DIR / "by_trace"


KNOWN_DEPTH_CONSTRAINTS = [
    {
        "constraint_id": "C01",
        "pair": ("Line3", "Line7"),
        "applies_to": "both",
        "expected_low_m": 16.0,
        "expected_high_m": 18.0,
        "source_note": "UavGPR Line7 page: near 3号测线 crossing basal/interface is about 17 m",
    },
    {
        "constraint_id": "C02",
        "pair": ("Line6", "Line9"),
        "applies_to": "both",
        "expected_low_m": 13.0,
        "expected_high_m": 16.0,
        "source_note": "UavGPR Line6/Line9 pages: Line9/ZK08 zone basal interface stays in the about 12-16 m band",
    },
    {
        "constraint_id": "C03",
        "pair": ("Line6", "LineL1"),
        "applies_to": "both",
        "expected_low_m": 17.0,
        "expected_high_m": 19.5,
        "source_note": "UavGPR Line6 and LineL1 pages: 6号-L1 crossing basal/interface is about 18 m",
    },
    {
        "constraint_id": "C04",
        "pair": ("Line3", "LineL1"),
        "applies_to": "LineL1",
        "expected_low_m": 12.0,
        "expected_high_m": 14.5,
        "source_note": "UavGPR LineL1 page: 3号 crossing basal/interface is about 13 m",
    },
    {
        "constraint_id": "C05",
        "pair": ("LineL1", "LineX1"),
        "applies_to": "both",
        "expected_low_m": 12.0,
        "expected_high_m": 15.0,
        "source_note": "UavGPR LineX1/LineL1 pages: X1-L1 crossing stays in the about 12-15 m band",
    },
]


PDF_DIRECTION_NOTES = {
    "Line3": "UavGPR-29 profile inset arrow points left; aerial inset shows ZK08 left and ZK07 right",
    "Line6": "UavGPR-30 profile inset arrow points left; aerial inset arrow points toward west/left",
    "Line7": "UavGPR-31 profile inset arrow points right; aerial inset arrow points toward east/right",
    "Line9": "UavGPR-32 profile/aerial arrows point left; ZK08 is near the east/right side",
    "LineL1": "UavGPR-33 aerial arrow points right, but CSV trace order is east-to-west; treat direction as medium confidence",
    "LineX1": "UavGPR-34 aerial arrow points right; no separate engineering-profile PDF",
}


def load_line(line: str) -> pd.DataFrame:
    path = BY_TRACE_DIR / f"{line}_terrain_label_by_trace.csv"
    df = pd.read_csv(path)
    for col in [
        "trace_idx",
        "distance_m",
        "lon",
        "lat",
        "ground_elevation_est_m",
        "relative_height_m",
        "label_depth_m",
        "label_interface_elevation_est_m",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def project_xy(lines):
    all_lon = np.concatenate([df["lon"].to_numpy(float) for df in lines.values()])
    all_lat = np.concatenate([df["lat"].to_numpy(float) for df in lines.values()])
    lon0 = float(np.nanmean(all_lon))
    lat0 = float(np.nanmean(all_lat))
    cos_lat = math.cos(math.radians(lat0))
    out = {}
    for line, df in lines.items():
        x = (df["lon"].to_numpy(float) - lon0) * 111_320.0 * cos_lat
        y = (df["lat"].to_numpy(float) - lat0) * 110_540.0
        out[line] = (x, y)
    return out


def nearest_pair(x1, y1, x2, y2, block=512):
    best = (float("inf"), -1, -1)
    xy2 = np.column_stack([x2, y2])
    for start in range(0, len(x1), block):
        stop = min(start + block, len(x1))
        dx = x1[start:stop, None] - xy2[None, :, 0]
        dy = y1[start:stop, None] - xy2[None, :, 1]
        d2 = dx * dx + dy * dy
        local = int(np.nanargmin(d2))
        i_local, j = np.unravel_index(local, d2.shape)
        d = float(math.sqrt(d2[i_local, j]))
        if d < best[0]:
            best = (d, start + i_local, int(j))
    return best


def bearing_deg(lon1, lat1, lon2, lat2):
    lon1r, lat1r, lon2r, lat2r = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2r - lon1r
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def compass_from_bearing(bearing):
    dirs = [
        "N",
        "NE",
        "E",
        "SE",
        "S",
        "SW",
        "W",
        "NW",
    ]
    return dirs[int((bearing + 22.5) // 45) % 8]


def value_at(df, idx, col):
    val = df.iloc[int(idx)][col]
    return "" if pd.isna(val) else float(val)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_direction_registry(lines):
    rows = []
    for line, df in lines.items():
        first = df.iloc[0]
        last = df.iloc[-1]
        b = bearing_deg(first["lon"], first["lat"], last["lon"], last["lat"])
        rows.append(
            {
                "line": line,
                "trace0_lon": float(first["lon"]),
                "trace0_lat": float(first["lat"]),
                "last_lon": float(last["lon"]),
                "last_lat": float(last["lat"]),
                "trace_order_bearing_deg": round(b, 2),
                "trace_order_compass": compass_from_bearing(b),
                "trace_count": int(len(df)),
                "line_length_m": round(float(df["distance_m"].iloc[-1]), 3),
                "pdf_direction_note": PDF_DIRECTION_NOTES.get(line, ""),
            }
        )
    return rows


def build_pair_candidates(lines, xy):
    names = sorted(lines)
    rows = []
    for a_i, line_a in enumerate(names):
        for line_b in names[a_i + 1 :]:
            x1, y1 = xy[line_a]
            x2, y2 = xy[line_b]
            d, ia, ib = nearest_pair(x1, y1, x2, y2)
            dfa = lines[line_a]
            dfb = lines[line_b]
            rows.append(
                {
                    "line_a": line_a,
                    "line_b": line_b,
                    "nearest_distance_m": round(d, 3),
                    "trace_a": int(ia),
                    "distance_a_m": round(float(dfa.iloc[ia]["distance_m"]), 3),
                    "depth_a_m": value_at(dfa, ia, "label_depth_m"),
                    "ground_a_m": value_at(dfa, ia, "ground_elevation_est_m"),
                    "interface_elev_a_m": value_at(dfa, ia, "label_interface_elevation_est_m"),
                    "trace_b": int(ib),
                    "distance_b_m": round(float(dfb.iloc[ib]["distance_m"]), 3),
                    "depth_b_m": value_at(dfb, ib, "label_depth_m"),
                    "ground_b_m": value_at(dfb, ib, "ground_elevation_est_m"),
                    "interface_elev_b_m": value_at(dfb, ib, "label_interface_elevation_est_m"),
                    "lon_a": round(float(dfa.iloc[ia]["lon"]), 9),
                    "lat_a": round(float(dfa.iloc[ia]["lat"]), 9),
                    "lon_b": round(float(dfb.iloc[ib]["lon"]), 9),
                    "lat_b": round(float(dfb.iloc[ib]["lat"]), 9),
                }
            )
    return rows


def check_constraints(lines, pair_rows):
    pair_lookup = {}
    for row in pair_rows:
        pair_lookup[(row["line_a"], row["line_b"])] = row
        pair_lookup[(row["line_b"], row["line_a"])] = {
            **row,
            "line_a": row["line_b"],
            "line_b": row["line_a"],
            "trace_a": row["trace_b"],
            "trace_b": row["trace_a"],
            "distance_a_m": row["distance_b_m"],
            "distance_b_m": row["distance_a_m"],
            "depth_a_m": row["depth_b_m"],
            "depth_b_m": row["depth_a_m"],
            "ground_a_m": row["ground_b_m"],
            "ground_b_m": row["ground_a_m"],
            "interface_elev_a_m": row["interface_elev_b_m"],
            "interface_elev_b_m": row["interface_elev_a_m"],
        }
    rows = []
    for item in KNOWN_DEPTH_CONSTRAINTS:
        line_a, line_b = item["pair"]
        pair = pair_lookup[(line_a, line_b)]
        applies = [line_a, line_b] if item["applies_to"] == "both" else [item["applies_to"]]
        for line in applies:
            is_a = line == pair["line_a"]
            depth = pair["depth_a_m"] if is_a else pair["depth_b_m"]
            trace = pair["trace_a"] if is_a else pair["trace_b"]
            dist = pair["distance_a_m"] if is_a else pair["distance_b_m"]
            delta = ""
            verdict = "missing_depth"
            if depth != "":
                low = item["expected_low_m"]
                high = item["expected_high_m"]
                if low <= depth <= high:
                    verdict = "ok_within_pdf_anchor_band"
                    delta = 0.0
                elif depth < low:
                    verdict = "too_shallow_vs_pdf_anchor"
                    delta = round(float(depth) - low, 3)
                else:
                    verdict = "too_deep_vs_pdf_anchor"
                    delta = round(float(depth) - high, 3)
            rows.append(
                {
                    "constraint_id": item["constraint_id"],
                    "line_pair": f"{line_a}-{line_b}",
                    "checked_line": line,
                    "nearest_pair_distance_m": pair["nearest_distance_m"],
                    "trace_idx": int(trace),
                    "distance_m": dist,
                    "current_label_depth_m": depth,
                    "expected_low_m": item["expected_low_m"],
                    "expected_high_m": item["expected_high_m"],
                    "verdict": verdict,
                    "delta_to_band_m": delta,
                    "source_note": item["source_note"],
                }
            )
    return rows


def build_crossing_consistency(pair_rows, max_crossing_distance_m=2.0):
    rows = []
    for row in pair_rows:
        if float(row["nearest_distance_m"]) > max_crossing_distance_m:
            continue
        depth_a = row["depth_a_m"]
        depth_b = row["depth_b_m"]
        if depth_a == "" or depth_b == "":
            diff = ""
            verdict = "missing_depth"
        else:
            diff = abs(float(depth_a) - float(depth_b))
            if diff <= 1.0:
                verdict = "ok_crossing_consistent"
            elif diff <= 2.0:
                verdict = "review_crossing_depth_offset"
            else:
                verdict = "inconsistent_crossing_depth"
        rows.append(
            {
                "line_pair": f"{row['line_a']}-{row['line_b']}",
                "nearest_distance_m": row["nearest_distance_m"],
                "line_a": row["line_a"],
                "trace_a": row["trace_a"],
                "distance_a_m": row["distance_a_m"],
                "depth_a_m": depth_a,
                "line_b": row["line_b"],
                "trace_b": row["trace_b"],
                "distance_b_m": row["distance_b_m"],
                "depth_b_m": depth_b,
                "abs_depth_difference_m": "" if diff == "" else round(float(diff), 3),
                "verdict": verdict,
                "interpretation": "Same GPS crossing should normally agree unless the PDF/profile marks different target interfaces.",
            }
        )
    return rows


def plot_crossings(lines, xy, pair_rows, out_png):
    fig, ax = plt.subplots(figsize=(13, 10), constrained_layout=True)
    colors = {
        "Line3": "#2563eb",
        "Line6": "#f97316",
        "Line7": "#16a34a",
        "Line9": "#dc2626",
        "LineL1": "#7c3aed",
        "LineX1": "#6b4f4f",
    }
    for line, df in lines.items():
        x, y = xy[line]
        ax.plot(x, y, color=colors.get(line), lw=1.5, label=line)
        ax.scatter([x[0]], [y[0]], s=45, color=colors.get(line), marker="o")
        ax.scatter([x[-1]], [y[-1]], s=45, color=colors.get(line), marker="x")
        ax.text(x[0], y[0], f" {line} tr0", fontsize=8)
        ax.text(x[-1], y[-1], f" {line} end", fontsize=8)
    important = {tuple(item["pair"]) for item in KNOWN_DEPTH_CONSTRAINTS}
    important |= {(b, a) for a, b in important}
    for row in pair_rows:
        if (row["line_a"], row["line_b"]) not in important:
            continue
        dfa = lines[row["line_a"]]
        x, y = xy[row["line_a"]]
        ia = int(row["trace_a"])
        ax.scatter([x[ia]], [y[ia]], s=80, facecolors="none", edgecolors="black", linewidths=1.5)
        ax.text(
            x[ia],
            y[ia],
            f" {row['line_a']}-{row['line_b']} {row['nearest_distance_m']}m",
            fontsize=8,
        )
    ax.set_title("GPS trace order and PDF-constrained crossing candidates")
    ax.set_xlabel("projected x m")
    ax.set_ylabel("projected y m")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main():
    lines = {}
    for path in sorted(BY_TRACE_DIR.glob("*_terrain_label_by_trace.csv")):
        line = path.name.split("_terrain_label_by_trace.csv")[0]
        lines[line] = load_line(line)
    xy = project_xy(lines)
    direction_rows = build_direction_registry(lines)
    pair_rows = build_pair_candidates(lines, xy)
    constraint_rows = check_constraints(lines, pair_rows)
    consistency_rows = build_crossing_consistency(pair_rows)

    write_csv(AUDIT_DIR / "trace_direction_registry.csv", direction_rows)
    write_csv(AUDIT_DIR / "nearest_line_pair_candidates.csv", pair_rows)
    write_csv(AUDIT_DIR / "pdf_anchor_depth_checks.csv", constraint_rows)
    write_csv(AUDIT_DIR / "crossing_depth_consistency.csv", consistency_rows)
    plot_crossings(lines, xy, pair_rows, AUDIT_DIR / "figures" / "crossing_trace_candidates.png")

    flagged = [r for r in constraint_rows if "too_" in r["verdict"]]
    inconsistent = [r for r in consistency_rows if r["verdict"] == "inconsistent_crossing_depth"]
    print(f"wrote {AUDIT_DIR / 'trace_direction_registry.csv'}")
    print(f"wrote {AUDIT_DIR / 'nearest_line_pair_candidates.csv'}")
    print(f"wrote {AUDIT_DIR / 'pdf_anchor_depth_checks.csv'}")
    print(f"wrote {AUDIT_DIR / 'crossing_depth_consistency.csv'}")
    print(f"flagged_constraints={len(flagged)}")
    print(f"inconsistent_crossings={len(inconsistent)}")


if __name__ == "__main__":
    main()
