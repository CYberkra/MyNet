#!/usr/bin/env python3
"""
PGDA_SYNTH_DATASET_V1 — Preflight Checker

Usage:
    python tools/preflight_check.py <case_dir_or_id>
    python tools/preflight_check.py 02_case_pool/batch_001/LINE9_STYLE_FLAT_001/
    python tools/preflight_check.py 01_templates/LINE9_LABEL_INSPIRED_V1/

Exit code: 0 = all PASS, 1 = any FAIL (preflight not cleared), 2 = WARNING only
"""

import sys, json, math, re, argparse
from pathlib import Path

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False

ROOT = Path(__file__).resolve().parents[1]
DT_NS = 700.0  # time window

# ── Material velocity lookup ──
MATERIAL_VELOCITY = {
    'air': 0.300,
    'moist_silty_clay': 0.082,
    'weak_cover_band': 0.084,
    'slide_zone': 0.061,
    'weathered_bedrock': 0.122,
    'background_surrogate': 0.082,
}

RESULTS = []  # accumulate (name, status, detail)

def check(name, status, detail=""):
    RESULTS.append((name, status, detail))
    return status

def find_in_file(case_dir):
    """Locate raw.in in the case directory tree."""
    candidates = sorted(Path(case_dir).rglob("raw.in"))
    if not candidates:
        candidates = sorted(Path(case_dir).rglob("*.in"))
    return candidates[0] if candidates else None

def parse_in_file(path):
    """Parse # commands from a gprMax .in file into a list of (type, args_dict)."""
    cmds = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#:"):
                continue
            if s.startswith("#"):
                parts = s[1:].split(":")
                cmd_type = parts[0].strip()
                args_str = ":".join(parts[1:]) if len(parts) > 1 else ""
                cmds.append((cmd_type, args_str.strip()))
    return cmds

def find_command(cmds, cmd_type):
    """Find first command of given type."""
    for t, a in cmds:
        if t == cmd_type:
            return a
    return None

def find_all_commands(cmds, cmd_type):
    """Find all commands of given type."""
    return [a for t, a in cmds if t == cmd_type]

def run_preflight(case_dir):
    global RESULTS
    RESULTS = []
    case_dir = Path(case_dir)
    if not case_dir.exists():
        print(f"ERROR: {case_dir} does not exist")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"  PREFLIGHT CHECK: {case_dir.name}")
    print(f"  Path: {case_dir.resolve()}")
    print(f"{'='*60}\n")

    # ── Locate input files ──
    in_path = find_in_file(case_dir)
    if not in_path:
        check("in_file_exists", "FAIL", "No .in file found")
        return
    try:
        check("in_file_exists", "PASS", str(in_path.relative_to(ROOT)))
    except ValueError:
        check("in_file_exists", "PASS", str(in_path))

    cmds = parse_in_file(in_path)

    # ── 1. Domain ──
    domain_x_m, domain_y_m = 480.0, 45.0
    dx_cell = 0.05  # fallback defaults
    domain_str = find_command(cmds, "domain")
    if domain_str:
        parts = domain_str.split()
        if len(parts) >= 2:
            domain_x_m, domain_y_m = float(parts[0]), float(parts[1])
            check("domain_declared", "PASS", f"domain: {domain_x_m}×{domain_y_m}")
        else:
            check("domain_declared", "WARN", f"Malformed: {domain_str}")
    else:
        check("domain_declared", "FAIL", "Missing #domain")

    # Parse dx from #dx_dy_dz (not domain's 3rd param which is domain_z in 3D)
    dx_dy_dz_str = find_command(cmds, "dx_dy_dz")
    if dx_dy_dz_str:
        dd_parts = dx_dy_dz_str.split()
        if dd_parts:
            dx_cell = float(dd_parts[0])
    # Check domain_y / dx_cell is integer
    ratio = domain_y_m / dx_cell
    if abs(ratio - round(ratio)) < 1e-6:
        check("domain_grid_integer", "PASS", f"domain_y/dx = {ratio:.0f}")
    else:
        check("domain_grid_integer", "FAIL", f"domain_y({domain_y_m})/dx({dx_cell}) = {ratio:.4f}, not integer")

    # ── 2. PML ──
    pml_str = find_command(cmds, "pml_cells")
    if pml_str:
        pml_parts = pml_str.split()
        if pml_parts == ["60", "60", "0", "60", "60", "0"]:
            check("pml_correct", "PASS", "60 60 0 60 60 0")
        else:
            check("pml_correct", "WARN", f"Got: {pml_str}, expected: 60 60 0 60 60 0")
        check("pml_declared", "PASS", f"#pml_cells: {pml_str}")
    else:
        check("pml_declared", "FAIL", "Missing #pml_cells")

    # ── 3. H5 /data prohibition ──
    h5_cmds = [a for a in find_all_commands(cmds, "geometry_objects_read")]
    if h5_cmds:
        check("no_h5_data", "FAIL", f"Found #geometry_objects_read: {h5_cmds[0][:80]}...")
    else:
        check("no_h5_data", "PASS", "No H5 /data usage")

    # ── 4. noair prohibition ──
    noair_found = any("noair" in a.lower() or "no_air" in a.lower() for t, a in cmds if t in ("box", "triangle"))
    for a in find_all_commands(cmds, "box"):
        if "noair" in a.lower():
            noair_found = True
            break
    if noair_found:
        check("no_noair", "FAIL", "Found noair/no_air reference")
    else:
        check("no_noair", "PASS", "No noair usage")

    # ── 5. Collect triangle commands ──
    tri_cmds = find_all_commands(cmds, "triangle")

    # ── 5b. GprMax v3.1.7 comment syntax check ──
    # Pure `#` lines without colon cause IndexError in check_cmd_names
    known_commands = {'title', 'domain', 'dx_dy_dz', 'time_window', 'pml_cells',
                      'material', 'waveform', 'hertzian_dipole', 'rx', 'src_steps',
                      'rx_steps', 'geometry_view', 'triangle', 'box', 'cylinder',
                      'geometry_objects_read', 'geometry_objects_write'}
    bad_comment_lines = []
    with open(in_path) as f:
        for lineno, line in enumerate(f, 1):
            s = line.strip()
            if s.startswith('#') and ':' not in s:
                kw = s[1:].strip().split()[0] if s[1:].strip() else ''
                if kw and kw not in known_commands:
                    bad_comment_lines.append(f"L{lineno}: {s[:80]}")
    if bad_comment_lines:
        check("no_bare_comments", "FAIL", f"Found {len(bad_comment_lines)} bare # comment lines (use #: instead):\n          " + "; ".join(bad_comment_lines[:5]))
    else:
        check("no_bare_comments", "PASS", "No bare # comment lines")

    # ── 5c. All triangles must have averaging=y ──
    if tri_cmds:
        no_smooth = [a[:60] for a in tri_cmds if not (len(a.split()) >= 12 and a.split()[11].lower() == 'y')]
        if no_smooth:
            check("all_triangles_smoothed", "FAIL", f"{len(no_smooth)}/{len(tri_cmds)} triangles lack averaging=y")
        else:
            check("all_triangles_smoothed", "PASS", f"All {len(tri_cmds)} triangles have averaging=y")

    # ── 6. Source/RX in PML? ──
    src_str = find_command(cmds, "hertzian_dipole")
    rx_str = find_command(cmds, "rx")
    pml_parts = pml_str.split() if pml_str else []
    pml_x_cells = int(pml_parts[0]) if pml_parts else 60
    pml_y0_cells = int(pml_parts[2]) if len(pml_parts) > 2 else 0
    pml_y1_cells = int(pml_parts[3]) if len(pml_parts) > 3 else 60
    pml_x_m = pml_x_cells * dx_cell
    pml_y0_m = pml_y0_cells * dx_cell
    pml_y1_m = pml_y1_cells * dx_cell

    for label, s, fmt in [
        ("source", src_str, "dipole"),
        ("receiver", rx_str, "rx"),
    ]:
        if s:
            parts = s.split()
            if fmt == "dipole":
                # #hertzian_dipole: z x y z_antenna wavelet
                if len(parts) >= 4:
                    x, y = float(parts[1]), float(parts[2])
                else:
                    continue
            else:
                # #rx: x y z [name]
                if len(parts) >= 2:
                    x, y = float(parts[0]), float(parts[1])
                else:
                    continue
            failures = []
            if x < pml_x_m or x > domain_x_m - pml_x_m:
                failures.append(f"x={x:.1f}m (PML edge [{pml_x_m:.1f}, {domain_x_m - pml_x_m:.1f}])")
            if pml_y0_m > 0 and y < pml_y0_m:
                failures.append(f"y={y:.1f}m in top PML (edge at {pml_y0_m:.1f}m)")
            if pml_y1_m > 0 and y > domain_y_m - pml_y1_m:
                failures.append(f"y={y:.1f}m in bottom PML (edge at {domain_y_m - pml_y1_m:.1f}m)")
            if failures:
                check(f"{label}_in_pml", "FAIL", f"{label} {'; '.join(failures)}")
            else:
                check(f"{label}_in_pml", "PASS", f"{label} at ({x:.3f}, {y:.3f}) — outside PML")
        else:
            check(f"{label}_defined", "INFO", f"No {label} command found")

    # ── 6b. TX/RX inside ground material check ──
    # Instead of inferring surface from min(tri_y), directly check if TX/RX
    # falls inside any non-air material triangle at that x-position.
    if src_str and rx_str:
        try:
            dip_parts = src_str.split()
            rx_parts = rx_str.split()
            tx_x = float(dip_parts[1]) if len(dip_parts) >= 3 else None
            tx_y = float(dip_parts[2]) if len(dip_parts) >= 3 else None
            rx_x_f = float(rx_parts[0]) if len(rx_parts) >= 2 else None
            rx_y = float(rx_parts[1]) if len(rx_parts) >= 2 else None

            buried = []
            for t, a in cmds:
                if t == "triangle":
                    pts = a.split()
                    if len(pts) >= 9:
                        # After parse_in_file strips '#triangle:', args are:
                        # x1 y1 z1 x2 y2 z2 x3 y3 z3 dx material [smoothing]
                        xs = [float(pts[0]), float(pts[3]), float(pts[6])]
                        ys = [float(pts[1]), float(pts[4]), float(pts[7])]
                        mat = pts[-2]
                        x_lo, x_hi = min(xs), max(xs)
                        y_lo, y_hi = min(ys), max(ys)
                        for label, ax, ay in [("TX", tx_x, tx_y), ("RX", rx_x_f, rx_y)]:
                            if ax is not None and x_lo - 0.01 <= ax <= x_hi + 0.01 and y_lo - 0.01 <= ay <= y_hi + 0.01:
                                buried.append(f"{label} inside {mat} at y=[{y_lo:.2f},{y_hi:.2f}]")
            if buried:
                check("antenna_in_ground", "FAIL", "; ".join(set(buried)))
            else:
                check("antenna_in_ground", "PASS", "TX/RX outside all ground material triangles")
        except Exception as e:
            check("antenna_in_ground", "INFO", f"Cannot verify: {e}")

    # ── 6c. Triangles within domain check ──
    try:
        tri_max_ys = []
        for t, a in cmds:
            if t == "triangle":
                pts = a.split()
                if len(pts) >= 9:
                    # #triangle: x1 y1 z1 x2 y2 z2 x3 y3 z3 dx material smoothing
                    tri_max_ys.append(max(float(pts[1]), float(pts[4]), float(pts[7])))
        if tri_max_ys and domain_str:
            dom_y = float(domain_str.split()[1])
            max_tri = max(tri_max_ys)
            if max_tri > dom_y + 0.01:
                check("triangles_in_domain", "FAIL", f"max_tri={max_tri:.1f}m > domain_y={dom_y:.0f}m")
            else:
                check("triangles_in_domain", "PASS", f"max_tri={max_tri:.1f}m <= domain_y={dom_y:.0f}m")
    except Exception as e:
        check("triangles_in_domain", "INFO", f"Cannot verify: {e}")

    # ── 7. Side boundary return time ──
    if domain_str and pml_str:
        try:
            # Determine scan range from TX/RX positions
            scan_start, scan_end = 0, domain_x_m
            # Extract TX x position
            if src_str:
                dip_parts = src_str.split()
                tx_x = float(dip_parts[1]) if len(dip_parts) > 1 else None
            else:
                tx_x = None
            # Extract first RX x position
            rx_x = None
            if rx_str:
                rx_parts = rx_str.split()
                rx_x = float(rx_parts[0]) if rx_parts else None
            # Extract rx_steps
            rx_steps_str = find_command(cmds, "rx_steps")
            rx_step = 1.7
            if rx_steps_str:
                rsp = rx_steps_str.split()
                rx_step = float(rsp[0]) if rsp else rx_step
            # Estimate scan range
            if rx_x is not None:
                scan_start = min(rx_x, tx_x or rx_x)
                num_traces = 128
                scan_end = rx_x + rx_step * (num_traces - 1)
            elif tx_x is not None:
                scan_start = tx_x
                num_traces = 128
                scan_end = tx_x + rx_step * (num_traces - 1)

            margin_left = scan_start - pml_x_m
            margin_right = domain_x_m - scan_end - pml_x_m

            # Use air velocity for side boundary check (fastest path = earliest return)
            v_air = MATERIAL_VELOCITY['air']
            return_left_air = 2 * margin_left / v_air if margin_left > 0 else 1e12
            return_right_air = 2 * margin_right / v_air if margin_right > 0 else 1e12
            side_air = min(return_left_air, return_right_air)

            # Also compute slowest-velocity path for context
            v_min = min(MATERIAL_VELOCITY.values())
            return_left_min = 2 * margin_left / v_min if margin_left > 0 else 0
            return_right_min = 2 * margin_right / v_min if margin_right > 0 else 0
            side_ground = min(return_left_min, return_right_min) if (return_left_min > 0 and return_right_min > 0) else max(return_left_min, return_right_min)

            if side_air >= 700:
                check("side_boundary_return", "PASS", f"Air-side return ~{side_air:.0f}ns > 700ns (ground = ~{side_ground:.0f}ns)")
            elif side_air >= 600:
                check("side_boundary_return", "WARN", f"Air-side return ~{side_air:.0f}ns, marginal (ground = ~{side_ground:.0f}ns)")
            else:
                check("side_boundary_return", "FAIL", f"Air-side return ~{side_air:.0f}ns < 700ns (ground = ~{side_ground:.0f}ns)")
        except Exception as e:
            check("side_boundary_return", "WARN", f"Could not compute: {e}")

    # ── 8. Label files ──
    label_dir = case_dir / "labels"
    if label_dir.exists():
        required_labels = [
            "target_geom_time_ns.npy",
            "target_visible_phase_time_ns.npy",
            "y_soft_501x128.npy",
            "interface_mask_bscan.npy",
        ]
        for rl in required_labels:
            if (label_dir / rl).exists():
                check(f"label_{rl}", "PASS", f"{rl} exists")
            else:
                check(f"label_{rl}", "FAIL", f"{rl} missing")

        # 8a. Label non-flat check (prevents flat horizontal line labels)
        if HAS_NP:
            try:
                vis = np.load(str(label_dir / "target_visible_phase_time_ns.npy"))
                vrange = float(np.nanmax(vis) - np.nanmin(vis))
                if vrange > 0.5:
                    check("label_non_flat", "PASS", f"label range={vrange:.1f}ns (>0.5)")
                else:
                    check("label_non_flat", "FAIL", f"label range={vrange:.1f}ns (<0.5) — flat line!")
            except Exception as e:
                check("label_non_flat", "WARN", f"Could not check: {e}")
        else:
            check("label_non_flat", "INFO", "numpy not available, skip")
    else:
        check("label_dir", "INFO", "No labels/ directory (may be template without design labels)")

    # ── 9. Generator record ──
    sw_json = case_dir / "geometry" / "scene_world.json"
    if sw_json.exists():
        try:
            sw = json.loads(sw_json.read_text())
            gen = sw.get("generator", {})
            if gen:
                check("generator_record", "PASS", f"Generator: {gen.get('script', 'N/A')}")
            else:
                check("generator_record", "INFO", "scene_world.json has no generator field")
        except:
            check("generator_record", "WARN", "scene_world.json unreadable")
    else:
        check("generator_record", "INFO", "No scene_world.json")

    # ── Summary ──
    print()
    passed = sum(1 for _, s, _ in RESULTS if s == "PASS")
    failed = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    warned = sum(1 for _, s, _ in RESULTS if s == "WARN")
    info = sum(1 for _, s, _ in RESULTS if s == "INFO")

    print(f"{'='*60}")
    print(f"  RESULTS: {passed} PASS, {failed} FAIL, {warned} WARN, {info} INFO")
    print(f"{'='*60}\n")

    for name, status, detail in RESULTS:
        icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "INFO": "ℹ️"}.get(status, "?")
        print(f"  {icon} [{status:5s}] {name}")
        if detail:
            print(f"          {detail}")

    print()

    if failed > 0:
        sys.exit(1)
    elif warned > 0:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PGDA_SYNTH_DATASET_V1 — Preflight Checker")
    ap.add_argument("case_dir", help="Path to case directory (e.g. 02_case_pool/batch_001/CASE_ID/)")
    args = ap.parse_args()
    run_preflight(args.case_dir)
