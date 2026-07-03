#!/usr/bin/env python3
"""
PGDA_SYNTH_DATASET_V1 — Promote to Accepted

Usage:
    python tools/promote_to_accepted.py <case_run_dir> [--force]
    python tools/promote_to_accepted.py 03_runs/batch_001/LINE9_STYLE_FLAT_001

Promotes a GREEN_ACCEPTED case from 03_runs/ to 05_accepted_dataset/.
Copies only training-essential files: input/, label/, metadata/, preview/.
Updates manifest_master.csv.
"""

import sys, json, shutil, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _detect_family(case_id):
    """Map case_id to 05_accepted_dataset/ subdirectory."""
    cid_lower = case_id.lower()
    if cid_lower.startswith('line9_style'):
        if 'terrain' in cid_lower or 'gentle' in cid_lower:
            return 'line9_style/terrain'
        elif 'flat' in cid_lower or 'v1' in cid_lower:
            return 'line9_style/flat'
        else:
            return 'line9_style/mixed'
    elif 'weak_cover' in cid_lower:
        return 'weak_cover'
    elif 'shallow' in cid_lower or 'perturb' in cid_lower:
        return 'shallow_perturbation'
    else:
        return 'generic_smooth'


def promote(case_run_dir, force=False):
    run_dir = Path(case_run_dir)
    case_id = run_dir.name
    batch_id = run_dir.parent.name

    # Check QC decision
    qc_dir = ROOT / '04_qc' / batch_id / case_id
    qc_decision = qc_dir / 'qc_decision.txt'
    qc_report = qc_dir / 'qc_report.json'

    grade = None
    if qc_report.exists():
        report = json.loads(qc_report.read_text())
        grade = report.get('qc_grade', None)
    elif qc_decision.exists():
        grade = qc_decision.read_text().strip().split('\n')[0].replace('QC GRADE: ', '')
    else:
        print(f"ERROR: No QC report found at {qc_dir}")
        print("Run tools/after_run_qc.py first")
        sys.exit(1)

    if grade != 'GREEN' and not force:
        print(f"ERROR: Case {case_id} has QC grade {grade}, not GREEN.")
        print("Use --force to override")
        sys.exit(1)

    # Determine family
    accepted_subdir = _detect_family(case_id)

    dest = ROOT / '05_accepted_dataset' / accepted_subdir / case_id
    if dest.exists():
        if not force:
            print(f"ERROR: {dest} already exists. Use --force to overwrite")
            sys.exit(1)
        else:
            shutil.rmtree(dest)

    # Copy training-essential files
    dest.mkdir(parents=True)
    (dest / 'input').mkdir()
    (dest / 'label').mkdir()
    (dest / 'metadata').mkdir()
    (dest / 'preview').mkdir()

    # input/: bscan.npy (hard requirement)
    src_bscan = run_dir / 'raw' / 'bscan.npy'
    if not src_bscan.exists():
        print(f"ERROR: bscan.npy not found at {src_bscan}")
        sys.exit(1)
    shutil.copy2(src_bscan, dest / 'input' / 'raw_bscan.npy')
    print(f"  input/raw_bscan.npy ✓")

    # label/: from template labels or run labels
    # Search in template labels first (from run_info.json), then inline labels
    run_info = run_dir / 'run_info.json'
    label_sources = []

    if run_info.exists():
        info = json.loads(run_info.read_text())
        template = info.get('template', '')
        if template:
            cand = ROOT / '01_templates' / template / 'labels'
            if cand.exists():
                label_sources.append(cand)

    # Also check <case_dir>/labels/ (some runs may have local labels)
    local_labels = run_dir / 'labels'
    if local_labels.exists():
        label_sources.append(local_labels)

    required_labels = [
        'interface_mask_bscan.npy',
        'interface_mask_wide_bscan.npy',
        'y_soft_501x128.npy',
        'target_visible_phase_time_ns.npy',
        'target_geom_time_ns.npy',
    ]

    copied_labels = 0
    for ls in label_sources:
        for rl in required_labels:
            src = ls / rl
            if src.exists():
                dst = dest / 'label' / rl
                if not dst.exists():
                    shutil.copy2(src, dst)
                    copied_labels += 1

    if copied_labels == 0:
        print(f"ERROR: No label files found/copied. Searched: {label_sources}")
        print("All cases need: interface_mask_bscan.npy, interface_mask_wide_bscan.npy, y_soft_501x128.npy, target_visible_phase_time_ns.npy, target_geom_time_ns.npy")
        sys.exit(1)
    else:
        print(f"  labels/: {copied_labels} files ✓")

    # metadata/: qc_report, design_metrics, scene_world
    for fname in ['qc_report.json']:
        src = qc_dir / fname
        if src.exists():
            shutil.copy2(src, dest / 'metadata' / fname)

    # Use run_info to find the correct template, then case_dir's own geometry
    metadata_sources = []

    # 1. Run_info template (most authoritative)
    if run_info.exists():
        info = json.loads(run_info.read_text())
        template = info.get('template', '')
        if template:
            cand = ROOT / '01_templates' / template
            if cand.exists():
                metadata_sources.insert(0, cand)

    # 2. Case dir's own tables/geometry
    case_dir_from_run = run_dir.parent.parent / '02_case_pool' / batch_id / 'cases' / case_id
    if case_dir_from_run.exists():
        metadata_sources.append(case_dir_from_run)

    # 3. Fallback: first template (least preferred)
    if not metadata_sources:
        for tpl_dir in sorted((ROOT / '01_templates').iterdir()):
            if tpl_dir.is_dir() and not tpl_dir.name.startswith('.'):
                metadata_sources.append(tpl_dir)
                break

    for ms in metadata_sources:
        dm = ms / 'tables' / 'design_metrics.csv'
        if dm.exists() and not (dest / 'metadata' / 'design_metrics.csv').exists():
            shutil.copy2(dm, dest / 'metadata' / 'design_metrics.csv')
        sw = ms / 'geometry' / 'scene_world.json'
        if sw.exists() and not (dest / 'metadata' / 'scene_world.json').exists():
            shutil.copy2(sw, dest / 'metadata' / 'scene_world.json')

    # preview/
    for ext in ['qc_target_zoom.png', 'geometry_preview.png']:
        src = qc_dir / ext
        if not src.exists():
            src = qc_dir / 'qc_preview_full.png'  # fallback
        if src.exists():
            shutil.copy2(src, dest / 'preview' / ext)

    # ── Update manifest ──
    manifest_path = ROOT / 'manifest_master.csv'
    if manifest_path.exists():
        lines = manifest_path.read_text().strip().split('\n')
        header = lines[0]
        # Find or create row
        found = False
        for i, line in enumerate(lines[1:], 1):
            if line.startswith(case_id + ','):
                # Update status, qc_grade, accepted_for_training
                parts = line.split(',')
                # Find accepted_for_training column index
                hdr_parts = header.split(',')
                if 'accepted_for_training' in hdr_parts:
                    idx = hdr_parts.index('accepted_for_training')
                    while len(parts) <= idx:
                        parts.append('')
                    parts[idx] = 'TRUE'
                if 'qc_grade' in hdr_parts:
                    idx = hdr_parts.index('qc_grade')
                    while len(parts) <= idx:
                        parts.append('')
                    parts[idx] = grade
                lines[i] = ','.join(parts)
                found = True
                break
        if not found:
            lines.append(f'{case_id},{batch_id},,,,,,,,,,,,,,,,,,,,,,,,,,,,,TRUE,,,,,,,')
        manifest_path.write_text('\n'.join(lines) + '\n')

    print(f"\n✅ {case_id} promoted to {dest}")
    print(f"   Grade: {grade}")
    return True


def main():
    ap = argparse.ArgumentParser(description="Promote GREEN case to accepted_dataset")
    ap.add_argument('case_run_dir', help='Path to run directory (e.g. 03_runs/batch_001/CASE_ID/)')
    ap.add_argument('--force', action='store_true', help='Force promote even if not GREEN')
    args = ap.parse_args()
    promote(args.case_run_dir, args.force)


if __name__ == '__main__':
    main()
