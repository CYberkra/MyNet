#!/usr/bin/env python3
"""Create an immutable 100 MHz Ricker source ablation from a native deck.

The copied geometry, labels, materials, acquisition positions, and paired
controls stay byte-identical. Only the waveform declaration changes. The deck
remains blocked from formal training until solver and morphology gates pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_native_256_correlated_voxel_batch import write_checksums  # noqa: E402


DEFAULT_SOURCE = (
    ROOT
    / "data"
    / "simulations"
    / "v2"
    / "03_native_256_80m_family_pilot_v1"
    / "N256_CV03_PATCHY_TRANSITION_POS_80M"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "simulations"
    / "v2"
    / "04_native_256_waveform_ablation_v1"
    / "N256_CV03_PATCHY_TRANSITION_POS_80M_RICKER100_SFCWPROXY"
)

WAVEFORM_RE = re.compile(
    r"(?m)^#waveform:\s*ricker\s+1\s+[^\s]+\s+native_cv_wavelet\s*$"
)
SOLVER_ARTIFACT_PATTERNS = (
    "*.out",
    "*.vti",
    "run_logs",
    "preflight",
    "run_manifest.json",
    "run_state.json",
    "postprocess_validation.json",
    "sfcw_band_equivalent",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rewrite_waveform(path: Path, frequency_hz: float) -> None:
    text = path.read_text(encoding="utf-8")
    replacement = f"#waveform: ricker 1 {frequency_hz:.12g} native_cv_wavelet"
    updated, count = WAVEFORM_RE.subn(replacement, text)
    if count != 1:
        raise RuntimeError(f"expected exactly one native Ricker waveform in {path}, found {count}")
    path.write_text(updated, encoding="ascii")


def create_waveform_ablation(
    source: Path,
    output: Path,
    *,
    center_frequency_hz: float = 100e6,
    overwrite: bool = False,
) -> dict[str, object]:
    """Copy one native source deck and change only its Ricker centre frequency."""
    source = source.resolve()
    output = output.resolve()
    if center_frequency_hz <= 0:
        raise ValueError("center_frequency_hz must be positive")
    if not source.is_dir():
        raise FileNotFoundError(f"source deck does not exist: {source}")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output exists; pass overwrite=True: {output}")
        shutil.rmtree(output)

    manifest_path = source / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not bool(manifest.get("target_presence")):
        raise ValueError("the first waveform ablation requires a positive full/control source deck")
    if str(manifest.get("source", {}).get("waveform")) != "ricker":
        raise ValueError("source deck is not a Ricker baseline")

    shutil.copytree(source, output, ignore=shutil.ignore_patterns(*SOLVER_ARTIFACT_PATTERNS))
    input_paths = sorted(output.glob("*.in"))
    if not input_paths:
        raise RuntimeError("source deck has no gprMax input files")
    for input_path in input_paths:
        _rewrite_waveform(input_path, center_frequency_hz)

    case_id = f"{manifest['case_id']}_RICKER{int(round(center_frequency_hz / 1e6))}_SFCWPROXY"
    source_info = dict(manifest["source"])
    source_info.update(
        {
            "waveform": "ricker",
            "center_frequency_hz": center_frequency_hz,
            "role": "broadband FDTD source for SFCW-band-equivalent postprocessing",
            "hardware_alignment": "100 MHz antenna centre frequency from UavGPR project material",
            "is_direct_sfcw_forward_simulation": False,
        }
    )
    manifest.update(
        {
            "case_id": case_id,
            "scene_family_id": f"{manifest['scene_family_id']}_ricker100_sfcw_proxy",
            "purpose": "Fixed-geometry 100 MHz Ricker waveform ablation with an SFCW-band-equivalent postprocess.",
            "formal_training_allowed": False,
            "training_block_reason": (
                "Requires 100 MHz source dispersion, attenuation, matched-pair, and SFCW-band proxy audits; "
                "not a direct stepped-frequency forward simulation."
            ),
            "source": source_info,
            "waveform_ablation": {
                "base_case_id": json.loads(manifest_path.read_text(encoding="utf-8"))["case_id"],
                "base_scene_manifest_sha256": sha256(manifest_path),
                "base_geometry_sha256": sha256(source / str(manifest["geometry"]["index_file"])),
                "changed_files": [path.name for path in input_paths],
                "only_intended_physics_change": "Ricker centre frequency 55 MHz to 100 MHz",
            },
            "sfcw_band_equivalent_contract": {
                "frequency_band_mhz": [20.0, 170.0],
                "reported_frequency_step_mhz": 1.0,
                "reported_frequency_point_count": 501,
                "frequency_grid_status": "unconfirmed: 20-170 MHz at 1 MHz has 151 tones, so the 501 value may be output interpolation or another acquisition setting",
                "method": "post-solver raised-cosine band limitation of 100 MHz Ricker FDTD output",
                "not_direct_sfcw_forward_simulation": True,
            },
        }
    )
    manifest["spec"] = dict(manifest["spec"])
    manifest["spec"]["center_frequency_hz"] = center_frequency_hz
    manifest["grid"] = dict(manifest["grid"])
    manifest["grid"]["waveform_dispersion_reaudit_required"] = True
    manifest["grid"]["prior_cells_per_min_wavelength_note"] = (
        "This field belongs to the 55 MHz source and must not be reused for the 100 MHz ablation."
    )
    (output / "scene_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "FILE_SHA256.csv").unlink(missing_ok=True)
    write_checksums(output)
    return {
        "case_dir": str(output),
        "case_id": case_id,
        "center_frequency_hz": center_frequency_hz,
        "geometry_sha256": sha256(output / str(manifest["geometry"]["index_file"])),
        "source_manifest_sha256": sha256(output / "scene_manifest.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--center-frequency-mhz", type=float, default=100.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = create_waveform_ablation(
        args.source,
        args.output,
        center_frequency_hz=args.center_frequency_mhz * 1e6,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
