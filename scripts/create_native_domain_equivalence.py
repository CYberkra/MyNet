#!/usr/bin/env python3
"""Create an exact cropped/shifted native-256 protected-window source deck.

The source case is left untouched. The resulting deck uses the same acquired
22.95 m window, same material maps, and the same voxel values in that window;
only remote lateral cells are removed. It is intended for a 0-500 ns
protected-window equivalence test, not as a 0-700 ns boundary-isolated model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_native_256_correlated_voxel_batch import Spec, input_text, write_checksums  # noqa: E402


C0 = 299_792_458.0
DEFAULT_SOURCE = (
    ROOT / "data" / "simulations" / "v2" / "01_native_256_correlated_voxel_batch_v1"
    / "N256_CV01_BALANCED_MULTISCALE_POS"
)
DEFAULT_OUTPUT = (
    ROOT / "data" / "simulations" / "v2" / "02_native_256_domain_equivalence_v1"
    / "N256_CV01_80M_DOMAIN_EQUIVALENCE"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_labels(source: Path, destination: Path, *, crop_start: int, crop_stop: int, shift_m: float) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    trace_x_fields = {"source_x_m.npy", "receiver_x_m.npy", "trace_midpoint_x_m.npy"}
    full_x_fields = {"full_x_m.npy"}
    for path in source.glob("*.npy"):
        values = np.load(path, allow_pickle=False)
        if path.name in trace_x_fields:
            values = values - np.asarray(shift_m, dtype=values.dtype)
        elif path.name in full_x_fields:
            values = values[crop_start:crop_stop] - np.asarray(shift_m, dtype=values.dtype)
        np.save(destination / path.name, values)


def _copy_materials(source: Path, destination: Path) -> None:
    for name in ("materials_full.txt", "materials_no_basal.txt"):
        source_path = source / name
        if source_path.is_file():
            shutil.copy2(source_path, destination / name)


def _cropped_case_id(base_case_id: str) -> str:
    if base_case_id == "N256_CV01_BALANCED_MULTISCALE_POS":
        return "N256_CV01_80M_DOMAIN_EQUIVALENCE"
    return f"{base_case_id}_80M"


def create_equivalence(source: Path, output: Path, *, crop_cells: int, overwrite: bool) -> dict[str, object]:
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output}")
        shutil.rmtree(output)
    manifest_path = source / "scene_manifest.json"
    base = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_case_id = str(base.get("case_id", ""))
    if not base_case_id.startswith("N256_CV"):
        raise ValueError(f"expected a native correlated-voxel source case, got {base_case_id!r}")
    target_presence = bool(base.get("target_presence"))
    grid = base["grid"]
    dl = float(grid["dl_m"])
    old_nx = int(grid["nx_ny_nz"][0])
    ny = int(grid["nx_ny_nz"][1])
    if crop_cells <= 0 or crop_cells * 2 >= old_nx:
        raise ValueError("crop_cells must leave a positive domain")
    crop_start, crop_stop = crop_cells, old_nx - crop_cells
    new_nx = crop_stop - crop_start
    shift_m = crop_cells * dl
    new_domain_x = new_nx * dl
    old_scan_x = float(base["spec"]["scan_start_x_m"])
    new_scan_x = old_scan_x - shift_m
    trace_count = int(grid["trace_count"])
    trace_spacing = float(grid["trace_spacing_m"])
    offset = float(base["spec"]["tx_rx_offset_m"])
    pml_cells = int(grid["pml_cells"][0])
    pml_thickness = pml_cells * dl
    first_source_to_inner_pml = new_scan_x - pml_thickness
    last_receiver = new_scan_x + (trace_count - 1) * trace_spacing + offset
    last_receiver_to_inner_pml = (new_domain_x - pml_thickness) - last_receiver
    if min(first_source_to_inner_pml, last_receiver_to_inner_pml) < 80.0:
        raise RuntimeError("cropped model does not retain the requested 80 m inner-PML guard")
    if min(first_source_to_inner_pml, last_receiver_to_inner_pml) < 75.0:
        raise RuntimeError("cropped model violates the absolute 75 m guard floor")

    output.mkdir(parents=True)
    source_h5 = source / "geology_indices.h5"
    output_h5 = output / "geology_indices.h5"
    with h5py.File(source_h5, "r") as src, h5py.File(output_h5, "w") as dst:
        data = src["data"][crop_start:crop_stop, :, :]
        if data.shape != (new_nx, ny, 1):
            raise RuntimeError(f"unexpected cropped geometry shape: {data.shape}")
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        dst.attrs["domain_equivalence_source_sha256"] = sha256(source_h5)
        dst.attrs["domain_equivalence_crop_start_cells"] = crop_start
        dst.attrs["domain_equivalence_crop_stop_cells"] = crop_stop
        dst.create_dataset(
            "data",
            data=data,
            dtype=np.int16,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            chunks=(min(256, new_nx), min(256, ny), 1),
        )
    _copy_materials(source, output)
    _copy_labels(source / "labels", output / "labels", crop_start=crop_start, crop_stop=crop_stop, shift_m=shift_m)

    spec = Spec(
        domain_x_m=new_domain_x,
        domain_y_m=float(grid["domain_y_m"]),
        dl_m=dl,
        pml_cells=pml_cells,
        trace_count=trace_count,
        trace_spacing_m=trace_spacing,
        scan_start_x_m=new_scan_x,
        source_y_m=float(base["spec"]["source_y_m"]),
        tx_rx_offset_m=offset,
        center_frequency_hz=float(base["source"]["center_frequency_hz"]),
        solver_time_window_s=float(base["spec"]["solver_time_window_s"]),
    )
    case_id = _cropped_case_id(base_case_id)
    (output / "full_scene.in").write_text(input_text(spec, f"{case_id} full", "materials_full.txt"), encoding="ascii")
    if target_presence:
        (output / "no_basal_contrast_control.in").write_text(
            input_text(spec, f"{case_id} no basal contrast", "materials_no_basal.txt"), encoding="ascii"
        )
    (output / "air_reference.in").write_text(input_text(spec, f"{case_id} air", None), encoding="ascii")
    (output / "geometry_check_full.in").write_text(
        input_text(spec, f"{case_id} geometry full", "materials_full.txt", geometry_view="geometry_full"), encoding="ascii"
    )
    if target_presence:
        (output / "geometry_check_control.in").write_text(
            input_text(spec, f"{case_id} geometry control", "materials_no_basal.txt", geometry_view="geometry_control"), encoding="ascii"
        )

    out_manifest = dict(base)
    out_manifest.update(
        {
            "case_id": case_id,
            "scene_family_id": f"{base['scene_family_id']}_80m_guard",
            "purpose": f"Exact {base_case_id} cropped/shifted 80 m inner-PML guard pilot.",
            "training_block_reason": (
                "Requires solved 0-500 ns full/control and human morphology audit."
                if target_presence
                else "Requires solved target-absent full-scene and human hard-negative audit."
            ),
            "generator_path": "scripts/create_native_domain_equivalence.py",
            "generator_sha256": sha256(Path(__file__).resolve()),
            "spec": {
                "domain_x_m": spec.domain_x_m,
                "domain_y_m": spec.domain_y_m,
                "dl_m": spec.dl_m,
                "pml_cells": spec.pml_cells,
                "trace_count": spec.trace_count,
                "trace_spacing_m": spec.trace_spacing_m,
                "scan_start_x_m": spec.scan_start_x_m,
                "source_y_m": spec.source_y_m,
                "tx_rx_offset_m": spec.tx_rx_offset_m,
                "center_frequency_hz": spec.center_frequency_hz,
                "solver_time_window_s": spec.solver_time_window_s,
            },
        }
    )
    out_manifest["grid"] = dict(grid)
    out_manifest["grid"].update(
        {
            "nx_ny_nz": [new_nx, ny, 1],
            "domain_x_m": new_domain_x,
            "left_scan_margin_m": new_scan_x,
            "right_scan_margin_m": new_domain_x - last_receiver,
            "physical_left_guard_from_inner_pml_m": first_source_to_inner_pml,
            "physical_right_guard_from_inner_pml_m": last_receiver_to_inner_pml,
            "earliest_free_space_side_roundtrip_ns": 2e9 * min(first_source_to_inner_pml, last_receiver_to_inner_pml) / C0,
            "protected_window_end_ns": 500.0,
            "unprotected_window_note": "500-700 ns remains diagnostic only in the cropped equivalence model.",
        }
    )
    out_manifest["geometry"] = dict(base["geometry"])
    out_manifest["geometry"].update(
        {
            "index_file_sha256": sha256(output_h5),
            "index_shape": [new_nx, ny, 1],
            "domain_equivalence": {
                "baseline_case_id": base["case_id"],
                "baseline_scene_manifest_sha256": sha256(manifest_path),
                "baseline_geometry_sha256": sha256(source_h5),
                "crop_start_cells": crop_start,
                "crop_stop_cells": crop_stop,
                "crop_each_side_cells": crop_cells,
                "crop_each_side_m": shift_m,
                "exact_voxel_crop": True,
                "source_receiver_shift_m": shift_m,
            },
        }
    )
    out_manifest["strict_pair"] = dict(base["strict_pair"])
    out_manifest["strict_pair"]["shared_geometry_sha256"] = sha256(output_h5)
    out_manifest["labels"] = dict(base["labels"])
    out_manifest["labels"]["domain_equivalence_status"] = (
        f"pending solved full/control audit for exact crop of {base_case_id}"
        if target_presence
        else f"pending solved target-absent audit for exact crop of {base_case_id}"
    )
    (output / "scene_manifest.json").write_text(json.dumps(out_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_checksums(output)
    return {
        "case_dir": str(output),
        "domain_x_m": new_domain_x,
        "crop_each_side_m": shift_m,
        "inner_pml_guard_m": min(first_source_to_inner_pml, last_receiver_to_inner_pml),
        "earliest_free_space_side_roundtrip_ns": 2e9 * min(first_source_to_inner_pml, last_receiver_to_inner_pml) / C0,
        "geometry_sha256": sha256(output_h5),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--crop-cells", type=int, default=1313)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(create_equivalence(args.source, args.output, crop_cells=args.crop_cells, overwrite=args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
