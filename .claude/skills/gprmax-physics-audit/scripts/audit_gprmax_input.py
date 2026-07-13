#!/usr/bin/env python3
"""Static, conservative audit for a gprMax input model."""

from __future__ import annotations

import argparse
import decimal
import json
import math
import re
from pathlib import Path
from typing import Any


COMMAND_RE = re.compile(r"^(#[A-Za-z0-9_]+:)\s*(.*)$")


def _float_tokens(values: list[str], count: int) -> list[float] | None:
    try:
        return [float(value) for value in values[:count]]
    except (TypeError, ValueError):
        return None


def _round_half_down(value: float) -> int:
    return int(decimal.Decimal(str(value)).quantize(decimal.Decimal("1"), rounding=decimal.ROUND_HALF_DOWN))


def parse_commands(path: Path, seen: set[Path] | None = None) -> list[dict[str, Any]]:
    seen = seen or set()
    path = path.resolve()
    if path in seen:
        return []
    seen.add(path)
    commands: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("##"):
            continue
        match = COMMAND_RE.match(line)
        if not match:
            continue
        name, body = match.groups()
        tokens = body.split()
        command = {
            "name": name.lower(),
            "tokens": tokens,
            "path": str(path),
            "line": line_number,
            "raw": line,
        }
        commands.append(command)
        if name.lower() == "#include_file:" and tokens:
            include = Path(tokens[0])
            if not include.is_absolute():
                include = path.parent / include
            if include.is_file():
                commands.extend(parse_commands(include, seen))
    return commands


def _first(commands: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((command for command in commands if command["name"] == name), None)


def audit(path: Path, max_frequency_multiplier: float = 2.8) -> dict[str, Any]:
    commands = parse_commands(path)
    errors: list[str] = []
    warnings: list[str] = []
    facts: dict[str, Any] = {}

    domain_cmd = _first(commands, "#domain:")
    grid_cmd = _first(commands, "#dx_dy_dz:")
    time_cmd = _first(commands, "#time_window:")
    for name, command in (
        ("#domain:", domain_cmd),
        ("#dx_dy_dz:", grid_cmd),
        ("#time_window:", time_cmd),
    ):
        if command is None:
            errors.append(f"missing essential command {name}")

    domain = _float_tokens(domain_cmd["tokens"], 3) if domain_cmd else None
    grid = _float_tokens(grid_cmd["tokens"], 3) if grid_cmd else None
    if domain:
        facts["domain_m"] = domain
    if grid:
        facts["grid_m"] = grid
    if time_cmd:
        try:
            facts["time_window_s"] = float(time_cmd["tokens"][0])
        except (IndexError, ValueError):
            errors.append("invalid #time_window value")

    pml_cmd = _first(commands, "#pml_cells:")
    pml = [10] * 6
    if pml_cmd:
        try:
            parsed = [int(value) for value in pml_cmd["tokens"]]
            if len(parsed) == 1:
                pml = parsed * 6
            elif len(parsed) == 6:
                pml = parsed
            else:
                errors.append("#pml_cells must contain one or six integers")
        except ValueError:
            errors.append("invalid #pml_cells value")
    facts["pml_cells"] = pml

    rounding: list[dict[str, Any]] = []
    if domain and grid:
        for axis, (extent, step) in enumerate(zip(domain, grid)):
            cells = _round_half_down(extent / step)
            realised = cells * step
            rounding.append({"kind": "domain", "axis": axis, "requested_m": extent, "cells": cells, "realised_m": realised})
            if abs(realised - extent) > max(1e-9, step * 1e-7):
                warnings.append(f"domain axis {axis} rounds from {extent:g} m to {realised:g} m")

        for command_name, offset in (("#hertzian_dipole:", 1), ("#rx:", 0), ("#src_steps:", 0), ("#rx_steps:", 0)):
            for command in (item for item in commands if item["name"] == command_name):
                values = _float_tokens(command["tokens"][offset:], 3)
                if not values:
                    continue
                for axis, (requested, step) in enumerate(zip(values, grid)):
                    cells = _round_half_down(requested / step)
                    realised = cells * step
                    rounding.append(
                        {
                            "kind": command_name.rstrip(":"),
                            "axis": axis,
                            "requested_m": requested,
                            "cells": cells,
                            "realised_m": realised,
                            "path": command["path"],
                            "line": command["line"],
                        }
                    )
                    if abs(realised - requested) > max(1e-9, step * 1e-7):
                        warnings.append(
                            f"{command_name} axis {axis} rounds from {requested:g} m to {realised:g} m "
                            f"at {command['path']}:{command['line']}"
                        )
    facts["grid_rounding"] = rounding

    materials: list[dict[str, Any]] = []
    for command in commands:
        if command["name"] != "#material:" or len(command["tokens"]) < 5:
            continue
        values = _float_tokens(command["tokens"], 4)
        if values:
            materials.append({"epsilon_r": values[0], "conductivity": values[1], "id": command["tokens"][4]})
    facts["inline_materials"] = materials

    waveforms: dict[str, float] = {}
    for command in commands:
        if command["name"] == "#waveform:" and len(command["tokens"]) >= 4:
            try:
                waveforms[command["tokens"][3]] = float(command["tokens"][2])
            except ValueError:
                warnings.append(f"cannot parse waveform frequency at {command['path']}:{command['line']}")
    facts["waveform_center_frequencies_hz"] = waveforms

    peplinski = [command for command in commands if command["name"] == "#soil_peplinski:"]
    if peplinski and waveforms:
        for frequency in waveforms.values():
            if frequency < 0.3e9 or frequency > 1.3e9:
                errors.append(
                    f"#soil_peplinski used with {frequency:g} Hz waveform outside documented 0.3-1.3 GHz validity"
                )

    imported: list[dict[str, Any]] = []
    imported_materials: list[dict[str, Any]] = []
    for command in commands:
        if command["name"] != "#geometry_objects_read:":
            continue
        if len(command["tokens"]) != 5:
            errors.append(
                f"local gprMax 3.1.7 expects five #geometry_objects_read parameters at "
                f"{command['path']}:{command['line']}"
            )
            continue
        geo_path = Path(command["tokens"][3])
        mat_path = Path(command["tokens"][4])
        source_dir = Path(command["path"]).parent
        geo_path = geo_path if geo_path.is_absolute() else source_dir / geo_path
        mat_path = mat_path if mat_path.is_absolute() else source_dir / mat_path
        record: dict[str, Any] = {"geometry": str(geo_path.resolve()), "materials": str(mat_path.resolve())}
        if not geo_path.is_file():
            errors.append(f"missing imported geometry {geo_path}")
        if not mat_path.is_file():
            errors.append(f"missing imported material map {mat_path}")
        material_count = 0
        if mat_path.is_file():
            for line in mat_path.read_text(encoding="utf-8-sig").splitlines():
                match = COMMAND_RE.match(line.strip())
                if not match or match.group(1).lower() != "#material:":
                    continue
                tokens = match.group(2).split()
                values = _float_tokens(tokens, 4)
                if values and len(tokens) >= 5:
                    imported_materials.append(
                        {"epsilon_r": values[0], "conductivity": values[1], "id": tokens[4], "source": str(mat_path)}
                    )
                    material_count += 1
            record["material_count"] = material_count
        if geo_path.is_file():
            try:
                import h5py  # type: ignore

                with h5py.File(geo_path, "r") as handle:
                    if "data" not in handle:
                        errors.append(f"{geo_path} lacks /data")
                    else:
                        data = handle["data"]
                        values = data[:]
                        record["shape"] = list(data.shape)
                        record["dtype"] = str(data.dtype)
                        record["min_index"] = int(values.min())
                        record["max_index"] = int(values.max())
                        if material_count and record["max_index"] >= material_count:
                            errors.append(
                                f"{geo_path} max material index {record['max_index']} exceeds material map limit {material_count - 1}"
                            )
                    resolution = handle.attrs.get("dx_dy_dz")
                    record["dx_dy_dz"] = list(resolution) if resolution is not None else None
                    if resolution is None:
                        errors.append(f"{geo_path} lacks dx_dy_dz root attribute")
                    elif grid and any(abs(float(a) - float(b)) > 1e-9 for a, b in zip(resolution, grid)):
                        errors.append(f"{geo_path} resolution does not match #dx_dy_dz")
            except ImportError:
                warnings.append("h5py unavailable; imported HDF5 content was not inspected")
        imported.append(record)
    facts["imported_geometry"] = imported
    facts["imported_materials"] = imported_materials

    all_materials = materials + imported_materials
    if all_materials and waveforms and grid:
        max_er = max(material["epsilon_r"] for material in all_materials)
        max_fc = max(waveforms.values())
        effective_frequency = max_fc * max_frequency_multiplier
        lambda_min = 299_792_458.0 / (effective_frequency * math.sqrt(max_er))
        ratio = lambda_min / max(grid)
        facts["max_material_epsilon_r"] = max_er
        facts["lambda_min_m"] = lambda_min
        facts["cells_per_lambda_min"] = ratio
        facts["frequency_multiplier_for_grid_check"] = max_frequency_multiplier
        if ratio < 10:
            errors.append(f"grid resolves estimated minimum wavelength with only {ratio:.2f} cells (<10)")

    if domain and grid:
        for command_name in ("#hertzian_dipole:", "#rx:"):
            for command in (item for item in commands if item["name"] == command_name):
                offset = 1 if command_name == "#hertzian_dipole:" else 0
                coords = _float_tokens(command["tokens"][offset:], 3)
                if not coords:
                    continue
                lower_cells = [coords[index] / grid[index] for index in range(3)]
                upper_cells = [(domain[index] - coords[index]) / grid[index] for index in range(3)]
                active_axes = [index for index, cells in enumerate(domain) if cells / grid[index] > 1]
                for axis in active_axes:
                    required_lower = pml[axis] + 15
                    required_upper = pml[axis + 3] + 15
                    if lower_cells[axis] < required_lower or upper_cells[axis] < required_upper:
                        warnings.append(
                            f"{command_name} at {command['path']}:{command['line']} is within 15 cells of a PML inner edge"
                        )

    fractal_commands = [command for command in commands if command["name"] in {"#fractal_box:", "#add_surface_roughness:"}]
    for command in fractal_commands:
        expected_seed_index = 13 if command["name"] == "#fractal_box:" else 12
        if len(command["tokens"]) <= expected_seed_index:
            warnings.append(f"unseeded {command['name']} at {command['path']}:{command['line']}")

    facts["command_count"] = len(commands)
    return {
        "input": str(path.resolve()),
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "facts": facts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--json", type=Path, help="Optional report path")
    parser.add_argument(
        "--max-frequency-multiplier",
        type=float,
        default=2.8,
        help="Frequency multiplier used for the conservative wavelength check",
    )
    args = parser.parse_args()
    result = audit(args.input, args.max_frequency_multiplier)
    output = json.dumps(result, indent=2)
    print(output)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(output + "\n", encoding="utf-8")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
