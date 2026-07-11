#!/usr/bin/env python3
"""Create an evidence-first Pillow review pack for completed V2 controls."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "outputs" / "simulation_v2_controls" / "official_audited_20260711"
DEFAULT_REPORT_DIR = ROOT / "reports" / "simulation_v2_control_stage_20260711" / "postrun_review"
PANEL_W, PANEL_H = 700, 410


def load_array(labels: Path, name: str) -> np.ndarray:
    return np.load(labels / name, allow_pickle=False).astype(np.float32)


def display_bscan(array: np.ndarray) -> tuple[np.ndarray, float]:
    data = array - np.median(array, axis=1, keepdims=True)
    return data, max(float(np.percentile(np.abs(data), 99.0)), 1e-8)


def raster(array: np.ndarray, limit: float, *, diverging: bool = False) -> Image.Image:
    scaled = np.clip(array / limit, -1.0, 1.0)
    if diverging:
        red = np.where(scaled >= 0, 255, (1.0 + scaled) * 255).astype(np.uint8)
        blue = np.where(scaled <= 0, 255, (1.0 - scaled) * 255).astype(np.uint8)
        green = ((1.0 - np.abs(scaled)) * 255).astype(np.uint8)
        rgb = np.dstack((red, green, blue))
    else:
        gray = ((scaled + 1.0) * 127.5).astype(np.uint8)
        rgb = np.dstack((gray, gray, gray))
    return Image.fromarray(rgb, mode="RGB").resize((PANEL_W, PANEL_H), Image.Resampling.BILINEAR)


def title(draw: ImageDraw.ImageDraw, text: str) -> None:
    draw.rectangle((0, 0, PANEL_W, 30), fill=(25, 31, 43))
    draw.text((10, 8), text, fill=(245, 247, 250), font=ImageFont.load_default())


def curve_points(values: np.ndarray) -> list[tuple[int, int]]:
    x = np.linspace(0, PANEL_W - 1, values.size).astype(int)
    y = np.clip(values / 700.0 * (PANEL_H - 1), 0, PANEL_H - 1).astype(int)
    return list(zip(x.tolist(), y.tolist()))


def support_panel(values: np.ndarray, label: str) -> Image.Image:
    image = Image.new("RGB", (PANEL_W, PANEL_H), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    title(draw, label)
    draw.line((40, PANEL_H - 35, PANEL_W - 15, PANEL_H - 35), fill=(70, 80, 90))
    draw.line((40, 45, 40, PANEL_H - 35), fill=(70, 80, 90))
    maximum = max(float(np.percentile(values, 99.0)), 1e-6)
    points = []
    for i, value in enumerate(values):
        x = 40 + int((PANEL_W - 55) * i / max(values.size - 1, 1))
        y = PANEL_H - 35 - int((PANEL_H - 85) * min(float(value) / maximum, 1.0))
        points.append((x, y))
    draw.line(points, fill=(41, 98, 255), width=2)
    draw.text((48, 50), f"median={np.median(values):.2f}; p99={np.percentile(values, 99):.2f}", fill=(25, 31, 43))
    return image


def save_case_figure(case_dir: Path, report_dir: Path) -> dict[str, Any]:
    manifest = json.loads((case_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    postprocess = json.loads((case_dir / "postprocess_validation.json").read_text(encoding="utf-8"))
    labels = case_dir / "labels"
    case_id = str(manifest["case_id"])
    target_presence = bool(manifest["target_presence"])
    full = load_array(labels, "full_scene_501x256.npy")
    air = load_array(labels, "air_reference_501x256.npy")
    display, limit = display_bscan(full)
    panels: list[Image.Image] = []
    panel = raster(display, limit)
    title(ImageDraw.Draw(panel), "Full scene: trace-median suppressed")
    panels.append(panel)

    if target_presence:
        contrast = load_array(labels, "contrast_response_501x256.npy")
        visible = load_array(labels, "visible_phase_time_ns.npy")
        reference = load_array(labels, "reference_arrival_time_ns.npy")
        support = load_array(labels, "visible_phase_support_ratio.npy")
        mask = load_array(labels, "target_mask_visible_phase_501x256.npy")
        contrast_display, contrast_limit = display_bscan(contrast)
        panel = raster(contrast_display, contrast_limit, diverging=True)
        title(ImageDraw.Draw(panel), "Matched contrast: full - no-basal")
        panels.append(panel)
        panel = raster(display, limit)
        draw = ImageDraw.Draw(panel)
        draw.line(curve_points(reference), fill=(39, 218, 229), width=2)
        draw.line(curve_points(visible), fill=(255, 78, 78), width=3)
        title(draw, "Phase overlay: red=visible, cyan=geometry")
        panels.append(panel)
        panels.append(support_panel(support, "Visible-phase support ratio"))
        support_median = float(np.median(support))
        target_nonzero = int(np.count_nonzero(mask))
        if bool(postprocess.get("reference_is_exact_flat_layered_model")):
            phase_contract_ok = float(postprocess["max_abs_trace_residual_after_phase_offset_ns"]) <= float(postprocess["arrival_tolerance_ns"])
        else:
            phase_contract_ok = bool(
                support_median > 1.0
                and postprocess.get("visible_curve_abs_step_p95_ns", np.inf) <= 14.0
                and postprocess.get("visible_curve_abs_step_max_ns", np.inf)
                <= postprocess.get("max_visible_step_tolerance_ns", 5.6) + 1e-6
            )
    else:
        difference, difference_limit = display_bscan(full - air)
        panel = raster(difference, difference_limit, diverging=True)
        title(ImageDraw.Draw(panel), "Full scene - air reference")
        panels.append(panel)
        mask = load_array(labels, "target_mask_confirmed_negative_501x256.npy")
        panel = Image.new("RGB", (PANEL_W, PANEL_H), (10, 14, 22))
        draw = ImageDraw.Draw(panel)
        title(draw, "Confirmed target mask: expected all zero")
        draw.text((30, 80), f"nonzero pixels = {np.count_nonzero(mask)}", fill=(240, 245, 250), font=ImageFont.load_default())
        draw.text((30, 105), "No target curve is drawn for this negative control.", fill=(240, 245, 250), font=ImageFont.load_default())
        panels.append(panel)
        residual = np.sqrt(np.mean(difference**2, axis=0))
        panels.append(support_panel(residual, "Residual full-air RMS by trace"))
        panel = raster(display, limit)
        title(ImageDraw.Draw(panel), "Full scene (negative control context)")
        panels.append(panel)
        support_median = None
        target_nonzero = int(np.count_nonzero(mask))
        phase_contract_ok = target_nonzero == 0

    canvas = Image.new("RGB", (PANEL_W * 2, PANEL_H * 2 + 44), (235, 239, 244))
    for index, panel in enumerate(panels):
        canvas.paste(panel, ((index % 2) * PANEL_W, 44 + (index // 2) * PANEL_H))
    header = ImageDraw.Draw(canvas)
    header.rectangle((0, 0, canvas.width, 44), fill=(18, 27, 42))
    header.text(
        (12, 14),
        f"{case_id} | gprMax {postprocess.get('gprmax_version')} | postprocess PASS | human physical review required",
        fill=(245, 247, 250),
        font=ImageFont.load_default(),
    )
    figure_path = report_dir / f"{case_id}_review.png"
    canvas.save(figure_path)
    review_ready = bool(postprocess.get("ok") and postprocess.get("postprocess_validated") and phase_contract_ok)
    return {
        "case_id": case_id,
        "target_presence": target_presence,
        "postprocess_ok": bool(postprocess.get("ok")),
        "gprmax_version": str(postprocess.get("gprmax_version")),
        "review_ready": review_ready,
        "formal_training_allowed": False,
        "figure": str(figure_path),
        "support_median": support_median,
        "target_mask_nonzero_count": target_nonzero,
        "phase_contract_ok": phase_contract_ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    args = parser.parse_args()
    root, report_dir = Path(args.root).resolve(), Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    cases = sorted(path for path in root.iterdir() if (path / "postprocess_validation.json").is_file())
    if not cases:
        raise SystemExit(f"No postprocessed controls found under {root}")
    results = [save_case_figure(case, report_dir) for case in cases]
    summary = {"root": str(root), "case_count": len(results), "review_ready_count": sum(item["review_ready"] for item in results), "formal_training_allowed": False, "results": results}
    (report_dir / "review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (report_dir / "review_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
