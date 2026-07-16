#!/usr/bin/env python3
"""Build auditable measured-reference segments for simulation calibration."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CORE_LINES = ("Line3", "Line6", "Line7", "Line9", "LineL1")
EXCLUDED_LINES = ("LineX1",)
PRIMARY_FORMAL_ROLES = {
    "Line3": "fit",
    "Line6": "validation",
    "Line7": "fit",
    "Line9": "held_out_test",
    "LineL1": "fit",
    "LineX1": "review_only",
}


@dataclass(frozen=True)
class Segment:
    line: str
    purpose: str
    start: int
    end: int
    reason: str

    @property
    def trace_count(self) -> int:
        return self.end - self.start + 1


def contiguous_segments(mask: np.ndarray, min_traces: int) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends)
        if end - start + 1 >= min_traces
    ]


def read_masks(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        status = np.asarray(data["status_code"])
        weight = np.asarray(data["label_weight"])
        ignored = np.asarray(data["v15_final_ignore_trace"], dtype=bool)
        outside_height = np.asarray(
            data["flight_height_outside_planned_2_20_m"], dtype=bool
        )

    acquisition_valid = ~ignored & ~outside_height
    morphology_valid = acquisition_valid & (status == 1) & (weight > 0)
    stress_only = outside_height & ~ignored
    return acquisition_valid, morphology_valid, stress_only


def build_segments(lines_dir: Path, min_traces: int) -> list[Segment]:
    segments: list[Segment] = []
    for path in sorted(lines_dir.glob("*.npz")):
        line = path.stem
        acquisition_valid, morphology_valid, stress_only = read_masks(path)

        if line in EXCLUDED_LINES:
            for start, end in contiguous_segments(acquisition_valid, min_traces):
                segments.append(
                    Segment(
                        line,
                        "review_only",
                        start,
                        end,
                        "X1 is excluded from calibration and promotion decisions",
                    )
                )
            continue

        if line not in CORE_LINES:
            continue

        for start, end in contiguous_segments(acquisition_valid, min_traces):
            segments.append(
                Segment(
                    line,
                    "signal_style",
                    start,
                    end,
                    "valid acquisition, outside V15 ignore and planned-height exceptions",
                )
            )
        for start, end in contiguous_segments(morphology_valid, min_traces):
            segments.append(
                Segment(
                    line,
                    "interface_morphology",
                    start,
                    end,
                    "V15 strong label with positive weight and valid acquisition",
                )
            )
        for start, end in contiguous_segments(stress_only, min_traces):
            segments.append(
                Segment(
                    line,
                    "stress_only",
                    start,
                    end,
                    "measured flight height outside planned 2-20 m range",
                )
            )
    return segments


def write_outputs(
    segments: list[Segment], output_dir: Path, source_dir: Path, min_traces: int
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "measured_reference_segments.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "segment_id",
                "line",
                "purpose",
                "trace_start",
                "trace_end",
                "trace_count",
                "development_all_lines_allowed",
                "paper_line9_holdout_role",
                "reason",
            ),
        )
        writer.writeheader()
        for index, segment in enumerate(segments, start=1):
            formal_role = PRIMARY_FORMAL_ROLES[segment.line]
            if segment.purpose in {"stress_only", "review_only"}:
                formal_role = segment.purpose
            writer.writerow(
                {
                    "segment_id": f"MR{index:03d}",
                    "line": segment.line,
                    "purpose": segment.purpose,
                    "trace_start": segment.start,
                    "trace_end": segment.end,
                    "trace_count": segment.trace_count,
                    "development_all_lines_allowed": str(
                        segment.purpose in {"signal_style", "interface_morphology"}
                    ).lower(),
                    "paper_line9_holdout_role": formal_role,
                    "reason": segment.reason,
                }
            )

    counts: dict[str, dict[str, int]] = {}
    for segment in segments:
        counts.setdefault(segment.line, {}).setdefault(segment.purpose, 0)
        counts[segment.line][segment.purpose] += segment.trace_count

    summary = {
        "contract_id": "PGDA_MULTILINE_MEASURED_REALISM_REFERENCE_V1",
        "source_dataset": "YINGSHAN_V15_FINAL_20260710",
        "source_lines_dir": source_dir.as_posix(),
        "minimum_contiguous_segment_traces": min_traces,
        "visual_ranking_lock": [
            "FORMAL06C_SUBTLE_INTERFACE_DEVELOPMENT",
            "IV2_F02_FORMAL06C_MECHANISM_TRANSFER_DEVELOPMENT",
            "IV2_F03_INSTRUMENT_BAND_WEAK_INTERFACE_PILOT",
        ],
        "development_all_lines": {
            "purpose": "measured-domain simulator development",
            "reference_lines": list(CORE_LINES),
            "excluded_lines": list(EXCLUDED_LINES),
            "line9_conditioned": True,
            "strict_unseen_line9_claim_allowed": False,
        },
        "paper_line9_holdout": {
            "fit_lines": ["Line3", "Line7", "LineL1"],
            "validation_lines": ["Line6"],
            "held_out_test_lines": ["Line9"],
            "review_only_lines": ["LineX1"],
            "line9_conditioned": False,
            "selection_must_finish_before_line9_visual_or_metric_review": True,
        },
        "usage_rules": {
            "signal_style": "frequency, background, continuity and dynamic-range calibration only",
            "interface_morphology": "basal-path and packet-morphology calibration",
            "stress_only": "robustness audit only; cannot select the nominal simulator",
            "review_only": "cannot tune or promote a simulator",
            "measured_arrays_may_be_copied_into_simulation": False,
        },
        "trace_counts_by_line_and_purpose": counts,
    }
    (output_dir / "multiline_reference_contract.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lines-dir",
        type=Path,
        default=Path("data/measured/yingshan_v15/lines"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/multiline_simulation_calibration_20260716"),
    )
    parser.add_argument("--min-traces", type=int, default=64)
    args = parser.parse_args()

    segments = build_segments(args.lines_dir, args.min_traces)
    write_outputs(segments, args.output_dir, args.lines_dir, args.min_traces)
    print(f"wrote {len(segments)} segments to {args.output_dir}")


if __name__ == "__main__":
    main()
