from pathlib import Path
import csv
import json
import math


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def read_csv_rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return math.nan
    return x


def pick_row(rows, **conds):
    for row in rows:
        ok = True
        for key, value in conds.items():
            if str(row.get(key, "")) != str(value):
                ok = False
                break
        if ok:
            return row
    return {}


def main():
    main_rows = []

    # Historical baseline from audited v1.7A stability table.
    v17 = read_csv_rows(REPORTS / "paper_v1_7a_multiseed_stability.csv")
    seed72 = pick_row(v17, seed="seed72")
    main_rows.append(
        {
            "experiment_group": "baseline",
            "model_id": "v1_7A_convnext_seed72_p065",
            "model_family": "ConvNeXt U-Net",
            "postprocess": "global-DP, P=0.65",
            "valid_avg_mae_ns": seed72.get("valid_avg_mae_ns", ""),
            "valid_avg_pick_rate": "",
            "line9_holdout_mae_ns": seed72.get("line9_holdout_mae_ns", ""),
            "line9_holdout_pick_rate": seed72.get("line9_holdout_pick_rate", ""),
            "role": "previous_primary_baseline",
            "decision": "superseded_by_v1_9D",
        }
    )

    # v1.9D preferred frozen result.
    frozen = json.loads((REPORTS / "paper_v1_9D_frozen_primary_breakable_dp_manifest.json").read_text(encoding="utf-8"))
    main_rows.append(
        {
            "experiment_group": "primary",
            "model_id": "v1_9D_mambavision_hybrid_seed1902_breakable_p050",
            "model_family": "MambaVision-style hybrid",
            "postprocess": "mask+center fusion, breakable-DP, P=0.50",
            "valid_avg_mae_ns": frozen["valid_avg_mae_ns"],
            "valid_avg_pick_rate": frozen["valid_avg_pick_rate"],
            "line9_holdout_mae_ns": frozen["line9_holdout_mae_ns"],
            "line9_holdout_pick_rate": frozen["line9_holdout_pick_rate"],
            "role": "current_frozen_primary",
            "decision": "promoted",
        }
    )
    main_rows.append(
        {
            "experiment_group": "ablation",
            "model_id": "v1_9D_mambavision_hybrid_seed1902_global_p050",
            "model_family": "MambaVision-style hybrid",
            "postprocess": "mask+center fusion, global-DP, P=0.50",
            "valid_avg_mae_ns": frozen["legacy_global_dp_valid_avg_mae_ns"],
            "valid_avg_pick_rate": 0.9415303192452825,
            "line9_holdout_mae_ns": frozen["legacy_global_dp_line9_holdout_mae_ns"],
            "line9_holdout_pick_rate": 0.5154061624649859,
            "role": "dp_ablation",
            "decision": "kept_as_ablation",
        }
    )

    # v1.9D ensemble upper bound from final summary.
    drows = read_csv_rows(REPORTS / "paper_v1_9d_mambavision_hybrid_finalist_eval_summary.csv")
    d_ens = pick_row(drows, eval_label="ensemble3seed", center_fusion_weight="0.5")
    main_rows.append(
        {
            "experiment_group": "upper_bound",
            "model_id": "v1_9D_mambavision_hybrid_3seed_ensemble_global_p065",
            "model_family": "MambaVision-style hybrid ensemble",
            "postprocess": "mask+center fusion, global-DP, P=0.65",
            "valid_avg_mae_ns": d_ens.get("valid_avg_mae_ns", ""),
            "valid_avg_pick_rate": d_ens.get("valid_avg_pick_rate", ""),
            "line9_holdout_mae_ns": d_ens.get("line9_holdout_mae_ns", ""),
            "line9_holdout_pick_rate": d_ens.get("line9_holdout_pick_rate", ""),
            "role": "upper_bound_not_primary",
            "decision": "comparison_only",
        }
    )

    # v1.9A unstable SSM ablation.
    arows = read_csv_rows(REPORTS / "paper_v1_9a_vmamba_lite_finalist_eval_summary.csv")
    a_seed = pick_row(arows, eval_label="seed1902", center_fusion_weight="0.5")
    a_ens = pick_row(arows, eval_label="ensemble3seed", center_fusion_weight="0.5")
    for row, model_id, role, decision in [
        (a_seed, "v1_9A_vmamba_lite_seed1902_global_p065", "strong_seed_ablation", "not_promoted_seed_sensitive"),
        (a_ens, "v1_9A_vmamba_lite_3seed_ensemble_global_p065", "stability_check", "not_promoted"),
    ]:
        main_rows.append(
            {
                "experiment_group": "ablation",
                "model_id": model_id,
                "model_family": "VMamba-lite",
                "postprocess": "mask+center fusion, global-DP, P=0.65",
                "valid_avg_mae_ns": row.get("valid_avg_mae_ns", ""),
                "valid_avg_pick_rate": row.get("valid_avg_pick_rate", ""),
                "line9_holdout_mae_ns": row.get("line9_holdout_mae_ns", ""),
                "line9_holdout_pick_rate": row.get("line9_holdout_pick_rate", ""),
                "role": role,
                "decision": decision,
            }
        )

    short_rows = read_csv_rows(REPORTS / "paper_v1_9_candidate_shorttrain_summary.csv")
    appendix_rows = []
    for row in short_rows:
        appendix_rows.append(
            {
                "experiment_group": "stage2_short_train",
                "model_id": f"{row.get('model_arch')}_w{row.get('center_fusion_weight')}",
                "model_family": row.get("model_arch", ""),
                "postprocess": row.get("curve_source", ""),
                "valid_avg_mae_ns": row.get("valid_avg_mae_ns", ""),
                "valid_avg_pick_rate": row.get("valid_avg_pick_rate", ""),
                "line9_holdout_mae_ns": row.get("line9_holdout_mae_ns", ""),
                "line9_holdout_pick_rate": row.get("line9_holdout_pick_rate", ""),
                "role": "candidate_screening",
                "decision": row.get("status", ""),
            }
        )

    fieldnames = list(main_rows[0].keys())
    out_main = REPORTS / "paper_v1_9_experiment_matrix_main.csv"
    with out_main.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(main_rows)

    out_appendix = REPORTS / "paper_v1_9_experiment_matrix_appendix_shorttrain.csv"
    with out_appendix.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(appendix_rows)

    summary = {
        "main_matrix": str(out_main.relative_to(ROOT)),
        "appendix_shorttrain_matrix": str(out_appendix.relative_to(ROOT)),
        "current_primary": "v1_9D_mambavision_hybrid_seed1902_breakable_p050",
        "primary_valid_avg_mae_ns": frozen["valid_avg_mae_ns"],
        "primary_line9_holdout_mae_ns": frozen["line9_holdout_mae_ns"],
        "decision": "Use v1.9D breakable-DP as frozen primary; keep v1.9A and ensemble as comparisons.",
    }
    out_json = REPORTS / "paper_v1_9_experiment_matrix_summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out_main)
    print(out_appendix)
    print(out_json)


if __name__ == "__main__":
    main()
