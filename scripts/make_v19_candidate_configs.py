from pathlib import Path
import copy
import json


ROOT = Path(__file__).resolve().parents[1]


CANDIDATES = [
    ("v1_9a_vmamba_lite", "v1.9A VMamba-lite cross-scan SSM", {"ssm_kernel": 31}),
    ("v1_9b_umamba_hybrid", "v1.9B U-Mamba hybrid SSM bottleneck/decoder", {"ssm_kernel": 47}),
    ("v1_9c_stripe_attention", "v1.9C Swin/CSWin-style stripe attention", {"attention_heads": 4}),
    ("v1_9d_mambavision_hybrid", "v1.9D MambaVision-style scan plus stripe attention", {"ssm_kernel": 31, "attention_heads": 4}),
    ("v1_9e_convnext_pp", "v1.9E ConvNeXt++ strong baseline", {}),
]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    base = json.loads((ROOT / "configs" / "v1_9_candidate_base.json").read_text(encoding="utf-8"))
    manifest = []
    for arch, note, extra in CANDIDATES:
        smoke = copy.deepcopy(base)
        smoke.update(
            {
                "model_arch": arch,
                "height_resize": 64,
                "width_resize": 64,
                "base_ch": 8,
                "epochs": 2,
                "max_train_samples": 12,
                "max_val_samples": 4,
                "max_preview_val": 0,
                "force_cpu": True,
                "run_dir": f"outputs/run_smoke_paper_{arch}",
                "version": f"paper_{arch}_smoke",
                "note": note,
            }
        )
        smoke.update(extra)
        short = copy.deepcopy(base)
        short.update(
            {
                "model_arch": arch,
                "epochs": 30,
                "max_preview_val": 0,
                "run_dir": f"outputs/run_gpu_paper_{arch}_short_line9holdout",
                "version": f"paper_{arch}_short_line9holdout",
                "note": note,
            }
        )
        short.update(extra)
        smoke_path = ROOT / "configs" / f"smoke_train_paper_{arch}.json"
        short_path = ROOT / "configs" / f"gpu_train_paper_{arch}_short_line9holdout.json"
        write_json(smoke_path, smoke)
        write_json(short_path, short)
        manifest.append(
            {
                "model_arch": arch,
                "note": note,
                "smoke_config": str(smoke_path.relative_to(ROOT)).replace("\\", "/"),
                "short_config": str(short_path.relative_to(ROOT)).replace("\\", "/"),
                "ranking_policy": "LineX1 review-only; rank by valid-line average and Line9 holdout.",
            }
        )
    write_json(ROOT / "reports" / "paper_v1_9_candidate_manifest.json", manifest)
    for item in manifest:
        print(item["model_arch"], item["smoke_config"], item["short_config"])


if __name__ == "__main__":
    main()
