from pathlib import Path
import copy
import json


ROOT = Path(__file__).resolve().parents[1]
SEEDS = [1901, 1902, 1903]
FINALISTS = [
    ("v1_9d_mambavision_hybrid", "v1.9D finalist: best short-train Line9 holdout; risk is weak valid-line average.", {"ssm_kernel": 31, "attention_heads": 4}),
    ("v1_9a_vmamba_lite", "v1.9A finalist: best short-train valid-line average among new SSM-style candidates.", {"ssm_kernel": 31}),
]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    base = json.loads((ROOT / "configs" / "v1_9_candidate_base.json").read_text(encoding="utf-8"))
    configs = []
    for arch, note, extra in FINALISTS:
        for seed in SEEDS:
            cfg = copy.deepcopy(base)
            cfg.update(
                {
                    "model_arch": arch,
                    "epochs": 80,
                    "seed": seed,
                    "max_preview_val": 0,
                    "run_dir": f"outputs/run_gpu_paper_{arch}_final_seed{seed}_line9holdout",
                    "version": f"paper_{arch}_final_seed{seed}_line9holdout",
                    "note": note,
                }
            )
            cfg.update(extra)
            path = ROOT / "configs" / f"gpu_train_paper_{arch}_final_seed{seed}_line9holdout.json"
            write_json(path, cfg)
            configs.append(str(path.relative_to(ROOT)).replace("\\", "/"))
            print(configs[-1])
    write_json(ROOT / "reports" / "paper_v1_9_finalist_config_manifest.json", configs)


if __name__ == "__main__":
    main()
