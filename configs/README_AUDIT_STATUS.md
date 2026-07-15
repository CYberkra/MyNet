# Active configuration set

Only the current AeroPath experiment contract is retained on `master`.

| File | Purpose | Status |
|---|---|---|
| `aeropath_ssd_smoke.json` | CPU-friendly one-step implementation smoke | enabled, debug only |
| `aeropath_ssd_v15_data_closure_debug.json` | Official-Mamba2 V15 one-step CUDA/data closure | enabled, debug only |
| `aeropath_ssd_v15_formal_blocked.json` | Locked 501x256 paper protocol | disabled by data gate |
| `paper_splits_v15_aeropath.json` | Authoritative whole-line measured split | locked |

The formal config is blocked only by missing confirmed real negatives and
missing approved independent V2 simulations. V15 labels and the split are
complete. Historical GprMambaSep, LOO, and superseded pilot configs are
available from `archive/pre-master-cleanup-20260715`.
