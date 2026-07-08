# PGDA-CSNet Current State (2026-07-08)

> **Purpose**: Quick-start document for new AI agents. Read this first, then dive into referenced code files.

---

## 1. Core Architecture: GprMambaSep v2.1

**Code**: `pgdacsnet/model_gprmambasep.py` → `GprMambaSep` class  
**Dispatch**: `pgdacsnet/model_raw_unet.py` → `build_model(cfg)` routes `model_arch=v2_1_gprmambasep_lite` to GprMambaSep  
**Interface**: `pgdacsnet/model_interfaces.py` → `GprMambaSepOutput`

### Architecture (simplified)

```
Input (B,1,H,W) radar + (optional) aux channels
  └─ ConvNeXt Stem → 4 Stages (ch 64→128→256→512)
       └─ AxialSSMLiteBlock (content-adaptive SSM mixers)
            └─ Split Bottleneck → 3 _ComponentDecoders
                 ├─ Decoder_A → A_hat (air wave)
                 ├─ Decoder_S → S_hat (surface)
                 └─ Decoder_G → g_task_feat → task heads:
                      ├─ mask_logits   (B,1,H,W) — per-pixel target
                      ├─ presence_logits (B,1,W) — per-trace presence
                      └─ center_logits (B,1,H,W) — per-column center depth
```

### Key design decisions
- **Explicit A/S/G decomposition**: Three `_ComponentDecoder`s with `_SkipFuse` gated skips
- **`component_gate`**: Softmax over pooled decoder features → per-trace A/S/G dominance diagnostic
- **AMP**: `torch.amp.GradScaler` makes 512×256 production resolution fit on 6GB RTX 3060
- **Validity-masked losses**: `*_valid` flags for mixed sim/real batches
- **PGDA output contract**: `mask_logits` (per-pixel) / `presence_logits` (per-trace 1D) / `center_logits` (per-column)
- **GprMambaSepOutput**: 6-tuple `(mask, presence, center, A_hat, S_hat, G_hat)` + dict/attribute access + backward-compat aliases

### Loss functions (`scripts/losses_gprmambasep.py`)

| Loss | Config Key | Status | What it does |
|------|-----------|--------|-------------|
| Segmentation (BCE+Dice+Core) | `core_weight`, `dice_weight` | ✅ Active | Standard PGDA mask supervision |
| Per-trace presence | `presence_weight` | ✅ Active | Per-trace BCE with negative weighting |
| Centerline L1 | `centerline_weight` | ✅ Active | Depth regression |
| Self-consistency | `self_consistency_weight` | ✅ Active | A_hat+S_hat+G_hat ≈ Y_full |
| L2 component supervision | `sim_supervised_weight` | ❌ OFF (0.0) | Requires Y_air/X_clean/G_target arrays |
| Arrival time prior | `arrival_prior_weight` | ✅ Active | Penalizes G_hat energy before expected arrival |
| Amplitude ratio prior | `amplitude_ratio_weight` | ✅ Active | Constrains A/S/G relative amplitudes |

---

## 2. Current Training Results

### Stage 1: Sim-only pretrain
- **Config**: `configs/gpu_pretrain_v2_1_gprmambasep_lite.json`
- **Data**: `data/simulation_pretrain_v1/` — 192 NPZ windows, 33 simulated lines
- **Settings**: lr=3e-4, epochs=80, batch_size=1, AMP on, self_consistency_weight=2.0
- **Result**: Stable convergence. Produced `outputs/run_gprmambasep_lite_pretrain_v2_1/checkpoint_best.pt` used as Stage 2 warm-start.

### Stage 2: Mixed sim-real, Line9 strict holdout
- **Config**: `configs/gpu_mixed_v2_1_gprmambasep_lite_line9holdout.json`
- **Train lines**: Line3, Line6, Line7, LineL1 + sim v1 (sim_batch_ratio=0.3)
- **Test line**: Line9 (full 2377 traces, strictly held out — zero training traces)
- **Settings**: lr=1e-4, epochs=80, AMP on, sim_supervised_weight=0.0

### Line9 Holdout Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| DP Center MAE | **25.24 ns** | P0-3 baseline: **3.27 ns** — 7.7× worse |
| Pick Rate | **0.56** | Model picks on ~56% of traces |
| Mean Center MAE | **66.72 ns** | All traces (picked + rejected) |
| IoU@0.2 | **0.073** | Very low spatial overlap |
| Self-consistency loss | **0.003** | Near-perfect A_hat+S_hat+G_hat ≈ Y_full |

### Critical Diagnosis

> **The model decomposes correctly but localizes incorrectly.**

- Self-consistency = 0.003 proves the A/S/G branches together fully reconstruct the input. The decomposition is **not collapsing**.
- But G_hat does not correspond to the correct geological interface. It self-organizes into an unconstrained residual bucket.
- The model picks the right traces (56% pick rate ≈ reasonable for a challenging line) but predicts the wrong depth (25.24 ns vs 3.27 ns baseline).
- **Root cause**: Component supervision (`sim_supervised_weight=0.0`) is disabled because no NPZ contains Y_air/X_clean/G_target arrays.

---

## 3. Data Bottleneck

### Current datasets

| Dataset | Used In | NPZ count | Has component arrays? | Source |
|---------|---------|-----------|----------------------|--------|
| `data/simulation_pretrain_v1/` | Stages 1,2 | 192 | ❌ No Y_air/X_clean/G_target | 20 Pilot-Mini gprMax cases |
| `data_corrected_v1_4_terrain_direction/` | Stage 2 | 78 | ❌ No | 6 real survey lines |
| `data/PGDA_SYNTH_DATASET_V1/05_accepted_dataset/` | Not used | varies | ❌ No | LINE9_LABEL_INSPIRED_V1 + batch_001 |

### What component arrays are needed

Per the locked signal model:

| Array | What it supervises | How to get it |
|-------|-------------------|--------------|
| `Y_air` (A only) | A_hat | gprMax air_only simulation (same geometry, target absent) |
| `X_clean` (S+G) | S_hat + G_hat | Y_target - Y_air, or raw - background_only |
| `G_target` (G only) | G_hat | gprMax basal_only simulation (S and A absent) |

### Planned solution: Batch4 paired simulations

Generate 24 scene families, each with:
- `raw_full.in` → Y_full
- `background_only.in` → A+S (target removed from geometry)
- `basal_only.in` → G (air replaced with ground material)

Pilot with 8 families first, then full 24.

---

## 4. Code Map (for AI navigation)

| File | Role | Key functions/classes |
|------|------|---------------------|
| `pgdacsnet/model_gprmambasep.py` | Main architecture | `GprMambaSep`, `_ComponentDecoder`, `_SkipFuse`, `build_gprmambasep()` |
| `pgdacsnet/model_interfaces.py` | Output contracts | `GprMambaSepOutput`, `PGDAOutput`, `unpack_pgda_output()` |
| `pgdacsnet/model_mamba.py` | SSM implementations | `AxialSSMLiteBlock`, `SelectiveSSMLite`, `make_axial_sequence_block()` |
| `pgdacsnet/model_raw_unet.py` | Model dispatcher | `build_model(cfg)` — routes arch names to constructors |
| `scripts/losses_gprmambasep.py` | Loss functions | `compute_gprmambasep_loss()`, `self_consistency_loss()`, `sim_supervised_component_loss()` |
| `scripts/train_raw_only.py` | Training loop | `DS.__getitem__`, `run_epoch()`, `compute_loss()`, component array loading |
| `scripts/eval_full_line.py` | Full-line evaluation | Stitched prediction, DP path, metrics CSV |
| `scripts/eval_gprmambasep_separation.py` | Component QC | `compute_separation_metrics()`, `save_diagnostic_plot()` |
| `tests/test_gprmambasep.py` | 40 tests | Architecture shapes, loss behavior, valid flags |
| `configs/gpu_pretrain_v2_1_gprmambasep_lite.json` | Stage 1 config | Sim-only pretrain |
| `configs/gpu_mixed_v2_1_gprmambasep_lite_line9holdout.json` | Stage 2 config | Mixed sim-real, Line9 holdout |

---

## 5. Immediate Next Steps

### Priority 1: Inject component arrays into existing data
- Run `scripts/inject_component_arrays_to_pretrain_v1.py` (currently broken — shape mismatches)
- Or regenerate simulation NPZs with component fields via `convert_pilot_to_training.py` extension

### Priority 2: Enable component supervision pilot
- Set `sim_supervised_weight=0.05` in config (small weight to start)
- Enable `Y_air → A_hat` and `X_clean → S_hat+G_hat` supervision
- Verify G_hat shifts toward correct geological interface

### Priority 3: Generate Batch4 data
- 8 family pilot → validate → 24 family full batch
- Each family: `raw_full + background_only + basal_only`
- Convert to NPZ with Y_air/X_clean/G_target fields

### Priority 4: Add missing losses (per deep-research report)
- G_band loss — constrain G_hat energy to target time window
- Distractor loss — penalize G_hat activation on shallow/hard-negative windows
- Global no-target head — window-level classification

---

## 6. Historical Context

- **Old architecture** (`v1_9d_mambavision_hybrid`, UNet-based): archived at `legacy/unet-arch` branch
- **P0-3 Baseline**: Center fusion heuristic, DP MAE=3.27ns, PR=0.56 — no A/S/G decomposition
- **X Pattern finding**: Air wave A + Surface reflection S create X-shaped interference in raw B-scans. Root cause: large velocity contrast between air (0.3 m/ns) and ground (0.07-0.12 m/ns). Production model must retain air layer.
- **Simulation management**: 5-tier governance under `data/PGDA_SYNTH_DATASET_V1/` — templates → runs → QC → accepted → archives
