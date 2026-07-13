# AeroPath-SSD V15 Data Closure Debug Run

Date: 2026-07-13

## Purpose

This is a one-step engineering closure, not a paper experiment. It verifies
the V15 loader, trace-resolved metadata, official Mamba2 model, structured loss,
AMP backward pass, optimiser update, checkpoint creation, and validation path.

## Guardrails

- Configuration: `configs/aeropath_ssd_v15_data_closure_debug.json`
- `run_type=debug`
- `allow_incomplete_dataset=true` is explicit and limited to this configuration.
- One train batch from Line3 and one validation batch from Line6.
- No test line and no simulation data were consumed.
- Line9 remains untouched; LineX1 remains review-only.
- The V15 dataset policy remains `training_allowed=false`.

## Accepted result

- Backend: official `mamba-ssm 2.2.6` on RTX 5070 under AMP.
- Input: 501 x 256, six channels: raw plus five terrain/acquisition channels.
- Training and validation each completed one batch.
- Checkpoints, history, configuration audit, split audit, and previews were
  written under `outputs/run_debug_aeropath_v15_data_closure/`.
- Train loss: `20.50665283203125`.
- Validation loss: `19.938846588134766`.
- Structured path supervision: 256 valid trace labels in each batch.
- No-pick supervision: zero valid windows, as expected.

## Fix discovered by the closure

Probability BCE losses for NULL and path-boundary posteriors were previously
executed inside CUDA autocast. PyTorch rejects that operation because BCE on
probabilities is unsafe in half precision. They now run in FP32 through
`_probability_bce`; logit-based no-pick BCE remains AMP-safe.

## Formal-release status

`scripts/check_dataset.py --data-root data_yingshan_v15_final_20260710` reports
structural success but `formal_ready=false`. The remaining blockers are still:

1. zero confirmed `status_code=0` true-negative traces;
2. no approved independent non-Line9-conditioned simulations; and
3. the V15 policy remains a final-label release rather than a formal-training
   release.

This debug result must not be used in a paper table or compared as a formal
Line9 holdout result.
