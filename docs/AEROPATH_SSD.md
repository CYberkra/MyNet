# AeroPath-SSD Architecture Contract

## Scope

AeroPath-SSD is the current research architecture for UAV-GPR basal-interface
picking.  Its primary prediction is a continuous, trace-conditioned interface
path, not a putative A/S/G physical decomposition.  The retained band mask is
only a compatible auxiliary head for comparison with historical segmentation
baselines.

## Inputs

The canonical input remains in CSV acquisition order.  The model receives:

- compressed raw B-scan amplitude;
- optional terrain/flight metadata channels;
- tracewise `flight_height_agl_m`, passed separately rather than inferred from
  a normalised feature map.

It constructs two amplitude views: raw time and an air-path-reduced view.  A
missing AGL value leaves that trace in raw time; it must never introduce NaN
values or trigger a fabricated default height.

## Network

1. Phase-aware anisotropic time/trace convolutional stem.
2. Metadata FiLM conditioning in every mixer block.
3. Local depthwise mixing plus bidirectional axial sequence mixing.
4. U-shaped multiscale decoder.
5. Unary interface-energy, band-mask, presence, no-pick, and log-variance
   heads.
6. A differentiable first-order soft dynamic-programming layer producing
   `path_marginals = P(time | trace)`.

`ssm_impl="official_mamba2"` is required for every formal AeroPath result.
`ssm_lite` exists solely for CPU/smoke development and must not be described
as a Mamba-2 experiment.  The config contract rejects a formal Lite run.

## Training Objective

The standard band/presence/center losses remain auxiliary.  The structured
objective adds:

- cross-entropy between soft-DP path marginals and the normalised label band;
- expected path-center L1;
- adjacent-trace path smoothness;
- heteroscedastic path uncertainty NLL using the log-variance field;
- a window no-pick BCE that is supervised only when at least one trace has a
  confirmed (non-weak) presence label.

Weak or ignored labels do not create a hard no-pick target.  Formal training
is still governed by the existing dataset contracts: confirmed negatives,
non-Line9-conditioned approved simulation, split isolation, and label-release
gates are all independent requirements.

## Inference Contract

`scripts/eval_full_line.py` forwards canonical tracewise AGL values when
available and writes semantically separate artifacts:

- `*_mask_prob.npy`: sigmoid pixel-mask probability;
- `*_curve_prob.npy`: unary curve distribution when available;
- `*_structured_path_prob.npy`: AeroPath soft-DP marginal, used for path DP;
- `*_path_log_variance.npy`: path uncertainty field;
- `*_no_pick_prob.npy`: stitched no-interface probability.

Only the mask artifact is eligible for Dice/IoU/BCE.  Path metrics use the
structured distribution and report expected/DP errors, coverage, hit rates,
and uncertainty risk-coverage diagnostics.  `no_pick_prob` can reject a
candidate path using `--no-pick-thr`; it is never silently fused with the mask.

## Configuration

Use `configs/aeropath_ssd_smoke.json` for implementation tests only.  It is
disabled on purpose and uses `ssm_lite`.  A future formal configuration must
explicitly select `official_mamba2`, enable structured loss, and pass the
project's formal data-release gate before any paper experiment is started.
