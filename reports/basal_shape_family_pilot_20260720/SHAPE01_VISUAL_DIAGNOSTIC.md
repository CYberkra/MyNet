# SHAPE01 sparse visual diagnostic

Date: 2026-07-20

## Scope

This diagnostic compares the causal pair `full_scene - no_basal_contrast_control` for two independent basal-interface shape families:

- `BS02_BROAD_RISE`
- `BS04_GENTLE_MULTISCALE`

The diagnostic runs use the 500 ns short window and eight traces at 32-trace stride. The physical spacing between sampled traces is therefore 2.88 m. They are visual screening runs only and are not canonical B-scans or training data.

## Visual audit method

The preview was inspected after correcting the earlier display-axis error. Solver arrays are read as `time x trace`; time is vertical and trace is horizontal. No interpolation is used for metrics. The display panels are:

1. raw full scene;
2. display-only lateral background removal plus bounded `t^1.35` gain;
3. signed causal difference `full - no_basal`.

The target zoom is 250--460 ns because the shape-induced time variation is only a few to several nanoseconds and is visually compressed in a full 500 ns panel.

## Results

### BS02

The signed difference forms a coherent broad response with a slow time shift across the sparse trace positions. Its peak times are approximately 438--450 ns in the 250--450 ns audit window. The response is regular and broad, consistent with the intended broad-rise family, but the eight-trace spacing is too coarse to assess continuity or fine morphology.

### BS04

The signed difference remains causally localized in the target window, with peak times approximately 415--425 ns for the eight sampled positions. Amplitude varies substantially between positions, including a strong response near the seventh sampled trace. The visual response is more heterogeneous than BS02, but it still cannot establish measured-like multiscale continuity at 2.88 m spacing.

## Interpretation

The pair difference is present for both shapes, so the basal contrast is not visually absent. However, neither sparse preview should be promoted as a realistic training example. A flat-looking band in the overview is partly a display-scale effect and partly a consequence of sparse sampling. The next valid morphology gate is a local canonical-spacing run over a limited span, using 0.09 m trace spacing and the same full/no-basal geometry.

The corrected previews and zooms are:

- `SHAPE01_SHORT8_BS02_bscan_preview_fixed.png`
- `SHAPE01_SHORT8_BS02_target_zoom.png`
- `SHAPE01_SHORT8_BS04_bscan_preview_fixed.png`
- `SHAPE01_SHORT8_BS04_target_zoom.png`

Static audits are recorded in `BS02_short_full_static.json` and `BS04_short_full_static.json`.

## Promotion decision

```text
formal_training_allowed = false
promotion_allowed = false
next_gate = local canonical-spacing morphology diagnostic
```

