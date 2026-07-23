# SHAPE01 superseded decision

Date: 2026-07-20

## Decision

The SHAPE01 basal-geometry pilot and its BS02 canonical-32 run are retained as
development evidence, but are superseded for batch design and are not eligible
for training, release, or shape-family promotion.

```text
status = superseded_diagnostic
formal_training_allowed = false
promotion_allowed = false
delete_outputs = false
successor_contract = BASAL_SHAPE_BATCH_V2_SHAPE02
```

## Reasons

1. The 32 canonical traces were contiguous and covered only 2.79 m of the
   22.95 m native aperture. They sampled the beginning of the scene instead of
   the full basal morphology, so a broad feature appeared nearly linear.
2. `BS02_BROAD_RISE` is misnamed. Its formula increases depth near the centre,
   which is a broad trough/depression rather than a bedrock rise.
3. The four cases used shape-specific transition-thickness seeds. Basal shape
   was therefore not the only changed physical factor.
4. The pilot mixed calibration shapes and morphology candidates without a
   frozen factor table or staged release rule.

The solver outputs remain useful for regression, failure analysis, and proving
that the basal contrast was causal in the matched pair. They must not be copied
to a trainable or accepted dataset path.

