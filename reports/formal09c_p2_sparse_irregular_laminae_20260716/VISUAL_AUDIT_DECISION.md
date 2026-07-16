# FORMAL09C-P2 Visual Audit Decision

## Decision

`FORMAL09C_P2_SPARSE_IRREGULAR_FINITE_LAMINAE` is accepted as a conservative
sparse physical-lamina factor, but it is not promoted as a measured-realism
baseline and remains blocked from training.

P2 should be reimplemented as an optional factor in a fold-safe independent
family. The complete P2 scene must not be copied because its FORMAL06C
predecessor remains Line9-conditioned.

## Evidence

- Static full/control audits: pass with zero errors and zero warnings.
- Geometry-only build: pass; transient VTI files were hashed and deleted.
- One-trace strict full/no-basal pair: pass.
- Native-spacing full scene: 64 consecutive traces, complete and finite.
- Exact predecessor comparison: the same 64 canonical traces from FORMAL06C,
  with identical source, materials, grid, acquisition, basal path, transition,
  processing, and display scales.
- Equal-width measured comparison: 64-trace windows from Line3, Line6, Line7,
  Line9, and LineL1. Line9 is held-out diagnostic evidence only.

## Numerical result

| Metric | FORMAL06C | FORMAL09C-P1 | FORMAL09C-P2 |
|---|---:|---:|---:|
| Basal path/geometric correlation | 0.8486 | 0.8453 | 0.8443 |
| Target/adjacent-background RMS | 10.4029 | 5.6106 | 10.3196 |
| Target dropout fraction | 0.0 | 0.0 | 0.0 |
| Significant signed lobes | 7 | 7 | 7 |
| Peak frequency | 79.37 MHz | 79.37 MHz | 79.37 MHz |
| Median aligned-template correlation | 0.7956 | 0.8059 | 0.8053 |

P2 restores 99.2% of the FORMAL06C basal target-to-background ratio and avoids
the P1 clutter penalty.

## Human visual audit

The 0-500 ns blind time-power view is almost indistinguishable from FORMAL06C.
This is desirable for basal preservation but shows that sparse laminae alone do
not close the measured-domain gap.

The cropped 0-350 ns shared-scale comparison reveals a weak finite perturbation
near the designed endpoint. It does not form the long smooth crossing stack
seen in P1. The response is spatially limited and does not compete with the
basal packet.

Against equal-width measured windows, P2 remains too smooth, coherent, and
clean. The measured lines contain interrupted continuity, trace-correlated
amplitude variation, timing variation, and multiscale texture that should not
be imitated by adding more idealised physical laminae.

## Successor rule

The next formal family should separate two auditable factors:

1. fold-safe independent geology with an optional P2-like sparse-lamina factor;
2. acquisition/processing nuisance calibrated only on Line3, Line7, and LineL1,
   validated on Line6, with Line9 held out.

Do not increase physical-lamina count to match signed-event density. Do not run
additional expensive P2 controls: its role as a safe lower-bound factor is now
resolved, while the complete scene remains ineligible for formal training.
