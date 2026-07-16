# FORMAL09C-P1 Visual Audit Decision

## Decision

`FORMAL09C_P1_DENSE_PHYSICAL_FINITE_LAMINAE` is rejected as a measured-realism
baseline and remains blocked from training. Retain it as a deterministic
physical-mechanism regression and strong coherent-clutter stress case.

## Evidence

- Static full/control audits: pass with zero errors and zero warnings.
- Geometry-only build: pass; transient VTI files were hashed and deleted.
- One-trace strict full/no-basal pair: pass.
- Native-spacing full scene: 64 consecutive traces, complete and finite.
- Exact predecessor comparison: the same 64 canonical traces from FORMAL06C,
  with identical source, materials, grid, acquisition, basal path, transition,
  processing, and display scales.
- Equal-width measured comparison: 64-trace windows selected from the longest
  strong-positive non-ignore runs in Line3, Line6, Line7, Line9, and LineL1.
  Line9 is diagnostic only and did not condition the generator.

## Numerical result

| Metric | FORMAL06C | FORMAL09C-P1 |
|---|---:|---:|
| Basal path/geometric correlation | 0.8486 | 0.8453 |
| Target/adjacent-background RMS | 10.4029 | 5.6106 |
| Target dropout fraction | 0.0 | 0.0 |
| Significant signed lobes | 7 | 7 |
| Peak frequency | 79.37 MHz | 79.37 MHz |
| Median aligned-template correlation | 0.7956 | 0.8059 |

P1 preserved the basal packet and did not create dropout, but its added
mid-cover energy roughly halved the basal target-to-background ratio.

## Human visual audit

At equal common-trace scales, P1 adds strong, smooth, crossing wave groups in
the middle-time region. The added response is substantially more coherent and
clean than the local measured windows, and it competes with the basal packet.
The native-width measured comparison shows that gently dipping continuity is
not itself invalid; the failure is the combination of excessive physical
lamina density, long idealised coherence, and insufficient multiscale weak
texture.

The original density mapping treated each detected signed event as one
physical lamina. That is not valid: one finite thin lamina can produce several
signed lobes through its two boundaries and the source wavelet. The P1 physical
lamina count therefore overstates the measured event prior by roughly a factor
of three to five.

## Successor rule

FORMAL09C-P2 must branch from FORMAL06C, not P1, and change only the lamina
factor:

- deconvolve signed-event density into a conservative physical-lamina prior;
- use shorter, weaker, tapered laminae;
- force at least one finite endpoint into the native 64-trace aperture;
- impose non-crossing centerline separation;
- add low-amplitude correlated roughness;
- keep acquisition nuisance as a separate auditable factor.

Do not run a 64-trace matched no-basal pair for P1. Its full-scene morphology
already fails the realism gate, and the existing one-trace pair is sufficient
to preserve the basal causal regression.
