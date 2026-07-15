# FORMAL03 source ablation and full-24 morphology audit

Date: 2026-07-15

Status: development only; formal training and promotion remain blocked.

## Scope

FORMAL03 compares three sources on one exact shared, generic correlated-cover
geometry: 65 MHz Ricker, 80 MHz Ricker, and a zero-mean finite-duration 80 MHz
Gaussian-modulated waveform. No measured line, label, waveform, or held-out
statistic was read by the generator.

All three variants completed one-trace and distributed eight-trace full/control
pairs. GABOR80 then completed a 24-trace full-span full-scene morphology run at
stride 11. The 24-trace run intentionally omitted the control because the
unchanged geometry already had short paired causal evidence and this gate was
only intended to reject or retain morphology.

## Correctness fix

The previous visible-phase dynamic program penalised absolute time steps. This
created a hidden preference for a flat sidelobe whenever the reference interface
sloped. Both the envelope and signed-phase stages now penalise residual-step
change relative to the supplied reference increment.

Relevant tests: 47 focused simulation, runner, morphology, and FORMAL03 tests
passed after the fix.

## Corrected distributed-eight source comparison

| Source | Reference span | Visible span | Retention | Pair target/background | Full local target/background | Amplitude CV |
|---|---:|---:|---:|---:|---:|---:|
| GABOR80 | 34.68 ns | 26.00 ns | 0.750 | 67.78 | 5.22 | 0.72 |
| RICKER65 | 34.68 ns | 32.50 ns | 0.937 | 22.33 | 3.14 | 0.47 |
| RICKER80 | 34.68 ns | 33.80 ns | 0.975 | 23.47 | 3.62 | 0.55 |

The earlier low retentions of 0.19, 0.45, and 0.34 were extractor artefacts,
not evidence that the basal event was absent.

## GABOR80 full-24 morphology result

- Runtime: 8 minutes 24 seconds on RTX 5070.
- Solver resources: about 891 MB host RAM and 2.21 GB GPU memory.
- Captured traces: 24/24; canonical indices 0 through 253 at stride 11.
- Causal pair complete: false by design.
- Output and morphology contract: pass.
- Path/reference correlation: 0.99997.
- Dynamic-range retention: 1.012.
- Target/adjacent-background RMS: 6.43.
- Target envelope CV: 0.595.
- Dropout below 25% of median: 0%.
- Aligned peak frequency: 88.18 MHz.
- Aligned spectral centroid: 82.70 MHz.

## Development-only Line9 comparison

The measured contract is diagnostic only and must not condition a strict
holdout generator.

| Metric | FORMAL03 GABOR80 | Line9 contract | Interpretation |
|---|---:|---:|---|
| Target/adjacent background RMS | 6.43 | 2.35 | Synthetic target remains about 2.7x too easy |
| Target envelope CV | 0.595 | 0.465 | Synthetic lateral amplitude variation is sufficient but differently distributed |
| Dropout | 0.0% | 1.85% | Synthetic event is too uniformly present |
| Template correlation median | 0.531 | 0.646 | Similar order, but sparse sampling limits interpretation |
| Peak frequency | 88.18 MHz | 79.69 MHz | Source band is close |
| Spectral centroid | 82.70 MHz | 84.64 MHz | Source band is close |

Visual comparison confirms that GABOR80 produces the desired multi-cycle basal
packet and avoids the earlier isolated-hyperbola appearance. It remains too
clean through the mid-depth region, too coherent, and too strong compared with
the measured line.

## Decision

Select GABOR80 as the source for the next geology ablation. Do not run the
current FORMAL03 geometry at 256 traces and do not promote any source variant.

The successor must keep the source and acquisition fixed while independently
testing:

1. weaker basal permittivity/conductivity contrast;
2. stronger geologically correlated non-target cover texture;
3. limited local coherence loss or attenuation variation without discrete
   anomaly bodies or full-depth walls.

Run one-trace full/control, then an eight-trace distributed pair. Only the best
geology variant should receive another 24-trace full-scene morphology run.
