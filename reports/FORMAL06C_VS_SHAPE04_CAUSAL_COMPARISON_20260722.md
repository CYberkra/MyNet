# FORMAL06C Versus SHAPE04 Causal Comparison

## Scope

This note compares the visually accepted development baseline
`FORMAL06C_SUBTLE_INTERFACE_DEVELOPMENT` with the completed 16-trace sparse
pair `GEO14_DISTRIBUTED_GRADIENT_MEANDER_POS / SHAPE04_SPAN16_GEO14_CUDA129`.
It explains why their processed B-scans look substantially different. This is
not a promotion report: neither case is approved formal training data.

## Finding

SHAPE04 is a basal-geometry isolation experiment, not a successor that keeps
FORMAL06C's successful signal-generation contract. The large visual difference
is expected and is not caused primarily by plotting.

| Factor | FORMAL06C | SHAPE04 GEO14 | Expected visual effect |
|---|---|---|---|
| Source | 80 MHz custom zero-mean Gabor | 55 MHz analytic Ricker | Gabor produces a longer narrow-band multi-cycle packet; Ricker produces a much shorter broad-band pulse. |
| Source support at 5% peak | 53.0 ns, 16 zero crossings | 26.5 ns, 2 zero crossings | This is the dominant reason FORMAL06C has a thick black/white reflection packet while GEO14 has only a few lobes. |
| Sparse display acquisition | 32 traces, 0.72 m effective spacing | 16 traces, 1.53 m effective spacing | GEO14 is visibly blockier and laterally under-sampled for morphology reading. |
| Basal depth profile | 1.474 m range, 2 smooth extrema, slope P95 0.152 | 1.071 m range, 0 extrema, slope P95 0.065 | FORMAL06C has gentle natural long-wave variation; GEO14 is intentionally close to a single dip. |
| Transition thickness | 0.711--1.177 m, spatially variable | fixed 1.350 m | FORMAL06C introduces controlled packet variation without discrete target bodies. |
| Cover texture | correlated long/meso field: 6.7/3.0 m and 1.8/0.96 m scales, meso weight 0.18 | long/meso/local field: 9.0/3.0, 2.4/1.2, 0.75/0.45 m scales | SHAPE04 has stronger short-scale quantised texture and more block-like background after suppression. |
| Domain/protected window | 179.73 m / 500 ns | 242.73 m / 700 ns | This primarily changes runtime and late-time context; it is not the main source of wavelet appearance. |

## Evidence

- FORMAL06C raw plus common-mode/time-power preview:
  `data/simulations/v2/02_released_solver_evidence/FORMAL06C_SUBTLE_INTERFACE_DEVELOPMENT/development_evidence_v1/preview/FORMAL06C_distributed32_raw_tpower15.png`
- FORMAL06C raw plus AGC preview:
  `data/simulations/v2/02_released_solver_evidence/FORMAL06C_SUBTLE_INTERFACE_DEVELOPMENT/development_evidence_v1/preview/FORMAL06C_distributed32_raw_agc13.png`
- GEO14 re-rendered with the same display transforms:
  `reports/basal_shape_batch_v4_20260722/span16_geo14_audit/geo14_raw_tpower15.png`
  and `geo14_raw_agc13.png`.
- GEO14 strict full/no-basal spatial pair passed automatic causality and
  continuity gates, but it remains screening-only.

## Correct Successor Direction

Create an independently generated `FORMAL06D` development successor. It must
retain the *measurement mechanism* that made FORMAL06C visually useful, while
removing its development-only Line9-conditioned status:

1. retain the explicitly labelled 80 MHz Gabor proxy, 0.03 m grid, and 0.09 m
   native trace spacing;
2. retain continuous, bounded variable transition thickness and smooth
   multiscale cover correlation, but generate new seeds and do not read any
   measured line, label, waveform, or held-out statistic;
3. generate a bounded broad-relief basal profile with a small number of
   aperture-scale extrema; reject compact bowls, paired converging flanks, and
   point-like focusing responses through a blind full/control sparse review;
4. run a strict full/no-basal pair at 32 traces and 0.72 m effective spacing
   before native-256 execution; render both the Gabor successor and FORMAL06C
   with identical raw, common-mode/time-power, and AGC processing;
5. promote only after an independent provenance audit, complete native-width
   full/control outputs, and human morphology acceptance.

Do not use FORMAL06C itself as formal training data. Its manifest marks it as
Line9-conditioned development evidence and its 32-trace morphology run lacks
a matched control at the same scope.

## F02 Mechanism-Bridge Result

`IV2_F02_FORMAL06C_MECHANISM_POS` completed a 32-trace, stride-8 full/no-basal
pair on 2026-07-22. Its output contracts are complete, cover 97.25% of the
declared 22.95 m acquisition span, and the static preflight audits passed.
The pair gate passed with a target-to-background signed-difference RMS of
128.11, full-scene target-to-local-background RMS of 9.33, and zero target
dropout. It therefore proves that the long Gabor packet and a smooth basal
path can be reproduced by a newly generated source deck without a joined-
hyperbola response.

It does **not** yet visually inherit all of FORMAL06C. The F02 cover generator
introduced a 0.75 x 0.45 m local component and raised the horizontal and
vertical material-bin change rates to 0.0594 and 0.1074. FORMAL06C used only
the 6.7 x 3.0 m long and 1.8 x 0.96 m meso components (meso weight 0.18), with
rates 0.0215 and 0.0475. In the same time-power display, F02 is consequently
cleaner and more block-like, while FORMAL06C has a softer, naturally varying
multi-cycle background.

The F02 result is thus a **causal bridge pass, visual-inheritance partial
pass**. It is development-only because its mechanism selection remains
Line9-conditioned, and it must not enter formal training. The next candidate
must begin from FORMAL06C's full cover-field and transition contract, replace
only the geometry seed with a documented independent prior, and run the same
32-trace full/control audit before a native-256 run.

Evidence:

- F02 paired audit:
  `reports/formal06d_successor_20260722/iv2_f02_span32_audit/family_spatial_pilot_audit.json`
- F02 paired preview:
  `reports/formal06d_successor_20260722/iv2_f02_span32_audit/positive_pair_spatial_preview.png`
- F02 blind full-scene preview:
  `reports/formal06d_successor_20260722/iv2_f02_raw_tpower15.png`
