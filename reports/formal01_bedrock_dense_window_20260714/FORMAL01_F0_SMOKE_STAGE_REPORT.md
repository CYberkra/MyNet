# FORMAL01 F0 Dense-Window Smoke Stage

Date: 2026-07-14

## Scope

`FORMAL01_BEDROCK_DENSE_WINDOW_F0_BASELINE` is an independent 2-D,
flat-ground cover-weathered-bedrock mechanism case. It is not derived from
Line9 traces, labels, geometry, terrain statistics, or amplitude statistics.
It remains development-only.

## Numerical Contract

| Item | Value |
| --- | ---: |
| Grid | 0.025 m |
| Domain | 40.2 m x 50.0 m |
| Active cells | 1608 x 2000 (2-D) |
| Source proxy | 100 MHz Ricker |
| Height | 8.0 m fixed |
| Tx/Rx offset | 0.20 m |
| Dense acquisition | 256 traces at 0.10 m, 25.5 m span |
| Physical guards beyond PML | 6.0 m left and right |
| Minimum wavelength resolution | 10.99 cells at 2.8 times centre frequency |

The earlier 0.04 m grid was rejected by static audit because it only resolved
the estimated shortest significant wavelength with 6.87 cells. The current
grid meets the ten-cell release gate. A gprMax geometry-only build confirmed a
maximum estimated numerical phase-velocity error of about 0.69%.

## Mechanism Contract

- Full and no-basal control use the same `geology_indices.h5`, source,
  receiver path, PML, scan trajectory, cover field, and topsoil.
- The control replaces only weathered-transition and bedrock constitutive
  properties with the cover property.
- The model has a continuous basal interface and a finite 0.55–1.75 m
  weathered transition; it has no discrete anomaly bodies.
- `geometric_reference_arrival_time_ns.npy` is a material-interface estimate,
  not a visible-phase training label.

## Eight-Trace GPU Smoke

Full and control outputs completed on the local RTX 5070. Pair audit result:

| Check | Result |
| --- | ---: |
| Paired traces | 8 / 8 |
| Time increment | 0.058966 ns |
| Early-window full-control RMS, maximum | 1.655e-05 |
| Target-window full-control RMS, minimum | 5.535e-03 |
| Strict pair causality | pass |

The strongest signed full-control difference falls 19.2–23.6 ns after the
geometric material-interface reference. This is treated as an expected
visible-phase offset produced by the finite transition, not as an error or an
automatic training label.

Evidence:

- `formal01_f0_smoke_pair_audit.json`
- `formal01_f0_smoke.png`
- `formal01_f0_smoke_metrics.json`

## Full 256-Trace Runtime Pair

The matching full and no-basal runs completed with 256 consecutive traces
each. Per-trace runtime contracts, coordinates, array shapes, and SHA256 hashes
are saved in the case directory as `runtime_full_256_trace_contract.json` and
`runtime_control_256_trace_contract.json`.

| Check | Result |
| --- | ---: |
| Full/control paired traces | 256 / 256 |
| Early-window full-control RMS, maximum | 1.705e-05 |
| Target-window full-control RMS, minimum | 4.882e-03 |
| Target/adjacent difference RMS | 7.62 |
| Difference-amplitude dropout | 0.00% |
| Strict pair causality | pass |

Naively taking the largest difference peak on every trace caused side-lobe
switches, even though the underlying response is continuous. The audit now
uses a geometry-window-constrained dynamic path whose transition penalty is
relative to the expected geometric trace-to-trace time increment. This is a
descriptive audit path only, not a generated training label.

| Continuity measure | Per-trace local maximum | Constrained audit path |
| --- | ---: | ---: |
| Median adjacent wavelet correlation | 0.986 | 0.970 |
| P10 adjacent wavelet correlation | -0.720 | 0.746 |
| Adjacent correlations below 0.5 | 16.08% | 1.96% |
| P95 adjacent path step | n/a | 0.843 ns |
| Median geometry-to-visible offset | 16.31 ns | 15.34 ns |

The constrained path follows one continuous deep wavelet lobe in the raw,
display-processed, and signed full-minus-control panels. The finite transition
therefore produces a visible-phase shift that must be extracted after runtime;
the geometric material boundary remains unsuitable as an automatic label.

## F1 Correlated-Cover 32-Trace Smoke

F1 keeps the same acquisition, source, indexed geometry contract, transition
and bedrock as F0, but replaces the uniform cover with an independently seeded,
horizontally correlated quantised cover field. This is the first stochastic
complexity check after F0, not a formal data release.

| Check | Result |
| --- | ---: |
| Full/control paired traces | 32 / 32 |
| Early-window full-control RMS, maximum | 4.307e-05 |
| Target-window full-control RMS, minimum | 5.442e-03 |
| Median adjacent constrained-path correlation | 0.965 |
| Causal-path dropout | 0.00% |
| Strict pair causality | pass |

The display preview is less dominated by isolated diffraction-like fragments
than the earlier discrete-body prototypes. The causal deep event remains a
continuous lobe beneath the upper response. This is a mechanism result only:
F1 has not been run at 256 traces, has no reviewed visible-phase labels, and
remains `development_only=true` with `formal_training_allowed=false`.

Evidence:

- `formal01_f1_smoke_pair_audit.json`
- `formal01_f1_smoke_continuity_audit.json`
- `formal01_f1_smoke.png`

## Release Status

F0 passes the *mechanism-baseline* gate, and F1 passes the correlated-cover
smoke gate: the deep response is causally tied to
the cover-weathered-bedrock contrast and is continuous under a defined
audit-time path constraint. Both remain **development-only and non-trainable**:
source/guard convergence, a reviewed visible-phase extractor, and human
acceptance still remain. F2–F3 have passed static input audit only. They are
deliberately not queued until the quality of the correlated-cover random field
is reviewed against these controlled references.
