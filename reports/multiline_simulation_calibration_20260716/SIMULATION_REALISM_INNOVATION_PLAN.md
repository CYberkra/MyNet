# Simulation Realism Innovation Plan

Date: 2026-07-16

## Recommended paper contribution

Develop a **multi-line, metadata-conditioned, causal-pair-preserving nuisance
transfer** around an auditable gprMax physics family.

The central idea is to model three domains explicitly:

```text
physical scene -> gprMax full/control -> empirical acquisition operator -> network
```

The operator is fitted from target-excluded measured statistics, samples new
realisations, and is shared exactly by the full/control pair. It therefore
improves measured-domain realism without allowing a generative model to move or
hallucinate the basal label.

## Why this is preferable here

- Five credible measured lines provide enough traces to estimate spectra,
  cross-trace covariance, height-conditioned amplitude envelopes, and robust
  feature distributions, but are too few for an unconstrained image generator.
- FORMAL06C already has the desired basal wavelet morphology; rebuilding the
  entire scene with a GAN would discard the strongest verified asset.
- FFT/covariance sampling is cheap and reproducible on the existing machines.
- The method supports a clean paper ablation and a leave-one-line-out claim.
- It is more project-specific than generic domain adversarial training while
  remaining interpretable and auditable.

## Staged experiment matrix

| Stage | Changed factor | Solver needed | Visual exit condition |
|---|---|---:|---|
| 09A | Hand-shaped colored nuisance | No | Mechanism only; completed |
| 09B-1 | Empirical per-line noise spectrum | No | Better frequency texture without target loss |
| 09B-2 | Cross-spectral lateral covariance and non-stationarity | No | Irregular measured-like background, no uniform ripple |
| 09B-3 | Height/slope-conditioned gain and <=2 ns smooth timing jitter | No | Higher envelope CV without packet fracture |
| 09C | Sparse finite coherent-event field from target-excluded event statistics | No | Local slopes/curvature and finite support without copied patches |
| 10A | Bounded dispersive material family | Yes | Natural attenuation/wavelet change under strict pair |
| 10B | Independent basal/cover morphology families | Yes | Multi-line range coverage without copied geometry |
| 11A | Sparse 3D correction patches | Yes, limited | Quantified 2D/3D residual benefit |

Every stage must provide raw, identically processed, target-crop, and blind
multi-line panels before the next factor is added.

## FORMAL09B outcome

FORMAL09B-1 passed as a temporal-spectrum component, but not as a complete
realism candidate. Its paper-fold fit remained stable without Line9.

FORMAL09B-2 separable covariance and FORMAL09B-2R1 joint 2D Gaussian sampling
were both visually rejected. They preserved marginal or joint second-order
power but continued to produce long regular ripples instead of finite local
events. Metadata conditioning is therefore deferred. FORMAL09C must model a
sparse coherent-event topology explicitly before gain or timing factors are
introduced.

## FORMAL09C outcome

FORMAL09C detected target-excluded finite events on the paper-fold fit lines
and generated new sparse events without copying measured coordinates or
patches. The mechanism worked technically, but no candidate passed the visual
promotion gate. Once the 32 sparse traces were displayed with horizontal
nearest-neighbour resizing, the apparent gain over FORMAL09B-1 became small
and short events were under-resolved. `light` was nearly unchanged,
`balanced` contained an isolated block-like event, and `rich` remained too
regular and too clean relative to the multi-line references.

The next stage is therefore not a larger post-solver event weight. Build
`FORMAL09C_P1_DENSE_PHYSICAL_FINITE_LAMINAE` directly from FORMAL06C with the
accepted source and basal mechanism locked. Introduce only finite,
low-contrast, gently dipping mid-cover laminae/lenses, then inspect a
native-spacing consecutive 64-trace full-scene checkpoint. Run a strict pair
and native 256 traces only after that blind morphology gate passes.

## Formal evaluation

Maintain two parameter sets:

- `development_all_lines`: broad measured realism; Line9-conditioned;
- `paper_line9_holdout`: fit Line3/Line7/L1, validate Line6, freeze, then test
  Line9.

Report simulator distribution coverage, not a single closest-image score:

- target/background RMS;
- envelope CV and dropout;
- aligned template correlation;
- peak frequency and spectral centroid;
- 2D power-spectrum distance;
- lateral coherence-length distribution;
- self-supervised feature MMD or sliced-Wasserstein distance.

## Deferred options

- A small conditional residual diffusion model may be tested only after the
  empirical operator is established. It must generate nuisance residuals,
  never full labelled radargrams.
- Full-image CycleGAN/CUT translation is not a primary path because it can move
  interfaces and silently invalidate labels.
- Direct injection of simulated targets into measured backgrounds requires
  confirmed target-free measured segments, which the current V15 dataset does
  not provide.
