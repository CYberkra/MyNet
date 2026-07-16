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
| 10A | Bounded dispersive material family | Yes | Natural attenuation/wavelet change under strict pair |
| 10B | Independent basal/cover morphology families | Yes | Multi-line range coverage without copied geometry |
| 11A | Sparse 3D correction patches | Yes, limited | Quantified 2D/3D residual benefit |

Every stage must provide raw, identically processed, target-crop, and blind
multi-line panels before the next factor is added.

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
