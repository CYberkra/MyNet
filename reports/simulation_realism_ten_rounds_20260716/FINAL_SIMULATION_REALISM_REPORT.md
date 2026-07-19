# Ten-Round Simulator Realism Research: Final Synthesis

## Scope and Claim Boundary

This report closes the required ten-round research cycle.  It does **not**
claim that a formal training simulator is ready.  It identifies a physically
auditable successor architecture, records negative mechanisms that must not be
reintroduced, and preserves the distinction between a Line9-informed visual
development track and a strict formal-fold track.

The solver source is gprMax 3.1.7.  Every physical attribution discussed here
uses a same-geometry `full_scene` / `no_basal_contrast_control` pair where the
pair exists.  Sparse subset runs are morphology evidence, never a released
native-256 training export.

## Research Ledger

| Round | Factor | Evidence result | Decision |
|---:|---|---|---|
| 01 | Bounded zero-phase response | Spectrum RMSE decreased about 33% on fit and validation; section remained too clean | Optional measurement component only |
| 02 | Trace gain and post-solve delay | Gain created regular bands; delay damaged the causal target | Reject delay; do not use as a height surrogate |
| 03 | Target-excluded low-rank residual | Rank-6 energy was 0.996 versus measured about 0.63; visual bands dominated | Reject |
| 04 | Finite post-solve events | One native-spacing event still read as an inserted packet | Reject |
| 05 | Physical Tx/Rx height | Causal residual moved -3.11 / +3.04 ns around 8.01 m, near air-path theory | Accept physical height only |
| 06 | Continuous roughness/coherence | Full-width stratigraphy looked constructed; sparse P2 factor preserved target | Continuous stack rejected; P2 optional lower bound |
| 07 | Volumetric cover texture | Weak factor gave little visual gain; stronger factor altered target conditioning | Reject as a single-factor successor |
| 08 | Finite facies/lamina topology | Dense P1 competed with basal response; sparse P2 was safe but insufficient | Optional weak physical factor only |
| 09 | Source/instrument proxy | Ricker is an independent baseline; zero-mean source remains development-only; amplitude proxy is not SFCW | Retain explicit source ablations |
| 10 | Minimum combination | Physics and measurement factors must remain separable and causal | Select two-track architecture; no promotion |

The machine-readable evidence index is `research_ledger.json`; existing solved
run references for Rounds 06-09 are verified in
`round06_09_physical_evidence.json`.

## Newly Solved Physical Evidence: Round 05

`IV2_F01` was copied to disposable solver cache and solved as three
one-trace, full/control height pairs.  Source and receiver moved together;
their 0.18 m lateral separation, HDF5 geometry, materials and source waveform
remained unchanged.

| Requested height | Grid-realised height | Full-control residual peak | Delta from 8.01 m | Air-path expectation |
|---:|---:|---:|---:|---:|
| 7.50 m | 7.50 m | 422.86 ns | -3.11 ns | -3.40 ns |
| 8.01 m | 8.01 m | 425.97 ns | 0.00 ns | 0.00 ns |
| 8.50 m | 8.49 m | 429.02 ns | +3.04 ns | +3.20 ns |

The common direct-pulse peak stayed at 26.61 ns because common elevation does
not change the Tx-Rx spacing.  This validates the model contract: use a
physical source/receiver coordinate update for height; do not apply a
post-solve trace warp.  The 8.50 m request lands at 8.49 m on the 3 cm grid
and is recorded that way in the evidence.

## Selected Architecture

### Formal Fold-Safe Physics Layer

1. Generate independent or fold-only basal profiles, cover property fields and
   finite facies parameters.  Do not read Line9 labels, time distributions,
   terrain arrays, source response, or visual morphology during generation.
2. Solve a same-geometry full/no-basal pair for every positive family.  The
   control restores the local cover/transition material rather than deleting
   arbitrary background.
3. Use verified per-trace height by moving source and receiver in the input
   deck.  If fifth-column height semantics or quality fail, disable the
   measured arrival prior rather than filling missing values with a constant.
4. Permit at most a weak, tapered P2-like finite lamina factor away from the
   basal corridor.  Do not use periodic full-width layering, point-target
   chains, vertical walls, or dense crossing laminae.
5. Preserve geometric reference and visible-phase labels as different arrays.
   Derive visible phase only after a signed pair review; never let an
   appearance augmentation move either label.

### Fold-Safe Measurement Layer

1. Fit a bounded zero-phase response on Line3, Line7 and LineL1 with equal
   line weighting; validate on Line6 before any diagnostic opening of Line9.
2. Allow only amplitude/spectral response whose provenance is stored with the
   case.  Do not transfer measured phase, copy waveform snippets or residual
   patches, inject low-rank stripes, or add post-solve delay fields.
3. Keep source models as explicit ablations.  A 55 MHz Ricker baseline, a
   fold-derived zero-mean packet, and any instrument-band proxy are separate
   cases.  An amplitude-only proxy cannot be labelled SFCW.

### Development-Only Visual Track

`FORMAL06C` and `IV2_F02` remain useful morphology references because they
retain a continuous multi-cycle basal packet that visually resembles part of
the project data.  They are permanently `line9_conditioned` and must be
described as development evidence only.  They cannot contribute to strict
Line9 training, selection, threshold calibration, or reported generalisation.

## Required Release Gates

No new case becomes formal training data until all are true:

```text
independent_non_line9_scene_family = true
full_no_basal_pair_complete = true
visible_phase_from_signed_pair = true
measurement_fit_excludes_line9 = true
true_measured_negative_windows > 0
V15_split_and_label_contract = pass
native_spacing_blind_visual_gate = pass
human_audit_no_constructed_stacks_or_hyperbola_chain = pass
```

## Storage and Reproducibility Rules

- Version source decks, manifests, control material maps, label arrays,
  reports, previews and hashes in Git/LFS as appropriate.
- Keep raw `.out` solver outputs in the solver cache or an explicitly managed
  data release, never as an accidental side effect in a source-deck directory.
- VTI is a transient geometry inspection product: hash it, record it, then
  delete it.  It is neither a solver input nor a training artifact.
- Record grid-realised source/receiver coordinates, not just user-requested
  coordinates.  This matters for non-grid-multiple heights.

## Next Executable Work

1. Implement one **new independent** formal candidate using the selected
   physics layer, with a fold-only measurement response and a small native
   spacing full/control pilot.
2. Generate and audit the matching true-negative family separately.
3. Complete the V15 data gates: true negatives, line-level split lock, and
   measured height semantics.
4. Only then export the 501 x 256 training NPZs and start the new path-first
   network smoke train.  The simulator and network studies remain separately
   auditable experiments.

## Primary References

- gprMax input and stepping documentation: https://docs.gprmax.com/en/latest/input.html
- gprMax Python scripting documentation: https://docs.gprmax.com/en/latest/python_scripting.html
- gprMax implementation paper: Warren et al., *Computer Physics
  Communications* (2016), https://doi.org/10.1016/j.cpc.2016.08.020
- Realistic GPR nuisance simulation example: Stephan et al., *Near Surface
  Geophysics* (2024), https://doi.org/10.1002/nsg.12273
