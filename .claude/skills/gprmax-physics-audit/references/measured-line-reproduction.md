# Measured-Line Reproduction Workflow

Use this workflow when a measured line, report figure, or interpreted profile is
used to diagnose a simulation family. It is an audit workflow, not permission
to train on a held-out line.

## 1. Quarantine The Reference

Record:

- line ID, split, source file, hash, label version, and orientation;
- whether the reference is raw, processed, migrated, interpreted, or a composite;
- every use restriction.

If the reference is held out, any geometry, parameter, threshold, or model
tuned from it is development-only. A later formal family must be regenerated
from independent priors and promoted without held-out conditioning.

## 2. Name The Signal Domain

Never compare unnamed images. Distinguish:

1. raw receiver time-domain field;
2. processed time-domain B-scan;
3. migrated distance/depth image;
4. elevation-domain profile;
5. interpreted geology overlay.

A migrated elevation figure is not a raw FDTD target. First validate the raw
full/control pair, then apply a frozen matched processing and migration chain.

## 3. Freeze The Display Contract

Before visual judgement, record and freeze:

- sample/time grid and crop;
- trace spacing, span, and distance-axis definition;
- acquisition versus profile orientation;
- dewow/background subtraction;
- gain function and parameters;
- band filtering or spectral window;
- interpolation/resampling;
- normalization percentile and color limits.

For a strict full/control pair, show one shared physical amplitude scale.
Across uncalibrated real and simulated datasets, use the same robust scaling
algorithm independently for morphology only and report each scale.

## 4. Build A Machine-Readable Reference Contract

Report at least:

### Acquisition

- sample and trace counts;
- time interval and window;
- trace spacing and span;
- Tx/Rx geometry;
- flight-height and terrain statistics;
- profile-display flip.

### Path Geometry

- raw visible-phase time;
- air-corrected subsurface time;
- declared reference velocity/permittivity;
- depth and elevation proxies;
- range, extrema, slope, and curvature statistics.

Depth/elevation conversions are hypotheses unless velocity is independently
measured. Do not use them as hidden truth.

### Signal

- target/adjacent-background RMS;
- target envelope coefficient of variation;
- dropout fraction under a declared threshold;
- target-aligned signed-wavelet correlation distribution;
- spectral peak and centroid after the declared processing;
- strong, weak, ignore, and negative fractions.

### Background

- early/direct-wave energy;
- coherent upper-layer energy;
- late-window energy;
- endpoint and boundary energy;
- dominant clutter slopes and spatial scales.

## 5. Diagnose The Gap Causally

Use full minus no-target before changing geometry. Classify each mismatch as:

- arrival timing;
- attenuation/detectability;
- interface spatial spectrum;
- transition-zone thickness;
- target-strength variation;
- upper clutter;
- source wavelet;
- terrain/acquisition;
- boundary/endpoint artifact;
- processing/migration mismatch.

Do not repair a timing error with display gain, an attenuation error with
deeper geometry, or a migrated-image mismatch by hand-copying an interpreted
curve.

## 6. Use The Calibration Ladder

Change one factor group at a time.

1. S0 source/grid/PML and coordinate rounding.
2. S1 flat-ground target depth and travel time.
3. S2 material and attenuation budget.
4. S3 transition thickness, target strength, and dropout.
5. S4 upper clutter and finite lenses.
6. S5 optional secondary/deep anomalies.
7. S6 independent terrain and flight-height variation.
8. S7 matched processing, migration, and elevation conversion.

Flat ground is preferred at S1-S5 when terrain is not the scientific variable.
It prevents acquisition geometry from hiding errors in subsurface physics.

## 7. Run Staged Solves

For every revision:

1. static audit;
2. geometry-only output;
3. one full/control trace;
4. 16-32 trace full-span sparse pair;
5. full pair;
6. optional air reference after the pair passes.

Stop when the declared target window, shared-scale detectability, or boundary
gate fails. Archive the rejected revision and change one factor group.

## 8. Acceptance Matrix

A usable development model requires:

- exact full/control alignment and shared geometry;
- causal signed target response;
- target visible in the full scene, not only a difference-only gain;
- timing inside the declared search window;
- no single oversized bowl/arch unless geologically intended;
- realistic target continuity, amplitude CV, dropout, and phase changes;
- nonblank but non-target-dominating background;
- no PML, endpoint, or vertical-wall explanation;
- identical processing/orientation for comparison;
- complete provenance and human visual decision.

Passing similarity does not make a held-out-conditioned case formal data.

## 8.1 Multi-Line Paired Nuisance Transfer

When a causal FDTD interface is visually accepted but its background remains
too clean, separate the physical scene from the acquisition/system domain.
Do not keep increasing transition-following geology merely to reproduce every
field-data nuisance.

Use this staged pattern:

1. Estimate target-excluded noise spectra from several credible measured
   lines. Prefer fixed-position noise recordings when available; otherwise
   label the field-window estimate as a pseudo-noise model.
2. Sample new complex spectral coefficients. Never paste measured traces into
   synthetic cases.
3. Model lateral cross-spectral covariance and a smooth non-stationary amplitude
   envelope; white Gaussian noise and one stationary colored field are only
   baselines.
4. Apply the identical gain, timing, and additive-noise realization to a
   strict full/control pair. The transformed signed difference must equal the
   declared shared operator applied to the original signed difference.
5. Keep canonical solver outputs immutable and store transformed arrays as a
   separately versioned derived domain.
6. Fit development parameters on a multi-line pool. Refit inside each
   leave-one-line-out fold for a strict generalization claim.
7. Review raw and identically processed blind panels. Reject a scalar metric
   improvement when the result looks like uniform colored noise, regular
   ripples, combing, or artificial dropout.

FORMAL09A demonstrated the stop rule: reducing target/background RMS from
17.29 to 2.50 with a stationary colored field brought scalar values closer to
measured data but produced visually uniform wavelet-like clutter. The 4.50
balanced variant preserved the packet and is a mechanism starting point, not a
release. The next ablation should replace the hand-shaped spectrum with a
multi-line empirical spectrum before adding lateral covariance, metadata
conditioning, or stronger amplitudes.

See Stephan, Allroggen, and Tronicke (2024),
https://doi.org/10.1002/nsg.12273, for a convolution-based GPR noise model using
measured noise spectra. Extend that idea only with explicit provenance,
multi-line folds, and pair-preserving transformations.

## 9. MyNet / Line9 Special Rule

Line9 is strict test-only. Line9-conditioned cases may be used to discover
which physical factors are missing, but they are permanently development-only.

Separate array independence from decision independence. A generator that reads
no measured arrays may still be Line9-conditioned when its source, material,
or transition mechanism was chosen from Line9 visual or numerical diagnostics.
Such a case is useful as a development-only mechanism-transfer test. It becomes
formal only after those parameter ranges are independently re-originated and
all gates are repeated; changing the filename or geometry is not sufficient.

For the current Line9 program:

- the PDF page is migrated elevation-domain evidence;
- V15 is the raw-time visible-phase reference;
- first calibrate a horizontal surface and fixed 8 m height;
- keep basal depth near the borehole/profile range and calibrate effective
  permittivity rather than deepening the interface to match time;
- add independent terrain and 8-12.6 m flight-height variation only after the
  flat pair passes;
- generate formal families from non-Line9 priors and rerun every gate.

## 10. Smooth Target-Strength Calibration

When an interface is too dominant, do not weaken it by assigning abrupt
full-depth lateral material classes. Those class boundaries become artificial
vertical reflectors and can resemble stitched hyperbolae.

Instead:

- define target-strength variation as a broad deterministic lateral field;
- apply it most strongly in the interface transition zone;
- recover smoothly toward the unmodified deep material with depth;
- preserve the same geometry indices and labels in the strict control;
- inspect both the processed full scene and the signed pair difference.

Target/background RMS and envelope CV are screening metrics, not sufficient
visual acceptance. A dense weak background and a weak target can produce a
similar ratio while having different morphology. Use orthogonal variants to
separate those causes before combining them.

A 16-32 trace full-span pilot is suitable for rejecting target dominance,
background sparsity, or gross coherence errors. It is not sufficient to
measure fine lateral dropout or phase continuity at canonical trace spacing.
Require a canonical-spacing confirmation before promoting a morphology.

Report the quantization of every sparse fraction. A one-trace event in an
N-trace pilot has a minimum nonzero fraction of `1/N`; do not compare that
percentage directly with a dense measured line. Preserve trace IDs and verify
that a designed weak response occurs inside the declared calibration zone and
recovers on neighbouring traces.

## 11. Canonical Local-Coherence Audit

Calibration fields are often stored on the FDTD material grid, while B-scan
metrics are indexed by acquisition trace. Never use `argmax(grid_field)` as a
trace ID. Interpolate the grid field onto the saved trace-midpoint coordinates
first, then identify and report the calibration trace and its neighbouring
recovery traces.

For a canonical-spacing confirmation, retain all of the following together:

- full/control pair audit with expected trace count;
- per-trace HDF5 contracts for both members;
- processed local full, control, and signed-difference view;
- target-envelope ratio profile with a declared dropout threshold;
- raw material-field and trace-coordinate provenance.

An acceptable weak zone is bounded, smooth, and causally present in the
signed pair difference. Reject a zone that is explained by an endpoint, PML,
material wall, missing trace, or a change to the interface centre label.

## 12. Dense-Window Production Rule

Do not use a long, sparsely sampled FDTD B-scan as evidence that an interface
matches a densely acquired field line. Record the physical trace spacing and
compare it with the in-medium wavelength and the measured trace increment.
Interpolation cannot recover lateral phase or continuity that was never
sampled.

For production synthetic windows, prefer independent 20-30 m local domains
with dense acquisition (typically 0.08-0.12 m trace spacing) over a 200 m
domain with only a few hundred traces. Establish lateral PML guard size with a
convergence test. Keep flat terrain while calibrating material, transition,
and source physics; introduce terrain and height variation as a separate
factor only after the flat full/control pair passes.

Ricker versus Gaussian is a source-calibration choice, not a realism label.
Use a simple pulse for mechanism studies, then fit the effective measured
spectrum and early response before making amplitude/phase fidelity claims. A
true SFCW synthetic must preserve its frequency-domain acquisition and apply
the same frequency-to-time conversion as the instrument.

## 13. Cover-Bedrock Interface Rule

Model a basal reflector as a continuous cover-weathered-transition-bedrock
system. Use correlated cover properties and a smooth basal centre surface.
Do not manufacture the default basal reflection from discrete high-contrast
scatterers; those belong to a separate anomaly family and naturally create
diffractions/hyperbolae.

Store at least two label concepts: the material/geometric interface arrival and
the visible-phase centre extracted from the causal full/control response.
Amplitude and phase of an overburden-bedrock reflection depend on the contrast,
loss, source, antenna, height, and processing, so they are not interchangeable.

For a pulse-level study, a Ricker source is an accepted broadband proxy even
when the physical system is SFCW. Name it honestly as a pulse proxy. For
instrument-faithful SFCW claims, preserve the stepped complex frequency samples
and apply the instrument's frequency-to-time reconstruction.

## 14. Dense-Window Causal Phase Extraction

The FORMAL01 F0 dense-window baseline established a reusable distinction:
the material-interface arrival and the most energetic signed full-control
wavelet lobe may differ materially when the interface contains a finite
weathered transition. Treat that offset as an extraction problem to audit, not
as permission to overwrite a geometric label.

For a continuous designed interface:

1. retain the geometric arrival as a search-window reference;
2. calculate the signed full-minus-control response;
3. record independent local maxima to quantify possible side-lobe switching;
4. derive one globally continuous candidate path inside the declared search
   window;
5. penalise only deviations from the expected trace-to-trace geometric time
   increment, not the absolute slope of the path;
6. report path step statistics, adjacent signed-wavelet correlations, dropout,
   and geometric-to-visible offsets;
7. require human review before exporting any visible-phase label.

Do not let dynamic programming erase a genuine discontinuity. This procedure
is appropriate for a model whose geometry contract explicitly states that the
basal interface is continuous. In a designed fault, pinch-out, or true
no-target segment, use breakable/null-state handling instead.
