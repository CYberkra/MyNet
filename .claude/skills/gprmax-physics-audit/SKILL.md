---
name: gprmax-physics-audit
description: Design, inspect, run, and maintain physically auditable gprMax FDTD models. Use for gprMax .in/.inc/HDF5 geometry, B-scan simulation design, source/grid/PML checks, heterogeneous media, full/no-target controls, GPU execution, provenance, or diagnosing unrealistic simulated GPR data.
---

# gprMax Physics Audit

Build simulations whose geometry, physics, controls, and provenance can survive a paper audit. Treat the installed source as the executable contract and the official manual as the conceptual contract.

## Start Here

1. Locate the exact gprMax package used by the run and record its version plus a source-tree hash.
2. Read `references/source-and-manual-contract.md` before changing input commands or material models.
3. Use `references/execution-flow.md` when debugging build, stepping, GPU, or output behavior.
4. For MyNet/PGDA work, also read `references/mynet-simulation-contract.md`.
5. Before tuning against measured imagery, read `references/measured-line-reproduction.md` and identify the signal domain.
6. Run `scripts/audit_gprmax_input.py MODEL.in --json REPORT.json` before geometry-only or GPU execution.
7. For deep conductive media, run `scripts/attenuation_budget.py` before the paired smoke.
8. Fix every error. Record justified warnings in the case manifest.

## Evidence Precedence

Use this order when sources disagree:

1. Installed source code that will execute the model.
2. Matching-version local documentation and tests.
3. Current official gprMax documentation.
4. Papers and domain references.
5. Project conventions and visual intuition.

Never assume an online command signature matches an older local installation. Record the discrepancy.

## Design Workflow

### 1. Define the Measurement Contract

Lock before geometry work:

- dimensionality and polarization;
- antenna model, center frequency, Tx/Rx offset, trace spacing, and trace count;
- solver and canonical time windows;
- acquisition span, domain margins, and coordinate convention;
- target definition and label semantics;
- formal split restrictions and forbidden conditioning sources.

Mark unknown hardware values as provisional. Do not hide them in defaults.

### 2. Pass the Physics Gate

Check all of the following:

- Grid step resolves the smallest relevant wavelength by at least the official `lambda_min / 10` rule.
- Every source, receiver, and target is at least 15 cells from the PML; include at least 15-20 air cells above the source.
- PML thickness is inside the declared domain.
- The time window captures the event and excludes unexplained late boundary returns.
- For long airborne scans, size each lateral physical guard from `2 * guard / c` and the protected target/search window; do not choose a domain from a scan-to-domain ratio alone. Validate a reduced domain with an exact cropped/shifted equivalence case.
- Geometry coordinates round to intended FDTD cells.
- Material values are frequency-appropriate and traceable.
- Dispersion, conductivity, and relaxation times are compatible with the time step.
- The target-depth two-way attenuation budget leaves a measurable response; estimate it before spending a long GPU run.

Do not use `#soil_peplinski` outside its documented 0.3-1.3 GHz validity range without a separate scientific justification. At tens of MHz, prefer measured/bounded custom materials or another frequency-valid constitutive model.

### 3. Choose Geometry Deliberately

- Use normal gprMax objects for simple analytic controls.
- Use seeded `#fractal_box`/rough surfaces only where their material-model and frequency contracts are valid.
- Use `#geometry_objects_read` for deterministic indexed heterogeneity or paired geometry. Validate HDF5 dtype, `dx_dy_dz`, dimensions, material indices, and local command signature.
- Remember that object commands are a layered canvas: later objects overwrite earlier ones.
- External voxel geometry is not dielectric-smoothed by gprMax 3.1.7. Smooth the spatial field physically before quantization and keep cells comfortably below the relevant wavelength.

Avoid abrupt vertical material partitions unless they are actual geology. They create coherent walls and crossing artifacts.

### 4. Build Strict Paired Controls

For `full` versus `no-target/no-basal`:

- share domain, grid, PML, source, receiver, stepping, time window, geometry indices, stochastic seeds, and upper-medium materials;
- change only the causal target contrast;
- hash the common geometry and both material maps;
- confirm identical trace count and sample timing;
- derive the target response from a signed pair difference before envelope or visible-phase extraction.

Prefer one shared geometry HDF5 with two audited material maps when the control changes constitutive contrast only. An air reference is useful for decomposition but is not a substitute for the strict paired control.

### 5. Validate Before Expensive Runs

Run, in order:

1. static input audit;
2. geometry-only build;
3. geometry-view or indexed-geometry preview;
4. one-trace CPU/GPU smoke;
5. short B-scan subset;
6. full paired run.

Inspect the geometry around all PML boundaries, source/receiver positions, imported-array extents, interfaces, and scan endpoints.

For a long B-scan, add an early checkpoint after roughly 10-25% of traces. Render the physically relevant late-time crop with fixed gain and compare target-band RMS against adjacent background. Stop and archive the attempt when the target is already below the declared detectability gate; do not let sunk GPU time turn a failed material budget into an accepted dataset.

### 6. Audit Outputs

Require:

- expected trace count and samples per trace;
- finite arrays and consistent time axes;
- full/control alignment before subtraction;
- a complete per-trace capture report before any merge that removes source files;
- target-support continuity, amplitude variation, and background/target energy ratio;
- comparison at identical gain, crop, distance axis, and color scale;
- provenance hashes, commands, environment, GPU, and completion state.

For a distributed sparse B-scan pilot, scale lateral continuity limits by the
actual trace stride and report both the sparse-step change and its
per-canonical-trace equivalent. A limit defined for 0.09 m traces cannot be
applied unchanged to traces sampled every 0.72 m.

For a designed true-negative scene, run and audit the target-absent full scene
without inventing a target path or requiring a no-target control. Keep solver
validity separate from human acceptance of the hard-negative semantics.

Visual similarity is supporting evidence, not a physics proof. A visually strong interface can still be leakage, a boundary artifact, or an unrealistically easy target.

Prefer a synchronous capture-and-validate step after gprMax returns and before the merge command. A resumable background watcher is useful for live progress and early-stop evidence, but it must not be the only barrier protecting per-trace provenance.

### Broadband FDTD and SFCW-Band Proxies

- A Ricker or Gaussian FDTD source is a broadband transient source, not a
  stepped-frequency (SFCW) forward simulation.
- A post-solver frequency window may be useful as a diagnostic band proxy,
  provided it is applied identically to a strict full/control pair and is
  labelled as a proxy in every artifact and report.
- Do not call zero-padding a denser measured frequency grid. Record the
  native time-derived spectral resolution separately from plotting-bin
  spacing.
- If the stated band, tone increment, and point count disagree arithmetically,
  freeze the ambiguity in the manifest and obtain the exported tone table or
  hardware-processing metadata before claiming an SFCW-equivalent dataset.
- Compare weak full-minus-control responses with both common gain (honest
  scale context) and difference-only gain (causal visibility). Neither view
  alone is sufficient for promotion.

## Maintenance

At the start of any substantial gprMax task:

1. compare the installed version and source hash with `references/version-baseline.md`;
2. check official documentation for commands being changed;
3. update the baseline date, differences, and affected guidance;
4. run the skill validator and one representative static audit;
5. keep changes factual and sourced; do not silently rewrite earlier constraints.

Generate a fresh machine-readable fingerprint with `scripts/fingerprint_gprmax.py GPRMAX_ROOT --json fingerprint.json`.

Update this skill whenever gprMax is upgraded, a source/manual discrepancy is found, a simulation failure reveals a missing guard, or the project measurement contract changes.

## Dense Cover-Bedrock Release Pattern

For a continuous cover-weathered-bedrock mechanism family, apply the following
additional gates before adding stochastic complexity:

1. Begin with a flat-ground, fixed-height, dense local window and no discrete
   anomaly body. A clean baseline must show that the deep event is caused by
   the basal contrast itself.
2. Audit the Ricker high-frequency content, not merely its nominal centre
   frequency. The project static gate evaluates wavelength resolution at
   `2.8 * fc`; preserve at least ten cells per estimated shortest wavelength.
   If this fails, reduce the grid step or change the source/grid contract
   before any GPU run. Do not waive the warning because a geometry preview
   looks smooth.
3. Use a full/no-basal pair with shared indexed geometry and acquisition.
   Capture all per-trace HDF5 contracts and SHA256 hashes immediately after the
   solve, before any merge or cleanup.
4. Keep the material-interface arrival and visible signed phase separate. A
   finite weathered transition can shift the strongest causal wavelet lobe by
   tens of nanoseconds relative to the geometric interface.
5. Never choose a visible phase independently on every trace. First restrict
   the signed full-control response to a declared geometric search window;
   then select one globally continuous path. Its transition penalty must be
   relative to the expected geometric/acquisition time increment (a
   `delta_chainage`-style transition), not an absolute preference for a flat
   time path. This is an audit candidate, not an automatic training label.
6. Record both unconstrained peak behaviour and constrained-path behaviour.
   A large reduction in side-lobe jumps is evidence of extraction quality only
   when the resulting path remains inside the signed causal response and has
   acceptable adjacent wavelet correlation.
7. Before running a correlated-cover variant, audit its quantised field on
   the actual cover voxels: used levels, horizontal/vertical neighbour-change
   rates, correlation scales, and absence of artificial full-depth walls.
   Do not use a resized preview alone to judge spatial texture.
8. A strict full/control pass proves causal attribution, not measured-line
   realism. Before scaling a family, inspect raw and identically processed
   B-scans for dominant direct/ground wavelets, repeated parallel layer bands,
   ringing combs, and diffraction/X-shaped structures. A multi-step transition
   encoded as several constant material layers can itself create coherent
   reflectors; it is not a physically smooth transition merely because its
   total thickness varies smoothly. Likewise, a quantised stochastic property
   field must not substitute for a geologically plausible multiscale texture.
   Keep such cases as regression controls if they fail the morphology review.
9. When an existing family has an accepted basal morphology, lock its basal
   path, transition, source, materials, grid, and acquisition before changing
   cover texture. Compare the predecessor and candidate at identical canonical
   trace positions, time crop, processing, and joint display scale. Do not
   horizontally interpolate a sparse comparison into apparent continuity.
   Review a label-free image first, then open material references only to check
   that the blind decision followed the intended event. Reject continuous
   stratigraphy that creates strong regular layer stacks, excessive wavelet
   repetition, or background energy that competes with the basal response,
   even when its causal and path-tracking metrics pass.

The FORMAL01 F0 baseline (2026-07-15) passed this mechanism gate with a
100 MHz Ricker proxy, 0.025 m grid, 256 traces at 0.10 m spacing, and a
continuous finite transition. The F1 correlated-cover variant subsequently
passed the same strict-pair and continuous-path checks on a 32-trace smoke
run. Visual audit subsequently found that F0--F3 remain overly layered and
wavelet-regular for the intended measured-line morphology. They are retained
as causal-control regressions, not realism candidates, and must not be scaled
to training data without a redesigned subsurface/source/processing contract.

### FORMAL02 Graded-Bedrock Successor

FORMAL02 replaces the failed F-series morphology with a deliberately smaller
claim: a non-periodic cover-to-weathered-bedrock baseline that must pass causal
and spatial gates before any realistic clutter is added.

1. Generate the basal path from seeded generic multiscale priors over the full
   solver domain, then crop the acquisition window. Reject crops that are nearly
   quadratic, lack multiple smooth extrema, exceed the slope budget, or have too
   little/too much vertical range. Do not read a measured line, label, waveform,
   or held-out statistic in the generator.
2. Use one shared indexed HDF5 geometry for full and no-basal models. In the
   control, map every transition and bedrock index back to the cover material;
   do not alter geometry or introduce a replacement interface.
3. Bound the number of transition material levels by the thinnest transition in
   cells. More named materials do not create a smoother model when several bins
   have no voxels. Record the maximum adjacent epsilon and conductivity step.
4. Keep the protected supervision window separate from the solver window. Size
   lateral guards so the earliest lateral boundary round trip occurs after the
   protected window; reserve the remaining samples for boundary diagnostics.
5. A generated source deck must satisfy the shared runner schema (`target_presence`,
   `grid.trace_count`, `grid.trace_spacing_m`, `grid.dl_m`, and
   `geometry.index_file`). Exercise staging in a test before spending GPU time.
6. Do not create a visible-phase label before a successful runtime pair. Store
   only an explicitly geometric reference, then extract a signed visible phase
   from `full - control` inside a declared search window.
7. A one-trace gate may omit air when it records `air_reference_included=false`.
   Air remains a later source/decomposition diagnostic, not a prerequisite for
   proving basal causality.
8. For a distributed sparse pilot, require full/control trace contracts, broad
   span coverage, target/background contrast, no dropout, low control spatial
   residual, correlation with the independent geometric path, and retention of
   its dynamic range. Read the analysis time limit from the output HDF5 rather
   than padding a shorter solver window to a historical constant.
9. Never draw the candidate target path over a no-basal control panel. Such an
   overlay can be mistaken for a physical control response. Report the measured
   control residual numerically and keep the control image unannotated.
10. Passing the sparse pilot does not promote the case. Run the full native trace
    count, inspect raw/common-gain/difference-only views, and obtain a human
    morphology decision before adding heterogeneity or exporting training data.
11. A completed full scene may be sufficient for an early morphology rejection,
    but never for causal attribution. If the control is stopped, record the
    exact completed control count, preserve the full capture contract, and mark
    the run `full_complete/control_partial/development_only`.
12. Do not create measured-like multi-cycle bands by inserting arbitrary thin
    layers or coarse transition bins. Separate geology from the acquisition
    response with a waveform ablation. A higher Ricker centre frequency narrows
    the pulse but does not by itself create a longer multi-cycle instrument
    response.
13. Re-run the minimum-wavelength audit whenever source frequency changes. On
    the FORMAL02 0.045 m grid, 65 MHz is approximately the highest Ricker centre
    frequency that still passes the project 2.8-times-centre-frequency,
    ten-cells-per-wavelength guard. An 80-100 MHz source requires a finer grid
    or a separately validated reduced-domain equivalence model.

### FORMAL03 Source and Path Lessons

The FORMAL03 source ablation (2026-07-15) used one exact shared correlated-cover
geometry at 0.03 m resolution to compare 65 MHz Ricker, 80 MHz Ricker, and a
finite-duration zero-mean 80 MHz Gaussian-modulated source. Keep these lessons
for later dense-interface families:

1. Store both the material-interface arrival and the source-referenced arrival.
   A custom waveform does not share gprMax's internal Ricker peak delay, so a
   geometric search window alone can be systematically mis-centred.
2. In both envelope and signed-phase dynamic programming, penalise the change
   in residual relative to the supplied reference increment. Penalising the
   absolute sample-to-sample time change creates a hidden preference for a flat
   sidelobe and can falsely reject a correctly sloping causal event.
3. Report absolute path steps for interpretation, but gate continuity on
   residual steps. For sparse full-span pilots, scale the residual-step allowance
   by the canonical trace stride and retain the unscaled per-trace equivalent.
4. A `full_scene_only` run is legitimate for an early morphology stop. Its run
   manifest must contain only the full-scene input, set
   `causal_pair_complete=false`, and block release, causal attribution, and
   visible-phase labels. Existing short full/control pairs remain the causal
   evidence for the unchanged geometry.
5. Do not judge a sparse B-scan as though interpolated columns were canonical
   dense traces. Use it to reject source character and gross morphology; a
   dense run is still required for local coherence and diffraction structure.
6. The 80 MHz Gaussian-modulated source matched the development-only Line9
   spectral peak/centroid more closely than the narrow Ricker controls and
   produced the intended multi-cycle basal packet. It did not solve geology:
   the 24-trace full-scene target/adjacent-background ratio remained about 6.43
   versus 2.35 in the diagnostic measured contract, and the synthetic mid-depth
   field remained too clean. Preserve this source for the successor ablation,
   reduce basal contrast, and add geologically plausible non-target multiscale
   texture. Do not condition a formal generator on the held-out measured line.

### FORMAL04 Constitutive-Factor Lessons

The FORMAL04 geology factorial (2026-07-15) held the FORMAL03 GABOR80 source,
grid, acquisition, basal path, transition thickness, latent stochastic field,
and indexed geometry fixed. It changed only basal contrast and the amplitude
of the correlated cover-material mapping. Preserve these lessons:

1. Treat cover-texture strength and basal contrast as interacting factors. A
   wider cover epsilon/conductivity range also changes every local
   cover-to-transition-to-bedrock sequence; it does not add only harmless
   background clutter. In FORMAL04, strong texture with the original strong
   basal contrast increased the one-trace causal difference to about 2.10
   times the FORMAL03 reference instead of making the target harder.
2. Complete a small constitutive factorial on one exact shared index array
   before changing spatial geometry. Record an array-content hash in addition
   to the HDF5 file hash so equivalent voxel states remain provable when file
   metadata differs.
3. Use one-trace pairs only to reject non-causal, misaligned, undetectable, or
   grossly over-strong material mappings. They cannot measure lateral clutter,
   coherence, dropout, or realistic target/background balance.
4. On sparse full-span pairs, report full-scene target/local-background energy
   separately from signed-pair target/background energy. The pair ratio proves
   causal localisation and is expected to be large; the full-scene ratio is
   the relevant difficulty proxy. Never substitute one for the other.
5. A material interval may be bracketed without selecting either endpoint.
   FORMAL04 A reduced the full-scene target/local-background ratio from about
   5.22 to 1.70 while retaining relatively clean cover texture; FORMAL04 C
   reduced it to about 1.11 and gave a more plausible amplitude CV, but made
   the target too weak in the full scene. Neither endpoint advanced to 256
   traces. The successor must interpolate inside the simulation-tested
   constitutive bracket and repeat static, one-trace, sparse-pair, and human
   morphology gates.
6. Measured-line statistics may be reported as development-only diagnostics,
   but formal generator parameters must be derived from independent physical
   bounds and simulation ablations. Do not read held-out arrays or labels in a
   formal generator.

### FORMAL06 Interface-Conditioning Lessons

The FORMAL06 staged ablation (2026-07-15) kept one exact geometry and source
while reducing only the cap-to-bedrock contrast. Preserve these lessons:

1. Use a local blind checkpoint before a distributed morphology run. At eight
   consecutive traces, FORMAL06A/B/C target-to-adjacent-background RMS was
   about 25.23, 7.44, and 4.01 respectively. C entered the frozen 1-5 local
   visibility interval; A and B were stopped as overexposed.
2. A one-trace pair proves causal timing only. FORMAL06C produced a 1.21 ns
   visible-phase offset, 0.001456 signed target RMS, and 3.89e-7 early leakage,
   but none of those values establishes lateral coherence or realistic
   target/background balance.
3. Do not compare the local eight-trace background ratio with a distributed
   full-span ratio as if they were the same statistic. The distributed
   32-trace FORMAL06C run tracked geometry well (correlation 0.99993) but had a
   target/background ratio of 17.29 versus 2.35 in the development-only Line9
   diagnostic. This exposed a background deficit hidden by the local crop.
4. When source peak frequency and aligned wavelet coherence already match the
   measured diagnostic, do not keep weakening the basal contrast blindly.
   FORMAL06C peak frequency was 79.37 MHz versus 79.69 MHz for Line9 and median
   aligned correlation was 0.660 versus 0.646. The next factor is continuous
   non-target geology and gentler long-wave relief, not another source change.
5. Add background complexity as spatially continuous correlated or layered
   structure. Do not use isolated inclusions merely to raise background RMS;
   they create hyperbolas and solve the metric while worsening morphology.
6. A full-only distributed run remains development evidence. It may reject or
   select a successor for a matched pair, but it cannot release visible-phase
   labels or support causal attribution at every trace.
7. Project-owner blind visual review accepted FORMAL06C on 2026-07-15 as the
   desired development morphology: a continuous, gently varying, multi-cycle
   basal packet that is visible after background suppression and restrained
   time-power gain, without a chain of isolated hyperbolas. Preserve it as the
   morphology baseline even though it is not a training release.
8. Cross-domain target/background ratios remain diagnostics, not automatic
   visual-rejection rules. A measured line, a sparse simulated crop, and a
   distributed full-only run have different clutter and gain domains. Human
   morphology acceptance can retain a development baseline while the strict
   matched-pair and independent-data gates remain closed.
9. For a weak-contrast successor, retain two geometry previews: one with a
   fixed physical scale for honest comparison and one explicitly labelled
   enhanced scale for inspecting subtle continuous texture. A fixed broad
   `full-control` colour range can make a valid weak interface look absent;
   an adaptive preview must never be presented as amplitude evidence.
10. Treat continuous stratigraphy as a factor-group hypothesis, not a solved
    realism problem. Lock source, grid, acquisition, and constitutive values;
    change the basal relief and non-target spatial organisation together;
    record neighbour-change rates and forbid point targets, isolated
    inclusions, and vertical partitions. Only a staged solver run can decide
    whether the added layers improve background balance or become an overly
    coherent horizontal comb.

### FORMAL07 Controlled-Background Lessons

The FORMAL07A/07B comparison (2026-07-15) tested two ways of increasing
non-target background complexity after FORMAL06C had been visually accepted.
Keep these rules:

1. Preserve an accepted morphology with explicit predecessor locks. Compare
   source waveform, material decks, basal profile, transition profile, grid,
   PML, and acquisition before spending solver time. Store raw file hashes for
   provenance, but use LF-normalised text-content hashes for cross-Windows/Linux
   physics locks; CRLF conversion is not a source or material change.
2. Reject background generators that improve an energy ratio by creating a
   regular horizontal comb. Measure layer-coherent depth variance and vertical
   spectral-peak concentration inside an acquisition-aligned cover crop before
   runtime. A candidate may not increase either beyond a documented tolerance.
3. Quantised-bin change fraction is not a strength metric by itself. FORMAL07B
   changed about half of cover-bin assignments, yet 99% moved by at most two
   0.025-epsilon bins. Always report perturbation RMS and bin-delta percentiles
   with the changed fraction.
4. A strict successor comparison uses identical canonical traces, time crop,
   processing, and joint display scale. Do not horizontally interpolate sparse
   traces. Review the no-overlay blind image first, then open the reference
   overlay for timing interpretation.
5. FORMAL07B showed the intended magnitude of a one-factor background change:
   the 32-trace target/adjacent-background RMS decreased from about 17.29 to
   16.74 while seven signed lobes and a 79.37 MHz peak were preserved. Treat a
   small, controlled result as success when the purpose is to protect an
   accepted basal packet; do not demand a dramatic visual change from a weak
   ablation.
6. Development morphology is not formal-data eligibility. FORMAL07B is still
   Line9-conditioned and blocked from training. Independent scene generators
   must use physical priors that do not read held-out geometry, timing, labels,
   or morphology.

### Independent V2 Pilot Lessons

The first independent Family 01 pilot (2026-07-15) was deliberately generated
without measured arrays, Line9 timing, or FORMAL06/07 geometry. Preserve these
rules:

1. Independence is a generator input contract, not a filename. Record an empty
   measured-file input list, forbid development arrays, and keep positive and
   target-absent cases in one indivisible scene-family split group.
2. The cleanest designed true negative is the exact physical state of the
   positive no-basal control. Share the HDF5 index array, make material decks
   byte-identical, retain target-independent acquisition arrays, and create no
   target path or visible-phase label for the negative.
3. Stage runtime evidence: static and transient-VTI geometry checks, one-trace
   positive pair plus negative equality, distributed full-span causal pair, then
   a denser full-only morphology stop before the native run. Delete transient VTI
   after hashing it.
4. Never horizontally interpolate sparse traces. Resize vertically as needed,
   then use horizontal nearest-neighbour display and disclose the effective trace
   spacing. Family 01's 32 traces were 0.72 m apart, not native 0.09 m data.
5. A full-only morphology path is never a training label. Family 01's 32-trace
   path tracked independent geometry at 0.9981 correlation with no dropout, but
   causal labels still require the complete matched full-minus-control pair.
6. A physics pass is not a difficulty pass. Family 01 had four signed lobes and a
   target/adjacent-background RMS ratio of 7.22, so it was accepted only as
   `pilot_passed_pending_full256`. Scale multiple harder independent families
   with stronger continuous non-target background; do not let the easy pilot
   define the training domain.

### Mechanism Transfer Without False Independence

The Family 01/Family 02 comparison (2026-07-15) separates geometry provenance
from mechanism provenance. Apply these rules when an accepted development case
was selected with held-out diagnostics:

1. Do not throw away a physically successful mechanism merely because its
   selection was held-out-conditioned. Transfer it onto independently generated
   geometry as an explicitly development-only, single-factor experiment.
2. Record conditioning scope precisely. `No measured arrays read` does not make
   a case formal when source or material choices were selected using held-out
   morphology or metrics. Mark that case `line9_conditioned=true` at the
   mechanism-selection scope.
3. Lock geometry, acquisition, grid, PML, seeds, and strict controls while
   changing source/material mechanism. This makes the predecessor comparison
   interpretable and prevents a geometry change from masquerading as a source
   improvement.
4. Family 02 showed why this distinction matters. On Family 01's regenerated
   geometry, the FORMAL06C mechanism changed four significant lobes at 44.09 MHz
   into seven alternating lobes at 79.37 MHz, while retaining a continuous
   non-hyperbolic path. The accepted FORMAL06C morphology was therefore a
   transferable physical mechanism, not an accidental copy of its geometry.
5. Keep the positive and its exact target-absent control indivisible. A negative
   does not become formal merely because it contains no target; its family-level
   parameter provenance still governs eligibility.
6. To create a formal successor, re-origin source and constitutive ranges from
   independent physical measurements, literature bounds, or a predeclared
   non-held-out factorial. Re-run static, causal-pair, sparse blind, dense, and
   human gates. Never relabel the development transfer itself as independent.

### Instrument-Band Source Proxies

When approximating a stepped-frequency or wideband instrument without measured
complex system phase, read
`references/instrument-band-source.md`. Treat a magnitude-derived zero-phase
pulse as an explicit proxy, freeze geometry/materials during source ablations,
and require blind raw, time-power, and AGC review. Hardware-band consistency is
necessary but does not guarantee the preferred solved wavelet morphology.

### Measured Realism Versus Strict Holdout

Do not confuse a split-integrity rule with the simulator's visual objective.
For MyNet, maintain two explicit contracts:

1. A measured-realism calibration track may use Line9 morphology, spectrum,
   target prominence, and continuous-background character to select a
   development simulator. Mark every resulting family
   `line9_conditioned=true` and never describe Line9 as unseen for that track.
2. A strict-holdout track must select generator parameters without its held-out
   line, or recalibrate the simulator independently inside each leave-one-line-
   out fold. Formal eligibility is a claim boundary, not a reason to keep an
   unrealistic simulator.

Project-owner visual review is binding for lineage selection. On 2026-07-16,
the corrected ranking was `FORMAL06C > Independent Family 02 > Independent
Family 03`. Family 02 retained FORMAL06C's source and near-identical material
values but replaced its basal profile and cover field; the new field had about
2.75 times the horizontal and 2.26 times the vertical neighbour-bin change
rates and the path had seven smoothed extrema instead of two. Family 03 then
added a sharper 100 MHz amplitude-only zero-phase band proxy and shifted the
solved centroid to about 116.2 MHz. Keep both as ablations. Do not extend the
Family 03 frequency sweep, and build the next realism candidate directly from
FORMAL06C with source, materials, grid, acquisition, basal packet, and
transition locked before changing continuous non-target geology.

FORMAL08A records the corresponding pre-solver pattern. Add non-target
complexity with a smooth depth envelope: protect the near-surface and
basal-neighbour bins exactly, apply aperiodic multiscale 2D texture only in the
middle cover, and forbid isolated bodies, point targets, vertical partitions,
and sinusoidal slabs. Gate a full-domain candidate with absolute cross-domain
limits (predecessor correlation, perturbation RMS, spectral concentration, bin
delta, and protected-bin equality). Do not require a relative increase over a
mini-test domain because that ratio is domain-size sensitive. For FORMAL08A,
the full-domain pre-solver values were correlation 0.9204, perturbation RMS
0.3396, changed-bin fraction 0.1524, bin-delta P99 3, and exact protected bins.
These values permit an eight-trace checkpoint; they do not establish solved
realism. Always review the short full-scene checkpoint before distributed or
dense runs.

The solved FORMAL08A result adds an important stop rule. Eight consecutive
traces preserved the packet, and the exact common 32-trace run retained a
0.99989 path correlation, seven signed lobes, 79.37 MHz peak frequency, and no
dropout. The full-span target/adjacent-background RMS decreased from about
17.29 to 14.77, but blind review found only a weak increase in middle-time
background and no material visual improvement over FORMAL06C. A numerically
effective one-factor ablation is not automatically a new realism baseline.
Retain FORMAL06C as the mother model, archive FORMAL08A as a background
ablation, and skip matched-control/native runs when the full-span blind gate
does not produce a meaningful visual gain.

FORMAL08B adds a second stop rule. Increasing a transition-following continuous
deep-cover field is not equivalent to adding harmless background clutter. The
candidate preserved FORMAL06C's basal geometry, seven signed lobes, 79.37 MHz
peak, and zero dropout, but its full-span target/adjacent-background RMS rose
from about 17.29 to 21.33 and the blind background did not improve. A cover
field close enough to the protected transition can reshape illumination and
the interface packet even when the material deck and basal geometry are locked.
Stop such a factor before matched controls or native 256. The next non-target
factor should be spatially separated from the basal corridor, use a predeclared
amplitude budget, and vary broad orientation or local continuity without
isolated bodies. Do not treat a stronger texture multiplier as a new geology
mechanism.

### Empirical Nuisance and Sparse-Event Lessons

The FORMAL09A-09C sequence (2026-07-16) tested target-excluded measured-domain
statistics around the released FORMAL06C solver output. Preserve these rules:

1. Fit paper-fold nuisance statistics with equal line weighting. Remove a
   stable common-mode trace and exclude the target corridor before estimating
   spectra or event distributions. Keep a separate all-lines development fit
   and never call it strict holdout.
2. A matched temporal spectrum is useful only as a diffuse nuisance component.
   Separable temporal/lateral covariance and a true joint 2D power spectrum
   both failed to reproduce finite local event topology; second-order Gaussian
   power is not a substitute for sparse coherent geology or acquisition events.
3. Do not copy measured residual patches, waveform snippets, or event
   coordinates. Save only fold-safe distributions and sample new locations,
   phases, supports, slopes, and amplitudes. Exclude the synthetic target
   corridor during both fitting and generation.
4. Short connected components provide unstable second derivatives. Do not clip
   noisy measured curvature estimates into a fixed bound, because this piles
   samples onto the limits and creates artificial hyperbolas. Use a declared
   conservative near-zero geometry prior until native-resolution evidence can
   support curvature.
5. Scale a sparse coherent field by a robust peak statistic before mixing it
   with diffuse noise. Global standard-deviation normalisation can make a few
   short events unrealistically bright.
6. Never resize a sparse B-scan with horizontal bilinear interpolation. Resize
   vertically as needed, then use horizontal nearest-neighbour display and
   disclose the effective trace spacing. FORMAL09C's apparent visual gain
   largely disappeared after this correction.
7. A 32-trace stride-8 checkpoint cannot prove finite-event continuity when a
   typical event occupies only a few observed traces. Use a native-spacing
   consecutive checkpoint before promoting the topology. For a physical
   successor, preserve the accepted basal mechanism and add only low-contrast,
   finite, gently dipping mid-cover laminae/lenses; forbid point targets,
   vertical partitions, and periodic slabs.
8. FORMAL09C light/balanced/rich were all rejected for promotion. They remain
   deterministic mechanism evidence only, with no training eligibility or
   gprMax causal claim. The next gate is a native-spacing 64-trace physical
   ablation before any strict pair or 256-trace run.
9. Do not map a detected signed-event count one-to-one onto a physical lamina
   count. A single thin lamina has two boundaries and its response is convolved
   with a multi-lobe source wavelet, so it can create roughly three to five
   connected signed events. Record the assumed lobe multiplicity and treat the
   resulting physical density as a conservative prior, not measured geology.
10. Compare a physical clutter ablation against its exact predecessor at the
    same canonical native traces before attributing any visible wave group to
    the new geometry. FORMAL09C-P1 preserved the basal path and seven-lobe
    79.37 MHz packet, but its dense laminae reduced target/adjacent-background
    RMS from 10.40 to 5.61 and introduced over-coherent crossing mid-cover
    wave groups. It is a mechanism/stress regression, not a realism baseline.
11. A fair measured visual check must also match physical aperture. Do not
    compare a 5.7 m synthetic subset only against a whole 150-220 m measured
    line compressed to the same panel width. Use equal trace count and similar
    native spacing, disclose independent display scaling, and keep held-out
    lines diagnostic-only.
12. For a sparse physical successor, taper both contrast and thickness to zero
    at finite endpoints, require at least one endpoint inside the native pilot,
    add bounded correlated centreline roughness, impose a non-crossing minimum
    separation, and keep acquisition nuisance as a separate auditable factor.
13. FORMAL09C-P2 demonstrates the conservative lower bound: changing only
    0.0157% of cover cells with a P99 cover-bin delta of three preserved 99.2%
    of the FORMAL06C basal target/background ratio (10.32 versus 10.40), kept
    zero dropout, and produced a weak finite endpoint without P1-style crossing
    stacks. Retain this topology as an optional physical factor, not a complete
    measured-realism baseline.
14. Weak finite factors can be hidden in a 0-500 ns panel. After the full-window
    blind gate, inspect a disclosed cropped blind view at the exact same traces
    and shared scale as the predecessor. A difference-only response is not
    sufficient: the factor must be faintly readable in the full-scene crop and
    must not damage the target packet.
15. If a sparse physical factor preserves the target but the equal-aperture
    synthetic panel remains much smoother than measured data, do not add more
    coherent layers. Split the successor into independent geology and a
    separately auditable acquisition/processing nuisance factor. Calibrate
    nuisance only on fit lines, validate on the validation line, and keep the
    held-out line diagnostic-only.

### MyNet VTI Lifecycle

VTI is a geometry visualization export, not an FDTD solver input or a training
artifact. Apply this project contract:

1. Generate VTI only for a new domain/PML layout, source or receiver placement,
   geometry topology, or voxel index array. It is an optional first-build
   inspection, not a prerequisite for every run.
2. If the locked index-array SHA256, domain, PML, and acquisition coordinates
   are unchanged, material/source/trace-count ablations may skip VTI entirely.
3. Keep `#geometry_view` out of production full/control/air decks. Put it only
   in explicit `geometry_check_*` inputs.
4. Use the project runner's transient lifecycle: generate, record filename,
   byte size, and SHA256 in `run_logs/geometry_view_cleanup.json`, then delete.
5. Never commit, LFS-track, release-package, or copy VTI between workstations.
   Retain one locally only while resolving a documented geometry dispute. Use
   `scripts/cleanup_gprmax_geometry_views.py --report REPORT.json --delete` to
   hash and clear legacy views under the project data tree.

## References

- `references/source-and-manual-contract.md`: official rules and installed-source behavior.
- `references/execution-flow.md`: reviewed 3.1.7 build, stepping, solve, and output call chain.
- `references/mynet-simulation-contract.md`: project-specific dataset and paired-control rules.
- `references/measured-line-reproduction.md`: raw/processed/migrated domain separation and staged measured-line calibration.
- `references/version-baseline.md`: reviewed version, source fingerprints, and maintenance log.
- `scripts/audit_gprmax_input.py`: reusable static audit utility.
- `scripts/attenuation_budget.py`: exact nondispersive field-attenuation plausibility budget.
- `scripts/capture_gprmax_trace_contract.py`: preserve per-trace positions, attributes, shapes, and hashes before merge removal.
- `scripts/fingerprint_gprmax.py`: version/source/manual fingerprint for maintenance.
