# FORMAL02 Full-Scene Morphology Audit

Date: 2026-07-15

## Execution State

- Full scene: 256/256 traces complete, capture contract complete, merged output retained.
- No-basal control: stopped after 26 traces by user-approved early decision.
- Air reference: not run.
- Solver output: 6126 native samples, 0.1061 ns step, 650.10 ns actual end.
- Protected interpretation window: 0-500 ns.

This run is `development_only_full_complete_control_partial`. It cannot support
full-resolution causal attribution, a visible-phase training label, or formal
training promotion.

## Verdict

The full output is numerically valid, but the measured-like morphology gate
fails. FORMAL02 should remain a clean basal-mechanism baseline. It should not be
scaled into a training family.

## Full-Only Metrics

| Metric | FORMAL02 | Development-only Line9 reference |
|---|---:|---:|
| Target / adjacent-background RMS | 13.003 | 2.349 |
| Target envelope CV | 0.204 | 0.465 |
| Dropout below 25% median | 0.0% | 1.85% |
| Aligned template correlation median | 0.296 | 0.646 |
| Aligned peak frequency | 35.3 MHz | 79.7 MHz |
| Spectral centroid | 100.8 MHz | 84.6 MHz |
| Significant signed lobes | 7 | not used as a formal target |

The numerical comparison is diagnostic only. Measured Line9 remains test-only,
and its statistics must not be read by a strict-holdout simulation generator.

The full-only continuous phase follows the generic geometry with correlation
0.861, but retains only 44.2% of the geometric time range. Its maximum adjacent
step is 0.64 ns. This means the event is continuous but too flat relative to the
designed interface.

## Visual Findings

1. Raw amplitude remains direct-wave dominated, as expected for an airborne
   transient model.
2. Time-power gain reveals one broad, very clean, laterally continuous basal
   wave group. It is more plausible than the repeated hyperbola-like F-series
   response, but substantially easier than the measured target.
3. AGC exposes several crossing and parallel weak branches below the main
   event. Without the completed control these cannot be classified as basal
   side lobes, multipath, or residual boundary/processing structure.
4. The black-white bands are still an idealised transient wavelet. Matching the
   measured multi-cycle appearance by inserting arbitrary thin layers would
   confound source response with geology and is rejected.

## Source Decision

Do not replace 55 MHz with 100 MHz on the current 0.045 m grid. Under the
project's 2.8-times-centre-frequency resolution audit, 55 MHz has 12.24 cells
per minimum wavelength, 65 MHz has about 10.36, 80 MHz has about 8.4, and
100 MHz has about 6.7. The latter two require finer grids.

The next source pilot should compare:

1. 65 MHz Ricker on the current grid;
2. 80 MHz Ricker on an approximately 0.035 m grid;
3. a zero-mean, finite-duration 70-85 MHz Gaussian-modulated transient from an
   explicitly audited excitation file on the finer grid.

Each source is a transient proxy, not an SFCW forward model.

## FORMAL03 Plan

Build one successor family without measured-line conditioning:

- retain flat ground and the generic non-periodic basal path for the first
  comparison;
- add a continuous correlated cover field, avoiding full-depth material walls
  and coarse quantisation steps;
- vary transition thickness smoothly and independently of basal depth;
- preserve one shared HDF5 geometry for full/no-basal controls;
- run one-trace and 24-32-trace full-span pairs only;
- test the three source choices above as an independent ablation axis;
- reject target/background ratios above 8, envelope CV below 0.30, unexplained
  periodic branches, or visible-path dynamic retention below 0.60;
- do not run another 256-trace pair until one pilot passes both causal and
  morphology review.

The formal generator must use broad hardware/geological priors. Line9 may be
shown in a development-only comparison report, but its label, waveform, and
statistics remain forbidden generator inputs.
