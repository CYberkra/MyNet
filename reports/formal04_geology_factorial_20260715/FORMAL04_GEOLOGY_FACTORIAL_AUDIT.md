# FORMAL04 geology factorial audit

## Scope

FORMAL04 kept the accepted FORMAL03 GABOR80 source, grid, acquisition,
flight height, basal path, transition thickness, stochastic latent field, and
indexed geometry fixed. Only basal contrast and correlated cover-material
amplitude changed. No measured data were read by the generator.

## Validation

- Six full/control static audits passed without warnings.
- FORMAL03 and all FORMAL04 cases share index-array SHA256
  `0ac1118fed1be74ee66d75797c527b39e32dd00d8179bae5113798ec07e17fbd`.
- The strongest material case retains 11.49 cells per conservative minimum
  wavelength at 2.8 times 80 MHz.
- A representative full/control geometry-only build passed.
- All three one-trace strict pairs passed alignment, finite-array,
  detectability, and source-reference-window gates.
- A and C each completed eight full-span full/control traces at canonical
  indices 0, 36, 72, 108, 144, 180, 216, and 252.

## Results

| Case | Texture | Basal | Full target/local background | Target CV | Dropout |
|---|---|---|---:|---:|---:|
| FORMAL03 reference | baseline | strong | 5.221 | 0.742 | 0.0% |
| FORMAL04 A | baseline | weak | 1.696 | 0.660 | 0.0% |
| FORMAL04 C | strong | weak | 1.110 | 0.448 | 0.0% |

The one-trace factor effects were: A/reference = 0.560,
B/reference = 2.102,
C/B = 0.340, and
C/A = 1.278.

## Decision

No FORMAL04 case advances directly to a dense 256-trace run. A is retained as
the causal anchor; C is retained as the upper texture bound; B is rejected
because stronger texture with strong basal contrast amplified the target.
FORMAL05 will interpolate the material mapping inside this tested bracket while
leaving source, acquisition, path, and indexed geometry unchanged. It remains
blocked from training until the same staged gates pass.
