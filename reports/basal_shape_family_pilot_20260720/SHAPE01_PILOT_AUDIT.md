# SHAPE01 basal geometry one-trace audit

All four cases completed a causal full/no-basal pair. This is a smoke audit, not a 256-trace release.

The signed difference is the only causal target evidence used here; separate full-scene energy is not treated as basal proof.

- **BS01_FLAT_REFERENCE**: signed peak 433.26 ns, geometry reference 407.13 ns, offset +26.13 ns, abs peak 2.251e-03, signed/full RMS 0.930.
- **BS02_BROAD_RISE**: signed peak 437.58 ns, geometry reference 426.24 ns, offset +11.33 ns, abs peak 1.813e-03, signed/full RMS 0.924.
- **BS03_DOUBLE_RELIEF**: signed peak 434.75 ns, geometry reference 407.47 ns, offset +27.27 ns, abs peak 1.579e-03, signed/full RMS 0.893.
- **BS04_GENTLE_MULTISCALE**: signed peak 423.35 ns, geometry reference 391.77 ns, offset +31.58 ns, abs peak 2.189e-03, signed/full RMS 0.982.

The 11-32 ns geometry-to-visible offset is expected to require a signed-wavelet/visible-phase extraction stage; it is not permission to overwrite the geometric label or to apply a post-solve time shift.

Decision for the next gate: keep all four as independent shape candidates, but promote none yet. Run a 32-trace canonical-spacing pair for BS02 and BS04 first; BS01 remains the flat control and BS03 is a multi-extrema stress case.

Formal promotion remains blocked until dense/canonical spacing, visible-phase extraction, and human semantic review pass.
