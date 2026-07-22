# FORMAL06G Native-64 Blind Decision

## Scope

`FORMAL06G_TERRAIN_ACQUISITION_DEVELOPMENT` inherited FORMAL06F's one-cap
transition, FORMAL06D's cover and basal mechanisms, source, materials, grid,
and absolute Tx/Rx traverse. Its only changed factor was a bounded,
non-periodic ground profile. The source retained a fixed absolute elevation,
so the AGL range is explicit (7.15--8.85 m) and basal depth remains relative
to local ground.

## Evidence

- Static input audit: passed without warnings.
- One-trace strict full/control pair: passed.
- Label-free native full scene: 64 consecutive traces, canonical indices
  109--172, at the native 0.09 m spacing.

Previews:

- `formal06g_native64_agc.png`
- `formal06g_native64_tpower15.png`

## Blind Morphology Verdict

**Reject before native-256.** Terrain produces modest, physically expected
warping of the background events, but the regular full-window multi-cycle
lobe train remains. The result excludes perfectly flat terrain/fixed AGL as
the sole cause of the comb. No matched 64-trace control, labels, or training
proposal will be generated for FORMAL06G.

## Next Factor

The next investigation should inspect the ideal point-source/receiver
abstraction and installed gprMax antenna options before more material or
terrain sweeps. This avoids treating an acquisition-system artifact as a
subsurface-geology parameter problem.
