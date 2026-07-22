# FORMAL06F Native-64 Blind Decision

## Scope

`FORMAL06F_SINGLE_CAP_TRANSITION_DEVELOPMENT` inherited FORMAL06D exactly
except for one physical mechanism: its eight-stage weathered transition was
replaced by a single variable-thickness full-contrast weathered cap. It is a
development-only causal diagnosis, not a training candidate.

## Evidence

- Static input audit: passed without warnings.
- One-trace strict pair: passed.
- Label-free native full scene: 64 consecutive traces, canonical indices
  109--172, at the native 0.09 m spacing.

Previews:

- `formal06f_native64_agc.png`
- `formal06f_native64_tpower15.png`

## Blind Morphology Verdict

**Reject before native-256.** The regular parallel lobe train remains under
both display transforms. The result excludes the eight-stage transition
staircase as the sole cause of the native-spacing comb. No matched 64-trace
control, labels, or training proposal will be generated for FORMAL06F.

## Next Factor

The next isolated diagnostic is acquisition/ground geometry: retain the basal
depth relative to the local surface, add bounded non-periodic terrain, and
keep the aerial source on a fixed absolute traverse to yield an explicitly
audited AGL range.
