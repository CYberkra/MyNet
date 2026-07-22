# FORMAL06E Native-64 Blind Decision

## Scope

`FORMAL06E_NONLAYERED_COVER_DEVELOPMENT` inherited FORMAL06D's source,
acquisition, basal profile, material endpoints, weathered transition and
strict full/no-basal mapping.  It changed only the non-target cover latent
covariance from an elongated field to a near-isotropic, non-periodic field.
It is development-only and blocked from training.

## Evidence

- Static input audit: passed without warnings.
- 14 m, 80 MHz attenuation plausibility budget: two-way field loss 28.62 dB;
  a paired smoke remains required.
- One-trace strict pair: passed. This establishes only causal integrity.
- Label-free native full scene: 64 consecutive traces, canonical indices
  109--172, at the native 0.09 m spacing.

Previews:

- `formal06e_native64_agc.png`
- `formal06e_native64_tpower15.png`

## Blind Morphology Verdict

**Reject before native-256.** Both displays retain a regular, full-window
parallel lobe train. The field no longer has the predecessor's elongated
covariance, so this result excludes cover-field anisotropy as the sole cause
of the artifact. No matched 64-trace control, labels, or training proposal
will be generated for FORMAL06E.

## Next Factor

Test the coupled ground/acquisition geometry rather than combining unproven
background variants: a bounded non-periodic terrain profile with a fixed
absolute aerial traverse, while preserving basal depth relative to local
ground.
