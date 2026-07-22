# FORMAL06D Distributed Pair Decision

Date: 2026-07-22

FORMAL06D is a development-only, Line9-conditioned mechanism-transfer case.
It preserves the FORMAL06C measurement, cover, transition, material, grid, and
acquisition contract while regenerating only the geometry and cover-field
seeds. It is not eligible for formal training or an independent Line9 claim.

## Valid strict pair

The valid run is `FORMAL06D_PAIR32_DISTRIBUTED_STRICT_CUDA129`, sampled at
canonical indices `0, 8, ..., 248` over 22.32 m of a 22.95 m acquisition.

- Full/control source and receiver maximum position error: `3.7e-06 m`.
- Signed pair target/background RMS: `100.81`.
- Full-scene local target/background RMS: `10.27`.
- Target dropout fraction: `0.0`.
- Visible-minus-reference median: `-1.60 ns`.
- Visible residual P95: `7.04 ns`.

The signed full-minus-control response therefore supports causal attribution of
the smooth, multi-cycle basal packet. The unlabelled 32-trace time-power view
is a morphology screen only; its 0.72 m effective spacing must not be treated
as native B-scan resolution.

## Audit interpretation

The first audit used a legacy hard upper limit of `5.0` for the full-scene
local target/background ratio. It correctly preserved `formal_promotion=false`
but labelled the automatic gate failed because FORMAL06D measures `10.27`.
That is a difficulty/realism review condition, not a failed solver or an
unaligned pair. The updated FORMAL06D source contract records the same value as
`full_scene_target_to_local_background_rms_review_above=5.0`: it requires blind
review, while lower-bound detectability and causal attribution remain separate.

The earlier `FORMAL06D_BLIND32_DISTRIBUTED_FULL_CUDA129` control run is invalid
for subtraction because the resumed control did not inherit the distributed
stride. It is retained only as a runner-regression audit record. The runner is
fixed and regression-tested before the valid strict pair above was executed.

## Native-spacing checkpoint and decision

The 64-trace native-spacing `FORMAL06D_NATIVE64_FEATURE_FULL_CUDA129` run
covered canonical traces 109--172 around the maximum low-curvature feature.
It passed static input validation but fails the blind morphology gate. Both the
restrained time-power and AGC views show a highly regular, parallel multi-cycle
wave train through much of the protected window. The basal event is continuous
and non-focusing, but its local native-resolution texture is too constructed
and layered for the intended measured-domain role.

Decision: stop FORMAL06D before a native-256 pair. Retain its strict 32-trace
pair as causal mechanism evidence and its native-64 full scene as a morphology
regression. Do not export labels or training windows. A successor must preserve
the useful broad basal packet while changing the non-target/background mechanism
in a separately auditable factor; it must pass a native-spacing blind checkpoint
before another matched native run is authorised.
