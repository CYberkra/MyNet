# UAV-GPR Hardware Measurement And 3D Preflight

This protocol starts the next realism stage without inventing a finite antenna
from a two-dimensional Hertzian line source. Its input is a measured hardware
contract, not a visually attractive B-scan.

## What Must Be Captured

Record the system model and antenna model/serial, then measure or obtain from
manufacturer documentation:

- antenna element type and physical length, width, and thickness;
- Tx/Rx centre separation, polarization, boresight, and aircraft mounting;
- usable band and nominal centre frequency;
- trace spacing, flight-height reference, and height measurement method.

Capture a representative raw direct-wave or free-space air-reference trace
with the same Tx/Rx assembly, channel, sampling clock, and acquisition mode as
the survey. Preserve the raw file, its SHA256, sample interval, time-zero
definition, channel/component, dewow/filtering recipe, normalization recipe,
and a preprocessing manifest. A processed screenshot is not source evidence.

## Evidence Boundaries

The Line9 report page is a migrated elevation-domain interpretation. It may be
used only as development morphology evidence. It must not choose antenna size,
source phase, pulse width, geometry, or a formal release threshold.

Until `hardware_measurement_contract_v1.json` validates as
`ready_for_3d_local_preflight`, all finite-antenna statements and 3D source
decks are blocked. A Gabor, Ricker, or zero-phase proxy remains only a source
family, not a substitute for an antenna measurement.

## First 3D Experiment

The first valid study is a bounded local-window mechanism check, not a 256-trace
production dataset:

1. Use at most 16 native traces and one fixed, flat, layered basal geometry.
2. Build exact `full_scene`, `no_basal_contrast_control`, and `air_reference`
   decks with the same finite-antenna geometry and acquisition coordinates.
3. Run static, geometry-only, one-trace, then local 16-trace checks. Record
   source/receiver positions, geometry hashes, output hashes, early equality,
   and signed target response before rendering a B-scan.
4. Compare raw, time-power, and AGC views under the frozen display contract.
   The aim is to determine whether the finite measurement model reduces the
   rejected full-window comb without destroying causal basal support.

Do not scale a bundled high-frequency gprMax antenna by intuition. If a finite
antenna cannot yet be represented from measured dimensions and feed details,
create a clearly named reduced-order proxy and state exactly what it cannot
claim.
