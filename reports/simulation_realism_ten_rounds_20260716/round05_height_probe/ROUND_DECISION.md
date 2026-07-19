# Round 05 Decision: Physical Flight-Height Probe

## Question

Can acquisition-height variability be represented by a post-solve trace delay,
or must it be solved in the physical model?

## Evidence

Three one-trace `full_scene` / `no_basal_contrast_control` FDTD pairs were
solved from the independent `IV2_F01` source deck.  Only source and receiver
height changed; the HDF5 geometry, materials, lateral offset and source
waveform remained locked.  The air-only reference was intentionally omitted:
it cannot replace the same-geometry no-basal causal control.

| Grid-realised height | Residual peak time | Difference from 8.01 m | Air-path expectation |
|---:|---:|---:|---:|
| 7.50 m | 422.86 ns | -3.11 ns | -3.40 ns |
| 8.01 m | 425.97 ns | 0.00 ns | 0.00 ns |
| 8.49 m | 429.02 ns | +3.04 ns | +3.20 ns |

The direct-wave peak remains at 26.61 ns because the Tx-Rx lateral separation
is fixed and both are raised together.  The signed basal residual changes by
the expected air-path scale.  The requested 8.50 m position lands on 8.49 m
on the 3 cm grid; both values are recorded in the JSON evidence.

## Decision

- Accept physical source/receiver height as an acquisition factor.
- Reject post-solve per-trace delay as a height surrogate.
- Do not promote this probe as a training case: it has one trace per height
  and exists only to establish the physical contract.
- Use the measured fifth-column height only after its semantics and quality
  gate are satisfied; otherwise disable the measured arrival prior.
