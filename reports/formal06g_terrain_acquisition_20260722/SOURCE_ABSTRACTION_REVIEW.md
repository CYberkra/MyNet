# Source-Abstraction Review After FORMAL06E/F/G

## Installed-Source Facts

The executable local gprMax source tree is
`F:\codex\PSGN-CSNet\gprMax-master`.

- `gprMax/input_cmds_multiuse.py` reports that a Hertzian dipole is a **line
  source in 2D**.
- The same parser rejects `#transmission_line` when GPU solving is requested.
- Bundled antenna examples are finite 3D models such as `GSSI_400` and
  `GSSI_1500`; they are not 80 MHz UAV antenna models.

## Why This Matters

FORMAL06E (cover covariance), FORMAL06F (transition staircase), and FORMAL06G
(bounded terrain/fixed absolute aerial traverse) all passed static and
one-trace causal checks but failed the same label-free native 64-trace
morphology gate. The unresolved shared component is the 2D ideal source and
receiver abstraction, which can preserve unrealistically coherent radiation,
ground interaction and multiples.

## Next Valid Experiment

Do not start another wide 2D material sweep. First define an antenna
measurement contract from hardware records: antenna dimensions, polarization,
feed/receiver configuration, altitude range and an 80 MHz-compatible source
spectrum. Then construct a reduced 3D local-window equivalence experiment
with an explicit compute budget. Compare its raw and identically processed
late-time response to the 2D proxy before using it to redesign any 2D dataset
generator.

A user-defined waveform is allowed as a separate temporal-support ablation,
but it cannot be represented as a finite-antenna/directivity replacement.
