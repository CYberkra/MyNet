# Native 256 Correlated-Voxel Batch V1

This directory contains blocked, pre-promotion gprMax source decks. It inherits
MACRO03's correlated-voxel geology representation while retaining the canonical
501 x 256 acquisition at 0.09 m trace spacing.

## Cases

- `N256_CV01_BALANCED_MULTISCALE_POS`: representative moderate positive.
- `N256_CV02_LOW_CONTRAST_RIDGED_POS`: low-contrast, thicker-transition positive.
- `N256_CV03_PATCHY_TRANSITION_POS`: locally weakened, lens-rich positive.
- `N256_CV04_UPPER_CLUTTER_TRUE_NEG`: target-absent correlated hard negative.

Positive full/control inputs share one case-local `geology_indices.h5`. Only
transition and bedrock material mappings change. The true negative does not
contain transition or bedrock region indices and therefore has no control run.

All cases remain `formal_training_allowed=false` until complete solver outputs,
signed-pair visible-phase extraction, independent audit, and explicit human
promotion are complete. Geometric reference arrays are audit priors only.

Generate the decks with:

```powershell
F:\codex\envs\psgn-csnet\python.exe scripts\generate_native_256_correlated_voxel_batch.py --overwrite
```

Run the representative distributed preflight with:

```powershell
F:\codex\envs\psgn-csnet\python.exe scripts\run_native_256_release_pilot.py `
  data\simulations\v2\01_native_256_correlated_voxel_batch_v1\N256_CV01_BALANCED_MULTISCALE_POS `
  --trace-count 32 --trace-stride 8 --execute
```
