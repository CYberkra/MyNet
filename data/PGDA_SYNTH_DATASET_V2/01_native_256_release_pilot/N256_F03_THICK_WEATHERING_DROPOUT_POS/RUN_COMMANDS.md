# N256_F03_THICK_WEATHERING_DROPOUT_POS

## Geometry-only checks
```bash
python -m gprMax geometry_check_full.in --geometry-only
python -m gprMax geometry_check_control.in --geometry-only
```

## B-scan runs (256 traces; source/receiver step = 0.09 m)
```bash
python -m gprMax full_scene.in -n 256 --geometry-fixed
python -m tools.outputfiles_merge full_scene --remove-files
python -m gprMax no_basal_contrast_control.in -n 256 --geometry-fixed
python -m tools.outputfiles_merge no_basal_contrast_control --remove-files
python -m gprMax air_reference.in -n 256 --geometry-fixed
python -m tools.outputfiles_merge air_reference --remove-files
```

The merged HDF5 outputs must then be passed to scripts/postprocess_physical_sim_v2.py.
