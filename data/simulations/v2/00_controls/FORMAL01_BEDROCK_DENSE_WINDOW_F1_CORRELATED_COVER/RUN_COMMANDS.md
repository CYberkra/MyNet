# FORMAL01_BEDROCK_DENSE_WINDOW_F1_CORRELATED_COVER

This case is not trainable until the release gates in `scene_manifest.json` pass.

```powershell
python -m gprMax geometry_check_full.in --geometry-only
python -m gprMax geometry_check_control.in --geometry-only
python -m gprMax full_scene.in -n 8 --geometry-fixed -gpu 0
python -m gprMax no_basal_contrast_control.in -n 8 --geometry-fixed -gpu 0
python -m gprMax full_scene.in -n 256 --geometry-fixed -gpu 0
python -m gprMax no_basal_contrast_control.in -n 256 --geometry-fixed -gpu 0
```

Run `air_reference.in` only when validating or changing the source/antenna proxy.
