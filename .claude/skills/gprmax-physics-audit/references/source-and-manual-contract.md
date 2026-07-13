# gprMax Source And Manual Contract

Reviewed against local gprMax 3.1.7 (`Big Smoke`) and the official documentation on 2026-07-12.

## Primary Documentation

- User guide: https://docs.gprmax.com/en/latest/
- Input commands: https://docs.gprmax.com/en/latest/input.html
- GPR modelling guidance: https://docs.gprmax.com/en/latest/gprmodelling.html
- Advanced heterogeneous-soil example: https://docs.gprmax.com/en/latest/examples_advanced.html
- GPU execution: https://docs.gprmax.com/en/latest/gpu.html
- Output and geometry views: https://docs.gprmax.com/en/latest/output.html
- Utilities: https://docs.gprmax.com/en/latest/utils.html
- Python scripting: https://docs.gprmax.com/en/latest/python_scripting.html

## Installed Source Map

Locate the active package rather than assuming a path. For the reviewed local tree, the important files are:

- `gprMax/_version.py`: executable version.
- `gprMax/input_cmds_geometry.py`: geometry command parsing and voxel import.
- `gprMax/fractals.py`: seeded FFT-based fractal surfaces and volumes.
- `gprMax/fractals_generate_ext.pyx`: compiled fractal kernels.
- `gprMax/input_cmds_multiuse.py`: material and source command parsing.
- `gprMax/materials.py`: constitutive and mixing-model calculations.
- `gprMax/pml.py` and `gprMax/pml_updates/`: absorbing-boundary construction and updates.
- `gprMax/model_build_run.py`: build/run lifecycle.
- `docs/source/input.rst`, `gprmodelling.rst`, `gpu.rst`, and `output.rst`: matching local manual.
- `user_models/heterogeneous_soil.in`: official seeded heterogeneous-soil pattern.

## Verified Rules

### Grid And PML

- Official rule of thumb: `dl <= lambda_min / 10` in the highest relevant permittivity and frequency content.
- For the reviewed Ricker models, gprMax's own dispersion report placed the significant upper frequency near `2.76 * fc`; use `2.8 * fc` as the default static-audit estimate unless the solver reports a different value.
- Local `grid.py` estimates the upper significant waveform frequency at 40 dB below peak power, floors the displayed cells-per-wavelength value, treats fewer than 3 cells as non-physical, and warns when estimated phase-velocity error exceeds 2%. These runtime thresholds are less conservative than the manual's `lambda/10` rule; a run that does not abort is not automatically well resolved.
- Keep sources and targets at least 15 cells from PML.
- Include 15-20 or more free-space cells above an airborne source.
- `#pml_cells` occupies cells inside the declared domain.
- Geometry coordinates are rounded to the grid.
- Local `utilities.round_value` uses decimal `ROUND_HALF_DOWN` for cell coordinates. A requested half-cell value rounds toward the lower integer (for example, 42.5 cells becomes 42), so audit source/receiver coordinates and steps after quantization.

### Conductive Attenuation Budget

Before a deep-target run, estimate attenuation from the actual permittivity, conductivity, source spectrum, air path, and target depth. For a nondispersive lossy dielectric, use the complex propagation constant rather than only the low-loss approximation when conductivity is not clearly small. Report at least:

- attenuation in Np/m and dB/m at the center and upper significant frequencies;
- one-way and two-way path loss to the target-depth range;
- expected full/control contrast relative to upper-layer clutter in a one-trace smoke;
- the material sensitivity range that would change the accept/reject result.

Passing the wavelength/dispersion gate does not imply a detectable deep reflection. In the first MACRO03 attempt, a deep-cover conductivity of `0.0075 S/m` at `55 MHz` and relative permittivity near `11.8` implied about `3.55 dB/m`; the roughly 26 m two-way path reduced field amplitude to about `2.4e-5`. The geometry was valid but the target was practically hidden. Lower-loss V2 materials preserved the exact geometry while raising the one-trace signed full/control target-window contrast from about `1.55%` to about `50%` of full RMS. Treat this as a workflow lesson, not a universal material value.

### Geometry Ordering

Object construction is ordered. Later objects overwrite prior material assignments. Treat the input as a layered canvas and audit include-file order.

### Fractal Media

- `#fractal_box` supports one normal material or multiple bins from a mixing model.
- In local 3.1.7 source, a single-material fractal box without a roughness/water/grass modifier is rejected; use `#box` instead.
- Seeds are optional in syntax but mandatory for reproducible research.
- Local fractal fields use seeded normal noise, FFT filtering, inverse FFT, scaling/binning, then voxel construction.
- `#add_surface_roughness` modifies one external face of a fractal box and supports a separate deterministic seed.
- Dielectric smoothing is optional for fractal boxes, but cannot rescue a frequency-invalid mixing model.

### Peplinski Guard

The official manual states that `#soil_peplinski` is valid from 0.3 to 1.3 GHz. Do not use it for 50-100 MHz PGDA cases by default. If a future study uses it outside that band, require an explicit cited derivation and sensitivity study.

The bundled `user_models/heterogeneous_soil.in` uses a 1.5 GHz Ricker waveform despite that stated upper limit and omits explicit fractal/roughness seeds. Treat it as a command demonstration, not an automatically paper-valid parameter contract.

### Imported Geometry

Local 3.1.7 source accepts exactly:

```text
#geometry_objects_read: x y z geometry.h5 materials.txt
```

The HDF5 file must contain `/data`, use the model resolution in root attribute `dx_dy_dz`, and have material indices compatible with the material-file command order. `-1` means leave the existing model unchanged.

The current online text mentions an optional smoothing flag, but the reviewed 3.1.7 parser requires exactly five parameters after the command. Installed source wins for this environment.

Externally generated voxel HDF5 does not carry gprMax `rigidE`, `rigidH`, and `ID` arrays, so local source builds plain voxels with dielectric averaging off. Generate spatially correlated fields before quantization and test scattering sensitivity to grid size and bin count.

### GPU And Output

- Official GPU invocation uses `-gpu`, optionally followed by the CUDA device ID.
- GPU receiver output includes the default field components even when a reduced component list is requested.
- Use `--geometry-only` before a full run.
- Use `#geometry_view` when practical; it also reveals PML, source, and receiver locations.
- In local `model_build_run.py`, `--geometry-fixed` preserves the built grid between model runs until the final trace. It is appropriate for a static-geometry B-scan, but it does not prove that separate full/control inputs are matched.

## Review Checklist For Source Changes

When gprMax changes, diff at least:

1. command token counts and optional parameters;
2. coordinate rounding and bounds checks;
3. material ordering for imported geometry;
4. fractal seeding and binning;
5. PML defaults and placement;
6. GPU output component behavior;
7. HDF5 schema and output timing.
