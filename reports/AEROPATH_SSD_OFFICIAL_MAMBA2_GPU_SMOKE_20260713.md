# AeroPath-SSD Official Mamba2 GPU Smoke

Date: 2026-07-13

## Scope

This smoke test validates the implemented AeroPath-SSD model independently of
the formal data-release gate. It does not start a paper experiment.

## Software and hardware

- WSL environment: `aeropath-mamba2`
- PyTorch: `2.7.1+cu128`
- CUDA runtime/toolkit: `12.8`
- Official `mamba-ssm`: `2.2.6`
- `causal-conv1d`: `1.6.2.post1`
- GPU: NVIDIA GeForce RTX 5070, 12 GiB

## Command

```bash
python scripts/smoke_official_mamba2_cuda.py --backward
```

## Accepted result

- Configuration: `configs/aeropath_ssd_v15_formal_blocked.json`
- Input shape: `(1, 6, 501, 256)`
- Backend: bidirectional, dual-axis `official_mamba2`
- Forward pass: passed
- Backward pass: passed
- Peak allocated VRAM: `2179.87 MiB`
- Maximum path-mass error: `6.556510925292969e-07`

The six input channels are one raw B-scan channel plus the five configured
trace-resolved acquisition and terrain metadata channels. The smoke program now
asserts that this contract is explicit, so it cannot accidentally validate a
raw-only input against a metadata-conditioned model.

## Consequence

The official-Mamba2 implementation is viable at the formal 501 x 256 window
shape on the RTX 5070. The formal configuration remains disabled because the
V15 data-release gate, true-negative data, and independent simulation gate are
separate requirements.
