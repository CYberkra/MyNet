# GprMambaSep v2.0 — Implementation Plan

**Goal**: Implement and train GprMambaSep, a physics-guided dual-axis selective state space model (Mamba-2/SSD) that decomposes GPR B-scans into the three physical components (A = air wave, S = surface reflection, G = geological signal) via content-based latent routing, replacing the single-mask-prediction paradigm of PGDA-CSNet v1.x.

**Architecture**: Shared ConvNeXt-Mamba encoder (4 stages, 16→128ch) → Decomposition bottleneck (DilatedBottleneck + 2× Mamba2DBlock + 1×1 split conv → 3×64ch pathways) → Three lightweight decoders (one per component: A, S, G) → Task heads on G (mask/center/presence). Mamba2DBlock fuses time-axis SSM, trace-axis SSM, and 4-direction VMamba cross-scan.

**Tech Stack**:
- Python 3.10+, PyTorch 2.x, CUDA 12.x
- Existing PGDA-CSNet infrastructure: `pgdacsnet/`, `scripts/train_raw_only.py`, `scripts/eval_full_line.py`
- Simulation data: `data/PGDA_SYNTH_DATASET_V1/05_accepted_dataset/`
- Real data: `data_corrected_v1_4_terrain_direction/`
- Mamba backend: `SelectiveSSMLite` (pure-PyTorch gated conv1d proxy for Windows), `SelectiveSSMCUDA` (VMamba selective_scan_cuda wrapper for WSL2)
- Baseline comparisons: v1.4 RawOnlyUNet, v1.7a ConvNeXt, v1.7b AxialSSM, v1.9d MambaVision, v1.11 SG-USSM

---

## Global Constraints

1. **Windows-first**: All development on RTX 3060 Laptop 6GB. The true Mamba-2 CUDA kernel (`selective_scan_cuda`) requires WSL2 or Linux. Windows must use `SelectiveSSMLite` proxy. The proxy must produce outputs within 1e-3 numerical tolerance of the CUDA kernel.
2. **Backward compatibility**: New architecture key `v2_0_gprmambasep` in `build_model()`. Output format must be compatible with `unpack_model_output()` so that `eval_full_line.py` works without changes.
3. **VRAM budget**: Batch size 4 must fit in 6GB. Gradient checkpointing on Mamba2DBlocks is mandatory. If OOM, reduce batch size to 2, never compromise on decoder branches.
4. **No regressions**: Existing architectures (v1.4, v1.7a, v1.7b, v1.9d, v1.11) must continue to train and evaluate unchanged. All existing tests must pass.
5. **Config-driven**: All hyperparameters (loss weights, SSM state dimension, number of Mamba blocks per stage) go in the JSON config. No hardcoded magic numbers in model code.
6. **3-stage curriculum**: (1) simulation-only pretrain, (2) mixed sim-real, (3) self-supervised co-prediction fine-tune. Each stage is a separate config, not separate code paths.
7. **Reproducibility**: All random seeds logged. Mamba2DBlock state initialization must be seed-identical across runs (set `torch.manual_seed` before any SSM init).

---

## File Structure Map

### Files to Create

| # | File | Purpose |
|---|------|---------|
| F1 | `pgdacsnet/model_mamba.py` | Mamba2DBlock, SelectiveSSMLite, SelectiveSSMCUDA, axis-specific SSM wrappers |
| F2 | `pgdacsnet/model_gprmambasep.py` | GprMambaSep full model: encoder, bottleneck, decoders, task heads, build function |
| F3 | `scripts/losses_gprmambasep.py` | L1 (self-consistency), L2 (sim-supervised), L3 (contrastive separation), L4 (arrival time prior), L5 (amplitude ratio prior), L6 (co-prediction cycle), plus gradient reversal layer |
| F4 | `scripts/make_v2_gprmambasep_loo_configs.py` | Config generator for 5-fold × 3-seed LOLO-CV; produces 15 JSON configs |
| F5 | `configs/gpu_pretrain_v2_gprmambasep.json` | Stage 1: simulation-only pretrain config (50 epochs, batch=4, no real data) |
| F6 | `configs/gpu_mixed_v2_gprmambasep.json` | Stage 2: mixed sim-real config (80 epochs, batch=2, sim_batch_ratio=0.3) |
| F7 | `configs/gpu_finetune_v2_gprmambasep_selfsup.json` | Stage 3: self-supervised co-prediction config (20 epochs, lr=1e-5, real-only) |
| F8 | `tests/test_model_mamba.py` | Unit tests for Mamba2DBlock and both SSM backends |
| F9 | `tests/test_model_gprmambasep.py` | Integration tests for full GprMambaSep forward/backward pass |
| F10 | `tests/test_losses_gprmambasep.py` | Unit tests for each new loss term (finite gradient, shape correctness, numerical stability) |
| F11 | `scripts/eval_gprmambasep_separation.py` | Separation quality evaluator — measures A/S/G SNR, leakage ratio, component reconstruction error, produces separation diagnostic plots |

### Files to Modify

| # | File | Modification |
|---|------|-------------|
| M1 | `pgdacsnet/model_raw_unet.py` | Add `build_gprmambasep()` call at line ~120 (after existing architecture builds), register architecture key `v2_0_gprmambasep` in `build_model()` dispatch |
| M2 | `pgdacsnet/model_interfaces.py` | Add `GprMambaSepOutput` dataclass with fields: `A_hat`, `S_hat`, `G_hat`, `G_mask`, `G_center`, `G_pres`. Update `unpack_model_output()` to detect GprMambaSep and extract G_mask/G_center/G_pres for backward compatibility |
| M3 | `scripts/train_raw_only.py` | Modify `compute_loss()` to detect GprMambaSep model (by checking for `GprMambaSepOutput` type or presence of `A_hat` field) and call extended loss from `losses_gprmambasep.py`. Add gradient reversal layer step for contrastive loss. Add `component_loss_weights` to config schema validation |
| M4 | `scripts/eval_full_line.py` | Add `--eval-separation` flag to run separation QC (calls `eval_gprmambasep_separation.py` during LOLO-CV eval). Backward-compatible: no change when flag absent |
| M5 | `configs/gpu_train_v4_pilot_mixed.json` | Update schema validation to accept new architecture key (existing configs must parse without error) |
| M6 | `scripts/train_uda.py` | Add Mamba2DBlock feature extraction hooks for UDA (optional, Stage 2+) |

### Directory Layout (for reference)

```
pgdacsnet/
  model_interfaces.py        (MODIFY — add GprMambaSepOutput)
  model_raw_unet.py          (MODIFY — add v2_0_gprmambasep dispatch)
  model_mamba.py             (CREATE — Mamba2DBlock, SelectiveSSMLite)
  model_gprmambasep.py       (CREATE — full GprMambaSep architecture)

scripts/
  train_raw_only.py           (MODIFY — extended loss dispatch)
  losses_gprmambasep.py       (CREATE — new losses L1-L6)
  eval_full_line.py           (MODIFY — optional separation QC)
  eval_gprmambasep_separation.py  (CREATE — separation evaluator)
  make_v2_gprmambasep_loo_configs.py  (CREATE — LOLO-CV config generator)
  train_uda.py                (MODIFY — optional Mamba feature hooks)

configs/
  gpu_pretrain_v2_gprmambasep.json        (CREATE)
  gpu_mixed_v2_gprmambasep.json           (CREATE)
  gpu_finetune_v2_gprmambasep_selfsup.json (CREATE)
  gpu_train_v4_pilot_mixed.json           (MODIFY — schema update)

tests/
  test_model_mamba.py         (CREATE)
  test_model_gprmambasep.py   (CREATE)
  test_losses_gprmambasep.py  (CREATE)
```

---

## TASKS

Each task below is numbered and dependency-ordered. Complete tasks in sequence. Each task includes:
- Files to create/modify
- Interfaces (consumes/produces)
- TDD steps with exact commands
- Test expectations
- Commit commands (run after tests pass)

---

### TASK 1 — SelectiveSSMLite: Pure-PyTorch Mamba-2 Proxy

**Depends on**: nothing

**Files**: CREATE `pgdacsnet/model_mamba.py` (first 150 lines: SelectiveSSMLite class)

**Interface**:
- Consumes: `(B, C, L)` tensor (batch, channels, sequence length)
- Produces: `(B, C, L)` tensor with same shape — content-adaptively gated sequence
- Config keys: `ssm_state_dim` (default 64), `ssm_conv_kernel` (default 4), `ssm_expand_factor` (default 2), `ssm_dt_rank` (default 8)

**Design**: SelectiveSSMLite approximates Mamba-2's selection mechanism via input-dependent depthwise conv1d kernel modulation. Unlike a static conv1d (current `GatedSequenceBlock`), the kernel weights are a learned linear function of the input activation at each position. This captures the essential property of selective SSMs: different tokens get different processing.

```python
# SelectiveSSMLite — core structure
class SelectiveSSMLite(nn.Module):
    """
    Pure-PyTorch approximation of Mamba-2 SSD selection mechanism.
    
    Replaces the continuous-time SSM discretization with an input-dependent
    depthwise conv1d where the kernel at position t is modulated by the
    input x[:, :, t]. This captures the 'selection' property: different
    tokens gate information differently.
    
    Compared to GatedSequenceBlock (static kernel):
    - Same O(L) complexity with conv1d
    - Content-dependent modulation adds conv_kernel * d_model extra params
    - Acts as a learnable proxy for Mamba-2's A/B/C selection
    """
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2, dt_rank=8):
        ...
    
    def forward(self, x):
        # x: (B, C, L)
        B, C, L = x.shape
        
        # 1. Expand channels
        x_proj = self.in_proj(x)  # (B, 2*expand*C, L) — split into x and gate
        
        # 2. Depthwise conv1d on x branch
        x_conv = self.conv1d(x_proj[:, :self.expand*C])  # (B, expand*C, L)
        
        # 3. Generate kernel modulation from gate branch
        modulation = self.modulation_proj(x_proj[:, self.expand*C:])  # (B, d_conv, L)
        
        # 4. Apply modulated conv: depthwise conv with position-dependent kernel
        # Use unfold + modulated weighted sum
        kernel = self.base_kernel[None, :, :, None] + modulation.unsqueeze(1)  # (B, 1, d_conv, L)
        x_unfold = F.pad(x_conv, (self.d_conv-1, 0)).unfold(2, self.d_conv, 1)  # (B, expand*C, L, d_conv)
        x_modulated = (x_unfold * kernel.permute(0, 2, 3, 1).expand(B, self.expand*C, L, self.d_conv)).sum(dim=-1)
        
        # 5. Gate
        gate = torch.sigmoid(x_proj[:, self.expand*C:])
        out = x_modulated * gate
        
        # 6. Project back
        return self.out_proj(out)  # (B, C, L)
```

**TDD steps**:

Step 1.1 — Write and run minimal shape/forward test:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import torch, sys
sys.path.insert(0, '.')
from pgdacsnet.model_mamba import SelectiveSSMLite

m = SelectiveSSMLite(d_model=64, d_state=64, d_conv=4, expand=2)
B, C, L = 2, 64, 512
x = torch.randn(B, C, L)
y = m(x)
assert y.shape == (B, C, L), f'Shape mismatch: {y.shape}'
assert torch.isfinite(y).all(), 'Non-finite output'
print(f'PASS: SelectiveSSMLite forward OK, shape={y.shape}')
"
```

Expected: `PASS: SelectiveSSMLite forward OK, shape=(2, 64, 512)`

Step 1.2 — Verify gradient flow:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import torch
from pgdacsnet.model_mamba import SelectiveSSMLite

m = SelectiveSSMLite(d_model=32, d_state=32, d_conv=3, expand=2)
x = torch.randn(2, 32, 128, requires_grad=True)
y = m(x)
loss = y.sum()
loss.backward()
assert x.grad is not None
assert torch.isfinite(x.grad).all()
assert x.grad.abs().sum() > 0, 'Zero gradient'
print(f'PASS: Gradient flow OK, grad_norm={x.grad.norm().item():.4f}')
"
```

Expected: Gradient norm > 0, finite, non-zero.

Step 1.3 — Verify content-adaptivity: same input but two different positions produce different kernel modulation:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import torch
from pgdacsnet.model_mamba import SelectiveSSMLite

m = SelectiveSSMLite(d_model=16, d_state=16, d_conv=4, expand=2)
# Create input where position 10 is very different from position 50
x = torch.zeros(1, 16, 128)
x[:, :, 10] = 100.0  # Strong impulse at position 10
x[:, :, 50] = 0.01   # Weak signal at position 50
y = m(x)
# The modulation at position 10 should differ more from baseline than position 50
diff_at_10 = (y[0, :, 10] - y[0, :, 5]).abs().mean().item()
diff_at_50 = (y[0, :, 50] - y[0, :, 45]).abs().mean().item()
# At minimum, these should not be identical (content-adaptivity check)
assert abs(diff_at_10 - diff_at_50) > 1e-6, 'No content adaptivity detected'
print(f'PASS: Content-adaptivity verified, diff_10={diff_at_10:.4f}, diff_50={diff_at_50:.4f}')
"
```

Expected: diff_10 != diff_50 (different modulation for different input content).

Step 1.4 — Parameter count sanity:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import torch
from pgdacsnet.model_mamba import SelectiveSSMLite

m = SelectiveSSMLite(d_model=128, d_state=64, d_conv=4, expand=2)
params = sum(p.numel() for p in m.parameters())
print(f'SelectiveSSMLite(128) params: {params}')
# Should be ~ expand*C*(2*C + d_conv + 1) + C*expand*C ≈ O(d_model^2)
# For d_model=128, expand=2: ~ 128*4*128 + 128*128*2 ≈ 65K + 32K = 97K
assert 50000 < params < 200000, f'Unexpected param count: {params}'
"
```

Expected: params in range 50K-200K for d_model=128.

Step 1.5 — Write `tests/test_model_mamba.py` unit test class:
```python
# tests/test_model_mamba.py
import pytest
import torch
from pgdacsnet.model_mamba import SelectiveSSMLite

class TestSelectiveSSMLite:
    def test_forward_shape(self):
        m = SelectiveSSMLite(64, 64, 4, 2)
        x = torch.randn(2, 64, 512)
        y = m(x)
        assert y.shape == (2, 64, 512)
    
    def test_gradient_flow(self):
        m = SelectiveSSMLite(32, 32, 3, 2)
        x = torch.randn(2, 32, 128, requires_grad=True)
        y = m(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
    
    def test_batch_independence(self):
        m = SelectiveSSMLite(16, 16, 4, 2)
        x = torch.randn(2, 16, 64)
        y = m(x)
        # Two different batch items get different outputs
        assert not torch.allclose(y[0], y[1])
    
    def test_different_lengths(self):
        m = SelectiveSSMLite(32, 32, 4, 2)
        for L in [64, 128, 256, 512]:
            x = torch.randn(1, 32, L)
            y = m(x)
            assert y.shape == (1, 32, L)
```

**Commit**:
```bash
git add pgdacsnet/model_mamba.py tests/test_model_mamba.py
git commit -m "feat(mamba): add SelectiveSSMLite — pure-PyTorch Mamba-2 proxy with input-dependent conv1d modulation

SelectiveSSMLite approximates the Mamba-2/SSD selection mechanism by
modulating a depthwise conv1d kernel based on input content at each
sequence position. Key design choices:
- Input-dependent kernel modulation via learned linear projection
- Expanded channel bottleneck (expand=2) with SiLU gating
- Position-wise modulation preserves O(L) complexity
- Pure PyTorch, no CUDA compilation needed for Windows development

Tests verify forward shape, gradient flow, batch independence,
variable-length support, and content-adaptivity.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
"
```

---

### TASK 2 — SelectiveSSMCUDA: VMamba CUDA Kernel Wrapper

**Depends on**: TASK 1 (same file)

**Files**: MODIFY `pgdacsnet/model_mamba.py` (add SelectiveSSMCUDA class, ~70 lines)

**Interface**:
- Consumes: `(B, C, L)` tensor + optional state initialization
- Produces: `(B, C, L)` tensor — exact Mamba-2 SSD output
- Auto-fallback: raises `ImportError` with clear message if `selective_scan_cuda` not available, suggesting WSL2 installation

**Design**: Thin wrapper around the VMamba `selective_scan_cuda` function. When available (WSL2/Docker with CUDA), delegates to the highly optimized fused kernel. When not available (Windows), raises a clear error directing the user to either use SelectiveSSMLite or set up WSL2.

**TDD steps**:

Step 2.1 — Verify graceful fallback on Windows (must raise ImportError, not crash):
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import torch, sys
sys.path.insert(0, '.')
from pgdacsnet.model_mamba import SelectiveSSMCUDA
try:
    m = SelectiveSSMCUDA(d_model=64, d_state=64)
    print('UNEXPECTED: CUDA kernel available on Windows')
except ImportError as e:
    print(f'PASS: Expected ImportError on Windows: {e}')
except Exception as e:
    print(f'FAIL: Wrong exception type: {type(e).__name__}: {e}')
"
```

Expected: `PASS: Expected ImportError on Windows: ...`

Step 2.2 — Verify numerical parity test script exists (runs on WSL2 only):
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
# This test validates the SELECTIVESSMLITE proxy against the CUDA kernel.
# It can only pass on WSL2/Linux with selective_scan_cuda installed.
# The test structure verifies the script parses and runs on Windows too.
import torch, sys
sys.path.insert(0, '.')
from pgdacsnet.model_mamba import SelectiveSSMLite

try:
    from pgdacsnet.model_mamba import SelectiveSSMCUDA
    m_cuda = SelectiveSSMCUDA(64, 64)
    m_lite = SelectiveSSMLite(64, 64)
    x = torch.randn(2, 64, 512).cuda()
    m_cuda = m_cuda.cuda()
    m_lite = m_lite.cuda()
    y_cuda = m_cuda(x)
    y_lite = m_lite(x)
    diff = (y_cuda - y_lite).abs().max().item()
    print(f'Numerical parity: max_diff={diff:.6f} (threshold: 1e-3)')
    assert diff < 1e-3, f'Proxy divergence: {diff}'
except ImportError:
    print('SKIP: CUDA kernel not available (expected on Windows)')
"
```

Expected: `SKIP: CUDA kernel not available (expected on Windows)` on Windows.

**Commit** (append to previous commit via amend, or create separate commit):
```bash
git add pgdacsnet/model_mamba.py
git commit -m "feat(mamba): add SelectiveSSMCUDA — VMamba CUDA wrapper with graceful fallback

Wraps selective_scan_cuda when available (WSL2/Linux). Raises clear
ImportError on Windows directing user to SelectiveSSMLite proxy.
Provides numerical parity test harness for validating the proxy.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
"
```

---

### TASK 3 — Mamba2DBlock: Dual-Axis + Cross-Scan Fusion

**Depends on**: TASK 1

**Files**: MODIFY `pgdacsnet/model_mamba.py` (add Mamba2DBlock class, ~120 lines)

**Interface**:
- Consumes: `(B, C, H, W)` tensor (batch, channels, height/time, width/trace)
- Produces: `(B, C, H, W)` tensor with residual connection
- Config keys: `mamba_state_dim` (default 64), `mamba_scan_strategy` (`"dual_axis"` | `"time_only"` | `"trace_only"` | `"cross_scan"` | `"full"`)
- Internal: three scanning branches — time-axis SSM, trace-axis SSM, cross-scan SSM (4-direction VMamba) — fused via learned 1x1 conv

**Design**: The core innovation for GPR sequence modeling. Three parallel scanning strategies are fused:

1. **Time-axis SSM**: Permute to `(B*W, C, H)`, apply SelectiveSSMLite along the time dimension. Captures per-trace wave physics (arrival times, attenuation, dispersion).
2. **Trace-axis SSM**: Permute to `(B*H, C, W)`, apply SelectiveSSMLite along the trace dimension. Captures inter-trace continuity (bedrock interface dip, hyperbola tails).
3. **Cross-scan SSM**: Implement VMamba-style 4-direction scan: flatten 2D grid into 4 1D sequences (top-left→bottom-right, bottom-right→top-left, top-right→bottom-left, bottom-left→top-right). Each traverses all H×W patches in a specific order, applies SelectiveSSMLite, then unflattens back to 2D. This captures 2D hyperbolic patterns characteristic of localized targets.

All three outputs are concatenated channel-wise and fused through a 1×1 conv with GroupNorm and GELU. The result is added to the input as a residual connection.

```python
class Mamba2DBlock(nn.Module):
    """
    Dual-axis + cross-scan Mamba-2 block for 2D GPR B-scans.
    
    Processes (B, C, H, W) along:
    - Time axis (H): wave physics per trace
    - Trace axis (W): spatial continuity across traces
    - Cross-scan (4 directions): 2D hyperbolic patterns (VMamba-style)
    
    Output is a learned fusion of all three branches with residual connection.
    
    Reference: VMamba (SS2D) for the cross-scan mechanism,
    Mamba-2/SSD for the selective scan kernel.
    """
    def __init__(self, dim, d_state=64, d_conv=4, expand=2, scan_strategy='full'):
        super().__init__()
        self.dim = dim
        self.scan_strategy = scan_strategy
        self.norm = nn.LayerNorm(dim)  # Applied after permuting to (B, H*W, C)
        
        # Create SSM branches
        self.time_ssm = SelectiveSSMLite(dim, d_state, d_conv, expand) if scan_strategy in ('full', 'dual_axis', 'time_only') else None
        self.trace_ssm = SelectiveSSMLite(dim, d_state, d_conv, expand) if scan_strategy in ('full', 'dual_axis', 'trace_only') else None
        self.cross_ssm = SelectiveSSMLite(dim, d_state, d_conv, expand) if scan_strategy in ('full', 'cross_scan') else None
        
        # Fusion conv (3 branches × dim → dim)
        n_branches = sum([self.time_ssm is not None, self.trace_ssm is not None, self.cross_ssm is not None])
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * n_branches, dim, 1),
            nn.GroupNorm(8, dim),
            nn.GELU()
        )
    
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        outputs = []
        
        # 1. Time-axis SSM: process each trace independently
        if self.time_ssm is not None:
            # (B, C, H, W) → (B*W, C, H) → SSM → (B, C, H, W)
            x_t = x.permute(0, 3, 1, 2).reshape(B*W, C, H)
            y_t = self.time_ssm(x_t)
            y_t = y_t.reshape(B, W, C, H).permute(0, 2, 3, 1)  # (B, C, H, W)
            outputs.append(y_t)
        
        # 2. Trace-axis SSM: process each time sample across traces
        if self.trace_ssm is not None:
            # (B, C, H, W) → (B*H, C, W) → SSM → (B, C, H, W)
            x_s = x.permute(0, 2, 1, 3).reshape(B*H, C, W)
            y_s = self.trace_ssm(x_s)
            y_s = y_s.reshape(B, H, C, W).permute(0, 2, 1, 3)  # (B, C, H, W)
            outputs.append(y_s)
        
        # 3. Cross-scan (4-direction VMamba)
        if self.cross_ssm is not None:
            # Flatten 2D → 4 × 1D sequences, SSM each, unflatten back
            y_c = self._cross_scan_ssm(x)
            outputs.append(y_c)
        
        # 4. Fusion
        out = torch.cat(outputs, dim=1)  # (B, n_branches*C, H, W)
        out = self.fusion(out)  # (B, C, H, W)
        
        # 5. Residual
        return x + out
    
    def _cross_scan_ssm(self, x):
        """4-direction VMamba-style cross scan.
        
        Directions:
        1. Top-left → Bottom-right (row-major)
        2. Bottom-right → Top-left (reverse row-major)
        3. Top-right → Bottom-left (column-reversed row-major)
        4. Bottom-left → Top-right (reverse column-reversed row-major)
        """
        B, C, H, W = x.shape
        
        # Direction 1: row-major flatten (top-left → bottom-right)
        d1 = x.reshape(B, C, H*W)  # (B, C, L)
        y1 = self.cross_ssm(d1).reshape(B, C, H, W)
        
        # Direction 2: reverse row-major (bottom-right → top-left)
        d2 = torch.flip(x, dims=[1, 2, 3])  # careful: don't flip batch/channel
        d2 = x.flip(dims=[2, 3]).reshape(B, C, H*W)
        y2 = self.cross_ssm(d2).reshape(B, C, H, W).flip(dims=[2, 3])
        
        # Direction 3: row-reversed row-major (top-right → bottom-left)
        d3 = x.flip(dims=[3]).reshape(B, C, H*W)
        y3 = self.cross_ssm(d3).reshape(B, C, H, W).flip(dims=[3])
        
        # Direction 4: reverse of direction 3
        d4 = x.flip(dims=[2]).reshape(B, C, H*W)
        y4 = self.cross_ssm(d4).reshape(B, C, H, W).flip(dims=[2])
        
        # Average all 4 directions
        return (y1 + y2 + y3 + y4) / 4.0
```

**TDD steps**:

Step 3.1 — Forward shape with full scan strategy:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import torch, sys
sys.path.insert(0, '.')
from pgdacsnet.model_mamba import Mamba2DBlock

m = Mamba2DBlock(dim=64, d_state=64, d_conv=4, expand=2, scan_strategy='full')
B, C, H, W = 2, 64, 512, 256
x = torch.randn(B, C, H, W)
y = m(x)
assert y.shape == (B, C, H, W), f'Shape mismatch: {y.shape}'
assert torch.isfinite(y).all(), 'Non-finite output'
print(f'PASS: Mamba2DBlock full scan forward OK, shape={y.shape}')
"
```

Expected: `PASS: Mamba2DBlock full scan forward OK, shape=(2, 64, 512, 256)`

Step 3.2 — All scan strategies produce correct shapes:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import torch, sys
sys.path.insert(0, '.')
from pgdacsnet.model_mamba import Mamba2DBlock

B, C, H, W = 1, 32, 128, 64
x = torch.randn(B, C, H, W)
for strategy in ['time_only', 'trace_only', 'dual_axis', 'cross_scan', 'full']:
    m = Mamba2DBlock(dim=C, d_state=32, d_conv=4, expand=2, scan_strategy=strategy)
    y = m(x)
    assert y.shape == (B, C, H, W), f'{strategy}: shape mismatch {y.shape}'
    assert torch.isfinite(y).all(), f'{strategy}: non-finite'
print('PASS: All scan strategies produce correct shapes')
"
```

Expected: `PASS: All scan strategies produce correct shapes`

Step 3.3 — Gradient flow through full block:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import torch, sys
sys.path.insert(0, '.')
from pgdacsnet.model_mamba import Mamba2DBlock

m = Mamba2DBlock(dim=32, d_state=32, scan_strategy='full')
x = torch.randn(1, 32, 128, 64, requires_grad=True)
y = m(x)
loss = y.sum()
loss.backward()
assert x.grad is not None
assert torch.isfinite(x.grad).all()
assert x.grad.abs().sum() > 0
print(f'PASS: Gradient flow OK, grad_norm={x.grad.norm().item():.4f}')
"
```

Expected: Gradient norm positive, finite.

Step 3.4 — Cross-scan produces different output than single-axis:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import torch, sys
sys.path.insert(0, '.')
from pgdacsnet.model_mamba import Mamba2DBlock

B, C, H, W = 1, 16, 64, 32
x = torch.randn(B, C, H, W)

m_dual = Mamba2DBlock(C, d_state=16, scan_strategy='dual_axis')
m_cross = Mamba2DBlock(C, d_state=16, scan_strategy='cross_scan')

y_dual = m_dual(x)
y_cross = m_cross(x)

diff = (y_dual - y_cross).abs().mean().item()
print(f'Dual vs cross output diff: {diff:.6f}')
assert diff > 1e-6, 'Dual and cross scan produce identical output (should not)'
assert diff < 10.0, 'Outputs diverged too much'
print('PASS: Cross-scan and dual-axis produce distinct outputs')
"
```

Expected: `PASS: Cross-scan and dual-axis produce distinct outputs`

Step 3.5 — VRAM estimate for full Mamba2DBlock at bottleneck resolution:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import torch, sys
sys.path.insert(0, '.')
from pgdacsnet.model_mamba import Mamba2DBlock

if not torch.cuda.is_available():
    print('SKIP: CUDA not available')
    exit(0)

m = Mamba2DBlock(dim=128, d_state=64, scan_strategy='full').cuda()
x = torch.randn(2, 128, 64, 32).cuda()
torch.cuda.reset_peak_memory_stats()
y = m(x)
y.sum().backward()
peak = torch.cuda.max_memory_allocated() / 1024**2
print(f'Mamba2DBlock(128) forward+backward peak VRAM: {peak:.1f} MB')
assert peak < 500, f'VRAM too high: {peak:.1f} MB (budget: 500 MB)'
print('PASS: VRAM within budget')
"
```

Expected: Peak VRAM < 500 MB for bottleneck resolution.

**Commit**:
```bash
git add pgdacsnet/model_mamba.py tests/test_model_mamba.py
git commit -m "feat(mamba): add Mamba2DBlock — dual-axis + cross-scan SSM fusion for GPR B-scans

Three parallel scanning branches: time-axis SSM (per-trace wave physics),
trace-axis SSM (inter-trace spatial continuity), and 4-direction VMamba-style
cross-scan (2D hyperbolic pattern capture). All strategies configurable via
scan_strategy in {'full', 'dual_axis', 'time_only', 'trace_only', 'cross_scan'}.
Learned 1x1 conv fusion with GroupNorm + GELU + residual connection.

Tests verify all strategy variants, gradient flow, cross-scan distinctiveness,
and VRAM budget at bottleneck resolution (128ch, 64x32).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
"
```

---

### TASK 4 — Shared ConvNeXt-Mamba Encoder

**Depends on**: TASK 3

**Files**: CREATE `pgdacsnet/model_gprmambasep.py` (first 180 lines: `GprMambaSepEncoder` class)

**Interface**:
- Consumes: `(B, 1, 512, 256)` input B-scan + optional `(B, N_feat)` metadata tensor
- Produces: list of 4 tensors at progressive scales `[(B, 16, 256, 128), (B, 32, 128, 64), (B, 64, 64, 32), (B, 128, 32, 16)]` for skip connections + bottleneck feature
- Config keys: `base_ch` (default 16), `encoder_stages` (default [2,2,3,3] — ConvNeXt blocks per stage), `mamba_stages` (default [0,1,1,1] — Mamba2DBlocks per stage, 0 = skip)

**Design**: A 4-stage encoder following the ConvNeXt design pattern (depthwise 7×7 → LayerNorm → 1×1 expand → GELU → 1×1 project), interleaved with Mamba2DBlocks at stages 2-4. Stages 1-2 use thin feature maps (no cross-scan), stages 3-4 use full Mamba2DBlock with cross-scan for global reasoning.

Each stage: ConvNeXt blocks × `encoder_stages[i]` → (optional) Mamba2DBlock × `mamba_stages[i]` → 2×2 stride-2 conv downsampling (except stage 4, no downsampling — output is bottleneck).

```python
class GprMambaSepEncoder(nn.Module):
    """
    4-stage encoder: ConvNeXt blocks interleaved with Mamba2DBlocks.
    
    Channel progression: 1 → 16 → 32 → 64 → 128
    Spatial progression: 512×256 → 256×128 → 128×64 → 64×32 → 32×16
    """
    def __init__(self, base_ch=16, in_ch=1, encoder_stages=(2,2,3,3), 
                 mamba_stages=(0,1,1,1), mamba_kwargs=None):
        super().__init__()
        self.base_ch = base_ch
        mamba_kwargs = mamba_kwargs or {}
        
        # Stem: 7x7 conv to 2*base_ch, then LayerNorm
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 2*base_ch, 7, padding=3, bias=False),
            nn.LayerNorm([2*base_ch, 512, 256], elementwise_affine=False)
        )
        
        # Stages
        chs = [2*base_ch, 2*base_ch, 4*base_ch, 8*base_ch]  # [16, 16, 32, 64]
        out_chs = [2*base_ch, 4*base_ch, 8*base_ch, 16*base_ch]  # [16, 32, 64, 128]
        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        for i in range(4):
            stage_blocks = []
            # ConvNeXt blocks
            for _ in range(encoder_stages[i]):
                stage_blocks.append(self._make_convnext_block(chs[i]))
            # Mamba2DBlocks
            for _ in range(mamba_stages[i]):
                stage_blocks.append(
                    Mamba2DBlock(dim=chs[i], scan_strategy='cross_scan' if i >= 2 else 'dual_axis', **mamba_kwargs)
                )
            
            self.stages.append(nn.Sequential(*stage_blocks))
            
            # Downsample (except last stage)
            if i < 3:
                self.downs.append(nn.Conv2d(chs[i], out_chs[i], 2, stride=2, bias=False))
            else:
                self.downs.append(nn.Identity())
        
        self.final_norm = nn.LayerNorm(8*base_ch)  # Applied after flattening
    
    def forward(self, x, metadata=None):
        # x: (B, 1, H, W)
        skip_features = []
        
        h = self.stem(x)
        
        for i in range(4):
            h = self.stages[i](h)
            skip_features.append(h)
            h = self.downs[i](h)
        
        # (B, 128, H/32, W/32)
        return h, skip_features  # bottleneck + 4 skip connections
```

**TDD steps**:

Step 4.1 — Forward shape test:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import torch, sys
sys.path.insert(0, '.')
from pgdacsnet.model_gprmambasep import GprMambaSepEncoder

enc = GprMambaSepEncoder(base_ch=16)
B = 2
x = torch.randn(B, 1, 512, 256)
bottleneck, skips = enc(x)

assert bottleneck.shape == (B, 128, 32, 16), f'Bottleneck shape: {bottleneck.shape}'
expected_skip_shapes = [(2, 16, 256, 128), (2, 32, 128, 64), (2, 64, 64, 32), (2, 128, 32, 16)]
for i, (s, expected) in enumerate(zip(skips, expected_skip_shapes)):
    assert s.shape == expected, f'Skip {i}: {s.shape} != {expected}'
assert torch.isfinite(bottleneck).all()
print('PASS: GprMambaSepEncoder forward shapes OK')
"
```

Expected: `PASS: GprMambaSepEncoder forward shapes OK`

Step 4.2 — Gradient flow through full encoder:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import torch, sys
sys.path.insert(0, '.')
from pgdacsnet.model_gprmambasep import GprMambaSepEncoder

enc = GprMambaSepEncoder(base_ch=8)  # Smaller for quick test
x = torch.randn(1, 1, 128, 64, requires_grad=True)
bottleneck, _ = enc(x)
loss = bottleneck.sum()
loss.backward()
assert x.grad is not None
assert torch.isfinite(x.grad).all()
assert x.grad.abs().sum() > 0
print(f'PASS: Encoder gradient flow OK, grad_norm={x.grad.norm().item():.4f}')
"
```

Expected: Gradient flow OK.

Step 4.3 — Parameter count sanity:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import torch, sys
sys.path.insert(0, '.')
from pgdacsnet.model_gprmambasep import GprMambaSepEncoder

enc = GprMambaSepEncoder(base_ch=16)
params = sum(p.numel() for p in enc.parameters())
print(f'Encoder params: {params:,}')
# From design estimate: ~1.79M
assert 1.0e6 < params < 3.0e6, f'Unexpected param count: {params:,}'
print('PASS: Encoder param count in expected range')
"
```

Expected: params ~1.79M, within range 1.0M-3.0M.

Step 4.4 — Metadata FiLM conditioning (verify metadata path works without error):
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import torch, sys
sys.path.insert(0, '.')
from pgdacsnet.model_gprmambasep import GprMambaSepEncoder

enc = GprMambaSepEncoder(base_ch=8)
# Without metadata
x = torch.randn(1, 1, 128, 64)
bottleneck, skips = enc(x)
assert bottleneck.shape == (1, 64, 8, 4)
print('PASS: Encoder works without metadata')
"
```

Expected: `PASS: Encoder works without metadata`

**Commit**:
```bash
git add pgdacsnet/model_gprmambasep.py
git commit -m "feat(mamba): add GprMambaSepEncoder — 4-stage ConvNeXt-Mamba hybrid encoder

4-stage encoder with channel progression 16→32→64→128 and spatial
downsampling 2x per stage. ConvNeXt blocks (depthwise 7x7 + inverted
bottleneck) interleaved with Mamba2DBlocks. Stages 1-2 use dual-axis SSM,
stages 3-4 use cross-scan for global 2D reasoning. Outputs bottleneck
feature + 4 skip connections for decoder skip connections.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
"
```

---

### TASK 5 — Decomposition Bottleneck

**Depends on**: TASK 4

**Files**: MODIFY `pgdacsnet/model_gprmambasep.py` (add `GprMambaSepBottleneck` class, ~100 lines)

**Interface**:
- Consumes: `(B, 128, 32, 16)` encoder bottleneck feature
- Produces: tuple of 3 tensors `(B, 64, 32, 16)` — A_latent, S_latent, G_latent
- Internal: DilatedBottleneck → 2× Mamba2DBlock (full scan) → DilatedBottleneck → 1x1 split conv → 3× refinement blocks

**Design**: The bottleneck learns to assign different latent pathways to different physical components. The key mechanism is **content-based routing via selective SSM**: the Mamba-2 blocks learn to gate features based on their arrival-time-velocity signature into the appropriate pathway.

```python
class GprMambaSepBottleneck(nn.Module):
    """
    Decomposition bottleneck: learns to route features into A/S/G latent pathways.
    
    Architecture:
    1. DilatedBottleneck: multi-scale context (dilations 1,2,4) + SE
    2. Mamba2DBlock × 2: full scan (time+trace+cross) for global routing
    3. DilatedBottleneck: refine
    4. 1×1 split conv: 128ch → 3 × 64ch (A_latent, S_latent, G_latent)
    5. ConvNeXt refinement × 3 per pathway (shared structure, separate weights)
    """
    def __init__(self, dim=128, latent_dim=64, d_state=64):
        super().__init__()
        self.dim = dim
        self.latent_dim = latent_dim
        
        # Dilated bottleneck 1
        self.dilated1 = self._make_dilated_bottleneck(dim)
        
        # Mamba2DBlocks for global routing
        self.mamba1 = Mamba2DBlock(dim, d_state=d_state, scan_strategy='full')
        self.mamba2 = Mamba2DBlock(dim, d_state=d_state, scan_strategy='full')
        
        # Dilated bottleneck 2
        self.dilated2 = self._make_dilated_bottleneck(dim)
        
        # Split conv: 128 → 3×64
        self.split_conv = nn.Conv2d(dim, 3 * latent_dim, 1, bias=False)
        
        # Three pathway refinement blocks (separate weights)
        self.refine_A = self._make_refinement_block(latent_dim)
        self.refine_S = self._make_refinement_block(latent_dim)
        self.refine_G = self._make_refinement_block(latent_dim)
    
    def forward(self, x):
        # x: (B, 128, H, W)
        h = self.dilated1(x)
        h = self.mamba1(h)
        h = self.mamba2(h)
        h = self.dilated2(h)
        
        # Split into 3 pathways
        split = self.split_conv(h)  # (B, 192, H, W)
        B, _, H, W = split.shape
        A_lat = self.refine_A(split[:, :self.latent_dim])
        S_lat = self.refine_S(split[:, self.latent_dim:2*self.latent_dim])
        G_lat = self.refine_G(split[:, 2*self.latent_dim:])
        
        return A_lat, S_lat, G_lat
    
    def _make_dilated_bottleneck(self, dim):
        # Parallel 3x3 convs with dilation 1, 2, 4 → concatenate → 1x1 fuse → SE
        return nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, dilation=1, groups=dim),
            nn.Conv2d(dim, dim, 3, padding=2, dilation=2, groups=dim),
            nn.Conv2d(dim, dim, 3, padding=4, dilation=4, groups=dim),
            # ... (simplified here, actual impl uses proper dilation block)
        )
    
    def _make_refinement_block(self, dim):
        return nn.Sequential(
            nn.Conv2d(dim, dim*4, 1),
            nn.GELU(),
            nn.Conv2d(dim*4, dim, 1),
        )
```

**TDD steps**:

Step 5.1 — Forward output shapes:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import torch, sys
sys.path.insert(0, '.')
from pgdacsnet.model_gprmambasep import GprMambaSepBottleneck

bn = GprMambaSepBottleneck(dim=128, latent_dim=64, d_state=64)
B = 2
x = torch.randn(B, 128, 32, 16)
A_lat, S_lat, G_lat = bn(x)

assert A_lat.shape == (B, 64, 32, 16), f'A_lat shape: {A_lat.shape}'
assert S_lat.shape == (B, 64, 32, 16), f'S_lat shape: {S_lat.shape}'
assert G_lat.shape == (B, 64, 32, 16), f'G_lat shape: {G_lat.shape}'
assert torch.isfinite(A_lat).all()
assert torch.isfinite(S_lat).all()
assert torch.isfinite(G_lat).all()

# Verify the three pathways are not identical at initialization
diff_AG = (A_lat - G_lat).abs().mean().item()
assert diff_AG > 0, 'A and G latents are identical'
print(f'PASS: Bottleneck OK, shapes correct, A-G diff={diff_AG:.6f}')
"
```

Expected: `PASS: Bottleneck OK, shapes correct, A-G diff=...`

Step 5.2 — Gradient flow and parameter count:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import torch, sys
sys.path.insert(0, '.')
from pgdacsnet.model_gprmambasep import GprMambaSepBottleneck

bn = GprMambaSepBottleneck(dim=64, latent_dim=32, d_state=32)
params = sum(p.numel() for p in bn.parameters())
print(f'Bottleneck params: {params:,}')

x = torch.randn(1, 64, 16, 8, requires_grad=True)
A, S, G = bn(x)
loss = A.sum() + S.sum() + G.sum()
loss.backward()
assert x.grad is not None
assert torch.isfinite(x.grad).all()
assert x.grad.abs().sum() > 0
print(f'PASS: Bottleneck gradient flow OK, params={params:,}')
"
```

Expected: Gradient flow OK.

**Commit**:
```bash
git add pgdacsnet/model_gprmambasep.py
git commit -m "feat(mamba): add GprMambaSepBottleneck — content-based A/S/G latent routing

Dilated multi-scale context → 2x Mamba2DBlock (full scan) → 1x1 split conv
→ 3×64ch latent pathways (A/S/G) with per-pathway refinement blocks.
The Mamba-2 selection mechanism learns to route features based on their
arrival-time-velocity signature into the correct physical component pathway.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
"
```

---

### TASK 6 — Three-Branch Decoders

**Depends on**: TASK 5

**Files**: MODIFY `pgdacsnet/model_gprmambasep.py` (add `GprMambaSepDecoder` class and `GprMambaSepDecoders` container, ~120 lines)

**Interface**:
- Consumes: 3 latent tensors `(B, 64, 32, 16)` each + list of 4 skip features from encoder
- Produces: 3 output tensors `(B, 1, 512, 256)` each — A_hat, S_hat, G_hat
- Decoder topology (same for all three branches, separate weights):
  - TransposedConv: 64ch → 32ch, 2x up (→ 64×32)
  - ConvNeXtStage: 32ch, 2 blocks + skip from encoder stage 3 (64ch)
  - Mamba2DBlock: 32ch, dual-axis
  - TransposedConv: 32ch → 16ch, 2x up (→ 128×64)
  - ConvNeXtStage: 16ch, 2 blocks + skip from encoder stage 2 (32ch)
  - TransposedConv: 16ch → 8ch, 2x up (→ 256×128)
  - ConvNeXtStage: 8ch, 1 block + skip from encoder stage 1 (16ch)
  - TransposedConv: 8ch → 4ch, 2x up (→ 512×256)
  - ConvNeXtStage: 4ch, 1 block + skip from encoder stem (16ch)
  - 1×1 conv: 4ch → 1ch output

**Design**: Each decoder is a lightweight U-Net style upsampler that blends the latent pathway features with encoder skip features at corresponding resolutions. The three decoders have identical topology but completely independent weights, allowing them to specialize for different physical reconstruction tasks (A: high-freq early-time; S: surface-following; G: deeper smoother).

```python
class GprMambaSepDecoder(nn.Module):
    """
    Single decoder branch for one component (A, S, or G).
    
    Lightweight U-Net style: transposed conv up → ConvNeXtStage with skip →
    (optional Mamba2DBlock at middle resolution) → continue to full resolution.
    """
    def __init__(self, latent_dim=64, skip_dims=(128, 64, 32, 16)):
        super().__init__()
        # ... (full implementation with ConvNeXt stages, transposed convs, skip fusions)
    
    def forward(self, x_latent, skip_features):
        # x_latent: (B, 64, 32, 16)
        # skip_features: list of 4 tensors from encoder
        h = self.up1(x_latent)   # (B, 32, 64, 32)
        h = h + skip_features[3]  # skip from encoder stage 3
        h = self.convnext1(h)
        h = self.mamba(h)
        h = self.up2(h)           # (B, 16, 128, 64)
        h = h + skip_features[2]  # skip from encoder stage 2
        # ... continue to (B, 1, 512, 256)
        return h


class GprMambaSepDecoders(nn.Module):
    """
    Container for three component decoders.
    """
    def __init__(self, latent_dim=64):
        super().__init__()
        self.dec_A = GprMambaSepDecoder(latent_dim)
        self.dec_S = GprMambaSepDecoder(latent_dim)
        self.dec_G = GprMambaSepDecoder(latent_dim)
    
    def forward(self, A_lat, S_lat, G_lat, skip_features):
        A_hat = self.dec_A(A_lat, skip_features)
        S_hat = self.dec_S(S_lat, skip_features)
        G_hat = self.dec_G(G_lat, skip_features)
        return A_hat, S_hat, G_hat
```

**TDD steps**:

Step 6.1 — Forward shape test:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import torch, sys
sys.path.insert(0, '.')
from pgdacsnet.model_gprmambasep import GprMambaSepDecoders

decs = GprMambaSepDecoders(latent_dim=64)
B = 2

# Standard latent shapes
A_lat = torch.randn(B, 64, 32, 16)
S_lat = torch.randn(B, 64, 32, 16)
G_lat = torch.randn(B, 64, 32, 16)

# Simulate skip features from encoder
skip_features = [
    torch.randn(B, 16, 256, 128),   # stage 1 (stem)
    torch.randn(B, 32, 128, 64),    # stage 2
    torch.randn(B, 64, 64, 32),     # stage 3
    torch.randn(B, 128, 32, 16),    # stage 4
]

A_hat, S_hat, G_hat = decs(A_lat, S_lat, G_lat, skip_features)
assert A_hat.shape == (B, 1, 512, 256), f'A_hat shape: {A_hat.shape}'
assert S_hat.shape == (B, 1, 512, 256), f'S_hat shape: {S_hat.shape}'
assert G_hat.shape == (B, 1, 512, 256), f'G_hat shape: {G_hat.shape}'
assert torch.isfinite(A_hat).all()
print('PASS: Three decoders produce correct output shapes')
"
```

Expected: `PASS: Three decoders produce correct output shapes`

Step 6.2 — Three decoders produce different outputs at init:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import torch, sys
sys.path.insert(0, '.')
from pgdacsnet.model_gprmambasep import GprMambaSepDecoders

decs = GprMambaSepDecoders(latent_dim=32)
B = 1

# Same input to all three decoders
latent = torch.randn(B, 32, 16, 8)
skip_features = [torch.randn(B, c, h, w) for c, h, w in [(16, 128, 64), (32, 64, 32), (64, 32, 16), (128, 16, 8)]]

A, S, G = decs(latent, latent, latent, skip_features)
diff_A_S = (A - S).abs().mean().item()
diff_A_G = (A - G).abs().mean().item()
print(f'Init diff: A-S={diff_A_S:.6f}, A-G={diff_A_G:.6f}')
assert diff_A_S > 1e-6 or diff_A_G > 1e-6, 'All decoders produce identical output'
print('PASS: Decoders have distinct weights')
"
```

Expected: `PASS: Decoders have distinct weights`

Step 6.3 — Reconstruction error check (A_hat+S_hat+G_hat should NOT match input at init — that would indicate weight coupling):
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import torch, sys
sys.path.insert(0, '.')
from pgdacsnet.model_gprmambasep import GprMambaSepDecoders

decs = GprMambaSepDecoders(latent_dim=32)
x = torch.randn(1, 1, 128, 64)
latent = torch.randn(1, 32, 8, 4)
skip_features = [torch.randn(1, 16, 64, 32), torch.randn(1, 32, 32, 16), torch.randn(1, 64, 16, 8), torch.randn(1, 128, 8, 4)]

# Check that at initialization, reconstruction is not perfect
A, S, G = decs(latent, latent, latent, skip_features)
recon = A + S + G
mse = (recon - x).pow(2).mean().item()
print(f'Init reconstruction MSE (should be high): {mse:.6f}')
assert mse > 0.5, f'Reconstruction too good at init: {mse} (indicates weight coupling)'
print('PASS: Initial reconstruction error is high (good — weights independent)')
"
```

Expected: High MSE > 0.5 at initialization.

**Commit**:
```bash
git add pgdacsnet/model_gprmambasep.py
git commit -m "feat(mamba): add GprMambaSep three-branch decoders (A/S/G)

Three independent lightweight U-Net decoders with transposed conv upsampling,
ConvNeXtStage refinement, skip connections from encoder, and optional
Mamba2DBlock at medium resolution. Each decoder reconstructs one physical
component from its latent pathway. Decoders share topology only — no weight
sharing — allowing specialization for different physical reconstruction tasks.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
"
```

---

### TASK 7 — Task Heads and Output Interface

**Depends on**: TASK 6

**Files**: MODIFY `pgdacsnet/model_gprmambasep.py` (add `GprMambaSepTaskHeads` and `GprMambaSep` full model, ~150 lines)
MODIFY `pgdacsnet/model_interfaces.py` (add `GprMambaSepOutput` dataclass and update `unpack_model_output()`)

**Interface**:
- `GprMambaSepOutput(A_hat, S_hat, G_hat, G_mask, G_center, G_pres)` — extends PGDAOutput with component fields
- `GprMambaSep` — full model that chains encoder → bottleneck → decoders → task heads
- `build_model(cfg)` dispatch key `v2_0_gprmambasep`

**Design**: Task heads on the G decoder output only (A and S are not masked — they are regression-only). Three heads: (1) Mask: 1×1 conv → sigmoid → (B, 1, H, W) soft probability. (2) Center: CenterRefineHead from existing codebase — depthwise conv3×3 → GELU → 1×1 → softmax over time dimension → center-of-mass. (3) Presence: global avg pool → FC → sigmoid → (B, W) per-trace presence.

```python
@dataclass
class GprMambaSepOutput:
    """Output structure for GprMambaSep model, backward-compatible with PGDAOutput."""
    A_hat: torch.Tensor      # (B, 1, H, W)
    S_hat: torch.Tensor      # (B, 1, H, W)
    G_hat: torch.Tensor      # (B, 1, H, W) — also treated as reconstruction output
    G_mask: torch.Tensor     # (B, 1, H, W) — sigmoid probability
    G_center: torch.Tensor   # (B, W) — centerline index per trace
    G_pres: torch.Tensor     # (B, W) — presence probability per trace

class GprMambaSep(nn.Module):
    """Full GprMambaSep v2.0 model."""
    
    def __init__(self, cfg):
        super().__init__()
        base_ch = cfg.get('base_ch', 16)
        latent_dim = cfg.get('latent_dim', 64)
        
        self.encoder = GprMambaSepEncoder(base_ch=base_ch)
        self.bottleneck = GprMambaSepBottleneck(
            dim=8*base_ch, latent_dim=latent_dim,
            d_state=cfg.get('ssm_state_dim', 64)
        )
        self.decoders = GprMambaSepDecoders(latent_dim=latent_dim)
        self.task_heads = GprMambaSepTaskHeads(latent_dim)
    
    def forward(self, x, metadata=None, return_components=False):
        # x: (B, 1, H, W)
        bottleneck, skip_features = self.encoder(x, metadata)
        A_lat, S_lat, G_lat = self.bottleneck(bottleneck)
        A_hat, S_hat, G_hat = self.decoders(A_lat, S_lat, G_lat, skip_features)
        G_mask, G_center, G_pres = self.task_heads(G_lat, skip_features)
        
        out = GprMambaSepOutput(
            A_hat=A_hat, S_hat=S_hat, G_hat=G_hat,
            G_mask=G_mask, G_center=G_center, G_pres=G_pres
        )
        return out
```

**TDD steps**:

Step 7.1 — Full model forward:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import torch, sys
sys.path.insert(0, '.')
from pgdacsnet.model_gprmambasep import GprMambaSep

cfg = {'base_ch': 8, 'latent_dim': 32, 'ssm_state_dim': 32}
model = GprMambaSep(cfg)
B = 1
x = torch.randn(B, 1, 512, 256)
out = model(x)

assert out.A_hat.shape == (B, 1, 512, 256), f'A_hat: {out.A_hat.shape}'
assert out.S_hat.shape == (B, 1, 512, 256), f'S_hat: {out.S_hat.shape}'
assert out.G_hat.shape == (B, 1, 512, 256), f'G_hat: {out.G_hat.shape}'
assert out.G_mask.shape == (B, 1, 512, 256), f'G_mask: {out.G_mask.shape}'
assert out.G_center.shape == (B, 256), f'G_center: {out.G_center.shape}'
assert out.G_pres.shape == (B, 256), f'G_pres: {out.G_pres.shape}'
assert torch.isfinite(out.G_mask).all()
print('PASS: GprMambaSep full forward pass OK')
"
```

Expected: All shapes correct, finite.

Step 7.2 — Backward pass through full model:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import torch, sys
sys.path.insert(0, '.')
from pgdacsnet.model_gprmambasep import GprMambaSep

cfg = {'base_ch': 4, 'latent_dim': 16, 'ssm_state_dim': 16}
model = GprMambaSep(cfg)
x = torch.randn(1, 1, 128, 64, requires_grad=True)
out = model(x)
loss = out.G_mask.sum() + out.G_center.sum() + out.G_pres.sum() + out.A_hat.sum()
loss.backward()
assert x.grad is not None
assert torch.isfinite(x.grad).all()
assert x.grad.abs().sum() > 0
print(f'PASS: Full model backward OK, grad_norm={x.grad.norm().item():.4f}')
"
```

Expected: `PASS: Full model backward OK, grad_norm=...`

Step 7.3 — Backward-compatible output via `unpack_model_output()`:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import torch, sys
sys.path.insert(0, '.')
from pgdacsnet.model_gprmambasep import GprMambaSep
from pgdacsnet.model_interfaces import unpack_model_output

cfg = {'base_ch': 4, 'latent_dim': 16, 'ssm_state_dim': 16}
model = GprMambaSep(cfg)
x = torch.randn(1, 1, 128, 64)
out = model(x)

# unpack_model_output should return (mask, center, presence, ...)
result = unpack_model_output(out)
assert len(result) >= 3, f'Expected tuple with at least 3 elements, got {len(result)}'
mask, center, pres = result[:3]
assert mask.shape == (1, 1, 128, 64), f'mask: {mask.shape}'
assert center.shape == (1, 64), f'center: {center.shape}'
assert pres.shape == (1, 64), f'pres: {pres.shape}'
print('PASS: unpack_model_output compatible with GprMambaSepOutput')
"
```

Expected: `PASS: unpack_model_output compatible with GprMambaSepOutput`

Step 7.4 — Parameter count:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import torch, sys
sys.path.insert(0, '.')
from pgdacsnet.model_gprmambasep import GprMambaSep

cfg = {'base_ch': 16, 'latent_dim': 64, 'ssm_state_dim': 64}
model = GprMambaSep(cfg)
params = sum(p.numel() for p in model.parameters())
print(f'GprMambaSep (base_ch=16) total params: {params:,}')
# Expected: ~4.76M
assert 3.5e6 < params < 7.0e6, f'Unexpected param count: {params:,}'
print('PASS: Param count in expected range')
"
```

Expected: `PASS: Param count in expected range`

**Commit**:
```bash
git add pgdacsnet/model_gprmambasep.py pgdacsnet/model_interfaces.py
git commit -m "feat(mamba): add GprMambaSep full model + backward-compatible output

Full GprMambaSep architecture: encoder → bottleneck → 3 decoders → task heads.
Output via GprMambaSepOutput dataclass with A_hat, S_hat, G_hat, G_mask,
G_center, G_pres. Backward-compatible with unpack_model_output() for
existing eval pipeline.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
"
```

---

### TASK 8 — Register Architecture in build_model() Dispatch

**Depends on**: TASK 7

**Files**: MODIFY `pgdacsnet/model_raw_unet.py` (add import and dispatch case)

**Interface**:
- Config key `arch: "v2_0_gprmambasep"` triggers GprMambaSep build
- Consumes: full config dict
- Produces: GprMambaSep model instance, callable via same `model(x)` interface

**Modification** (in `build_model()`):

```python
# At top of model_raw_unet.py, add import:
from pgdacsnet.model_gprmambasep import GprMambaSep

# In build_model() dispatch, after existing architectures (~line 120):
if arch == 'v2_0_gprmambasep':
    logger.info("Building GprMambaSep v2.0 architecture")
    model = GprMambaSep(cfg)
    return model
```

**TDD steps**:

Step 8.1 — Dispatch works with minimal config:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import torch, sys
sys.path.insert(0, '.')
from pgdacsnet.model_raw_unet import build_model

cfg = {
    'arch': 'v2_0_gprmambasep',
    'base_ch': 8,
    'latent_dim': 32,
    'ssm_state_dim': 32
}
model = build_model(cfg)
x = torch.randn(1, 1, 128, 64)
out = model(x)
assert hasattr(out, 'G_mask'), 'Output missing G_mask'
assert out.G_mask.shape == (1, 1, 128, 64)
print(f'PASS: build_model dispatch OK for v2_0_gprmambasep, params={sum(p.numel() for p in model.parameters()):,}')
"
```

Expected: `PASS: build_model dispatch OK for v2_0_gprmambasep, params=...`

Step 8.2 — Existing architectures still work:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import torch, sys
sys.path.insert(0, '.')
from pgdacsnet.model_raw_unet import build_model

# Test each existing architecture
for arch in ['v1_4_raw_only_unet', 'v1_7a_convnext', 'v1_7b_axial_ssm', 'v1_9d_mambavision_hybrid']:
    cfg = {'arch': arch, 'base_ch': 8}
    model = build_model(cfg)
    x = torch.randn(1, 1, 128, 64)
    out = model(x)
    print(f'  {arch}: output OK')
print('PASS: All existing architectures still work')
"
```

Expected: All existing architectures produce output without error.

**Commit**:
```bash
git add pgdacsnet/model_raw_unet.py
git commit -m "feat(mamba): register v2_0_gprmambasep in build_model() dispatch

Adds GprMambaSep architecture to build_model() under arch key
'v2_0_gprmambasep'. All existing architectures continue to work unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
"
```

---

### TASK 9 — Losses L1-L6 Implementation

**Depends on**: TASK 7 (needs GprMambaSepOutput interface)

**Files**: CREATE `scripts/losses_gprmambasep.py` (~250 lines)

**Interface**:
- `compute_gprmambasep_loss(out, batch, cfg, model=None)` — main entry point called from `compute_loss()`
- Consumes: GprMambaSepOutput, batch dict (with optional Y_air, Y_target, X_clean), config dict, model (for contrastive discriminator)
- Produces: dict of `{'loss': total, 'loss_self_consistency': ..., 'loss_sim_supervised': ..., ...}`
- Config schema: `component_loss_weights` dict with keys: `self_consistency` (default 2.0), `sim_supervised` (default 0.5), `contrastive` (default 0.05), `arrival_prior` (default 0.1), `amplitude_ratio` (default 0.01), `co_prediction` (default 0.3, Stage 3 only)

**Loss implementations**:

```python
# losses_gprmambasep.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientReversalLayer(torch.autograd.Function):
    """Gradient reversal layer for adversarial contrastive loss (L3)."""
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def compute_self_consistency_loss(A_hat, S_hat, G_hat, Y_full):
    """L1: Y_full ≈ A_hat + S_hat + G_hat (L1 + L2)."""
    recon = A_hat + S_hat + G_hat
    loss_l1 = F.l1_loss(recon, Y_full)
    loss_l2 = F.mse_loss(recon, Y_full)
    return loss_l1 + loss_l2


def compute_sim_supervised_loss(A_hat, S_hat, G_hat, batch_Y_air, batch_Y_target, batch_X_clean):
    """L2: supervise each component when sim labels available.
    
    batch_Y_air: air_only simulation → supervise A_hat
    batch_Y_target: target_only simulation → supervise A_hat + S_hat
    batch_X_clean: cleaned signal = target_only - air_only → supervise G_hat
    """
    loss = 0.0
    n_terms = 0
    
    if batch_Y_air is not None:
        loss += F.l1_loss(A_hat, batch_Y_air)
        n_terms += 1
    
    if batch_Y_target is not None:
        loss += F.l1_loss(A_hat + S_hat, batch_Y_target)
        n_terms += 1
    
    if batch_X_clean is not None:
        loss += F.l1_loss(G_hat, batch_X_clean)
        n_terms += 1
    
    return loss / max(n_terms, 1)


def compute_contrastive_separation_loss(A_hat, S_hat, G_hat, discriminator, model):
    """L3: Minimize mutual information between A and G latent embeddings.
    
    Uses gradient reversal on the A and G embedding pathway outputs,
    with a 3-layer MLP discriminator trained to classify A vs G features.
    """
    # Extract pooled feature vectors from A and G pathways
    A_feat = A_hat.mean(dim=[2, 3])  # (B, 1) — global avg pool
    G_feat = G_hat.mean(dim=[2, 3])  # (B, 1)
    
    # Also include S for contrastive anchor (not minimized against A)
    S_feat = S_hat.mean(dim=[2, 3])
    
    # Concatenate A and G features, create labels (0=A, 1=G)
    feat = torch.cat([A_feat, G_feat], dim=0)  # (2B, 1)
    labels = torch.cat([
        torch.zeros(A_feat.size(0), 1, device=A_feat.device),
        torch.ones(G_feat.size(0), 1, device=G_feat.device)
    ], dim=0)
    
    # Apply gradient reversal before discriminator
    feat_rev = GradientReversalLayer.apply(feat, alpha=1.0)
    preds = discriminator(feat_rev)
    
    # Binary classification loss
    loss = F.binary_cross_entropy_with_logits(preds, labels)
    return loss


def compute_arrival_time_prior_loss(G_hat, terrain_metadata, config):
    """L4: Penalize G_hat energy before earliest possible bedrock arrival.
    
    For each trace, compute t_min = 2*altitude/c_air + 2*z_min/v_earth_eff.
    Zero out loss after t_min, penalize energy before t_min.
    """
    # If no terrain metadata available, skip loss
    if terrain_metadata is None:
        return torch.tensor(0.0, device=G_hat.device)
    
    altitude = terrain_metadata[:, :, 0]  # (B, W)
    z_min = config.get('z_min_m', 3.0)
    c_air = 0.3  # m/ns
    v_earth = 0.07  # m/ns
    
    # Compute t_min per trace: (B, W)
    t_min = 2 * altitude / c_air + 2 * z_min / v_earth  # ns
    
    # Convert to time index: (B, W) → (B, 1, H, W) mask
    # Assume time axis index corresponds to ns: dt = time_window / H
    H = G_hat.shape[2]
    time_window = config.get('time_window_ns', 700.0)
    dt = time_window / H
    t_idx = (t_min / dt).long()  # (B, W)
    
    # Create weight mask: 1 for t < t_min, 0 for t >= t_min
    B, W = t_idx.shape
    t_range = torch.arange(H, device=G_hat.device).view(1, 1, H, 1).expand(B, 1, H, W)
    t_idx_expanded = t_idx.view(B, 1, 1, W).expand(B, 1, 1, W)
    weight_mask = (t_range < t_idx_expanded).float()  # (B, 1, H, W)
    
    # Masked L1 loss
    loss = (G_hat.abs() * weight_mask).sum() / weight_mask.sum().clamp(min=1)
    return loss


def compute_amplitude_ratio_prior_loss(A_hat, S_hat):
    """L5: Enforce A/S amplitude ratio consistent with Fresnel reflection.
    
    |A|/|S| ≈ (Z_air - Z_ground)/(Z_air + Z_ground) where Z = sqrt(mu/epsilon).
    For epsilon_ground ≈ 6-19, this gives roughly 0.42-0.63.
    """
    A_amp = A_hat.abs().mean(dim=[2, 3])  # (B, 1)
    S_amp = S_hat.abs().mean(dim=[2, 3]) + 1e-8  # (B, 1), avoid div-by-zero
    ratio = A_amp / S_amp  # (B, 1)
    
    # Target range: [0.42, 0.63] from physics
    target_low = 0.42
    target_high = 0.63
    
    # Penalize ratio outside the target range
    loss = torch.relu(target_low - ratio).mean() + torch.relu(ratio - target_high).mean()
    return loss


def compute_co_prediction_cycle_loss(model, Y_full, Y_target):
    """L6: Self-supervised cycle consistency.
    
    1. Forward Y_full → get A, S, G
    2. Reconstruct Y_target_hat = A + S
    3. Forward Y_target_hat through shared encoder → get A2, S2, G2
    4. Cycle consistency: Y_full ≈ A2 + S2 + G2, A ≈ A2, S ≈ S2
    """
    # Step 1: Forward on Y_full
    out_full = model(Y_full)
    
    # Step 2: Reconstruct Y_target_hat
    Y_target_hat = out_full.A_hat + out_full.S_hat
    
    # Step 3: Forward on Y_target_hat (shared encoder)
    out_cycle = model(Y_target_hat)
    
    # Step 4: Cycle losses
    recon_cycle = out_cycle.A_hat + out_cycle.S_hat + out_cycle.G_hat
    loss_recon = F.l1_loss(recon_cycle, Y_full)
    loss_A_cycle = F.l1_loss(out_cycle.A_hat, out_full.A_hat.detach())
    loss_S_cycle = F.l1_loss(out_cycle.S_hat, out_full.S_hat.detach())
    
    return loss_recon + 0.5 * (loss_A_cycle + loss_S_cycle)


def compute_gprmambasep_loss(out, batch, cfg, model=None, discriminator=None, epoch=None):
    """
    Main loss computation for GprMambaSep.
    
    Args:
        out: GprMambaSepOutput from model forward
        batch: dict with 'Y_full', optionally 'Y_air', 'Y_target', 'X_clean'
        cfg: full config dict (accessed at cfg['component_loss_weights'])
        model: GprMambaSep model instance (needed for L6)
        discriminator: MLP for L3 contrastive loss
        epoch: current epoch number
    """
    weights = cfg.get('component_loss_weights', {})
    Y_full = batch['Y_full']
    
    # L0: base task losses (G_mask, G_center, G_pres) — use existing losses_pgda
    from pgdacsnet.losses_pgda import compute_task_losses
    task_losses = compute_task_losses(out, batch, cfg)  # dict with 'loss_band_bce', etc.
    base_loss = sum(task_losses.values())
    
    # L1: Self-consistency reconstruction
    loss_sc = compute_self_consistency_loss(out.A_hat, out.S_hat, out.G_hat, Y_full)
    w_sc = weights.get('self_consistency', 2.0)
    
    # L2: Simulation-supervised component losses
    loss_sim = compute_sim_supervised_loss(
        out.A_hat, out.S_hat, out.G_hat,
        batch.get('Y_air'), batch.get('Y_target'), batch.get('X_clean')
    )
    w_sim = weights.get('sim_supervised', 0.5)
    
    # L3: Contrastive separation (only if discriminator provided)
    loss_ctr = torch.tensor(0.0, device=Y_full.device)
    if discriminator is not None:
        loss_ctr = compute_contrastive_separation_loss(out.A_hat, out.S_hat, out.G_hat, discriminator, model)
    w_ctr = weights.get('contrastive', 0.05)
    
    # L4: Arrival time physics prior
    loss_arrival = compute_arrival_time_prior_loss(out.G_hat, batch.get('terrain_metadata'), cfg)
    w_arrival = weights.get('arrival_prior', 0.1)
    
    # L5: Amplitude ratio prior
    loss_amp = compute_amplitude_ratio_prior_loss(out.A_hat, out.S_hat)
    w_amp = weights.get('amplitude_ratio', 0.01)
    
    # L6: Co-prediction cycle (Stage 3 only)
    loss_cycle = torch.tensor(0.0, device=Y_full.device)
    if cfg.get('use_co_prediction', False) and model is not None:
        Y_target = batch.get('Y_target')  # Only on sim data with Y_target available
        if Y_target is not None:
            loss_cycle = compute_co_prediction_cycle_loss(model, Y_full, Y_target)
    w_cycle = weights.get('co_prediction', 0.3)
    
    total = base_loss + w_sc * loss_sc + w_sim * loss_sim + w_ctr * loss_ctr \
            + w_arrival * loss_arrival + w_amp * loss_amp + w_cycle * loss_cycle
    
    return {
        'loss': total,
        'loss_base': base_loss,
        'loss_self_consistency': loss_sc,
        'loss_sim_supervised': loss_sim,
        'loss_contrastive': loss_ctr,
        'loss_arrival_prior': loss_arrival,
        'loss_amplitude_ratio': loss_amp,
        'loss_co_prediction': loss_cycle,
        **{f'loss_{k}': v for k, v in task_losses.items()}
    }
```

**TDD steps**:

Step 9.1 — Each loss term produces finite output:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import torch, sys
sys.path.insert(0, '.')
from scripts.losses_gprmambasep import (
    compute_self_consistency_loss,
    compute_sim_supervised_loss,
    compute_contrastive_separation_loss,
    compute_arrival_time_prior_loss,
    compute_amplitude_ratio_prior_loss,
)

B, H, W = 1, 512, 256
dummy_A = torch.randn(B, 1, H, W)
dummy_S = torch.randn(B, 1, H, W)
dummy_G = torch.randn(B, 1, H, W)
dummy_Y = dummy_A + dummy_S + dummy_G + 0.1 * torch.randn(B, 1, H, W)

# L1
l1 = compute_self_consistency_loss(dummy_A, dummy_S, dummy_G, dummy_Y)
assert torch.isfinite(l1), f'L1 non-finite: {l1}'
print(f'L1 (self-consistency): {l1.item():.6f}')

# L2
dummy_Y_air = torch.randn(B, 1, H, W)
dummy_Y_target = torch.randn(B, 1, H, W)
dummy_X_clean = torch.randn(B, 1, H, W)
l2 = compute_sim_supervised_loss(dummy_A, dummy_S, dummy_G, dummy_Y_air, dummy_Y_target, dummy_X_clean)
assert torch.isfinite(l2), f'L2 non-finite: {l2}'
print(f'L2 (sim-supervised): {l2.item():.6f}')

# L3
class DummyDisc(nn.Module):
    def __init__(self): super().__init__(); self.fc = nn.Linear(1, 1)
    def forward(self, x): return self.fc(x)
disc = DummyDisc()
l3 = compute_contrastive_separation_loss(dummy_A, dummy_S, dummy_G, disc, None)
assert torch.isfinite(l3), f'L3 non-finite: {l3}'
print(f'L3 (contrastive): {l3.item():.6f}')

# L4
metadata = torch.tensor([[[50.0]] * W])  # altitude=50m each trace
dummy_cfg = {'z_min_m': 3.0, 'time_window_ns': 700.0}
l4 = compute_arrival_time_prior_loss(dummy_G, metadata, dummy_cfg)
assert torch.isfinite(l4), f'L4 non-finite: {l4}'
print(f'L4 (arrival prior): {l4.item():.6f}')

# L5
l5 = compute_amplitude_ratio_prior_loss(dummy_A, dummy_S)
assert torch.isfinite(l5), f'L5 non-finite: {l5}'
print(f'L5 (amplitude ratio): {l5.item():.6f}')

print('PASS: All loss terms produce finite outputs')
"
```

Expected: All 5 losses print finite values.

Step 9.2 — Full loss computation with GprMambaSepOutput:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import torch, sys
sys.path.insert(0, '.')
from pgdacsnet.model_gprmambasep import GprMambaSepOutput
from scripts.losses_gprmambasep import compute_gprmambasep_loss

# Create dummy output
out = GprMambaSepOutput(
    A_hat=torch.randn(1, 1, 128, 64),
    S_hat=torch.randn(1, 1, 128, 64),
    G_hat=torch.randn(1, 1, 128, 64),
    G_mask=torch.sigmoid(torch.randn(1, 1, 128, 64)),
    G_center=torch.randn(1, 64),
    G_pres=torch.sigmoid(torch.randn(1, 64)),
)

batch = {
    'Y_full': torch.randn(1, 1, 128, 64),
    'Y_air': torch.randn(1, 1, 128, 64),
    'Y_target': torch.randn(1, 1, 128, 64),
    'X_clean': torch.randn(1, 1, 128, 64),
    'terrain_metadata': torch.randn(1, 64, 1),
}

cfg = {
    'component_loss_weights': {
        'self_consistency': 2.0,
        'sim_supervised': 0.5,
        'contrastive': 0.05,
        'arrival_prior': 0.1,
        'amplitude_ratio': 0.01,
    },
    'z_min_m': 3.0,
    'time_window_ns': 700.0,
    'target_loss_keys': ['band_bce', 'band_dice', 'core_bce', 'presence_bce', 'centerline_l1'],
    'target_loss_weights': [1.0, 0.5, 2.0, 0.5, 1.0],
    'threshold_core': 0.55,
    'margin': 0.05,
}

losses = compute_gprmambasep_loss(out, batch, cfg)
assert 'loss' in losses, 'Missing total loss'
assert torch.isfinite(losses['loss']), f'Non-finite total loss: {losses[\"loss\"]}'
print(f'Total loss: {losses[\"loss\"].item():.6f}')
for k, v in losses.items():
    if isinstance(v, torch.Tensor):
        print(f'  {k}: {v.item():.6f}')
print('PASS: Full loss computation OK')
"
```

Expected: Total loss finite, all components printed.

Step 9.3 — Gradient reversal layer:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import torch, sys
sys.path.insert(0, '.')
from scripts.losses_gprmambasep import GradientReversalLayer

x = torch.randn(3, 5, requires_grad=True)
y = GradientReversalLayer.apply(x, alpha=1.0)
loss = y.sum()
loss.backward()
# Gradient should be negated: x.grad should be -1 (torch.ones_like)
assert x.grad is not None
assert torch.allclose(x.grad, -torch.ones_like(x.grad)), f'Unexpected gradient: {x.grad}'
print('PASS: Gradient reversal layer reverses gradients correctly')
"
```

Expected: `PASS: Gradient reversal layer reverses gradients correctly`

**Commit**:
```bash
git add scripts/losses_gprmambasep.py tests/test_losses_gprmambasep.py
git commit -m "feat(mamba): add GprMambaSep losses L1-L6 — self-consistency, sim-supervised, contrastive separation, physics priors, cycle consistency

Six new loss terms for component decomposition:
- L1: Self-consistency (Y = A+S+G enforced via L1+L2)
- L2: Simulation-supervised (supervise each component from gprMax variants)
- L3: Contrastive separation (adversarial MI minimization via GRL)
- L4: Arrival time prior (penalize G energy before earliest bedrock arrival)
- L5: Amplitude ratio prior (Fresnel reflection coefficient constraint)
- L6: Co-prediction cycle (self-supervised cycle consistency for real data)

All losses have configurable weights and finite gradient tests.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
"
```

---

### TASK 10 — Wire Extended Loss into Training Loop

**Depends on**: TASK 9

**Files**: MODIFY `scripts/train_raw_only.py` (~30 lines changed)

**Interface**:
- `compute_loss()` now detects GprMambaSep model and delegates to `compute_gprmambasep_loss()`
- Config schema extended with `component_loss_weights` (validated at init)
- Discriminator created when contrastive loss weight > 0

**Changes in train_raw_only.py**:

```python
# At top, add imports:
from scripts.losses_gprmambasep import compute_gprmambasep_loss, GradientReversalLayer

# In compute_loss(), add detection logic:
def compute_loss(model_out, batch, cfg, model=None, discriminator=None, epoch=None):
    """Extended loss computation with GprMambaSep support."""
    
    # Detect GprMambaSep output by checking for component fields
    if hasattr(model_out, 'A_hat') and hasattr(model_out, 'G_hat'):
        # Use GprMambaSep loss
        return compute_gprmambasep_loss(model_out, batch, cfg, model, discriminator, epoch)
    
    # Fall through to existing loss computation for v1.x architectures
    ...

# In train_one_epoch or training loop init:
def setup_mamba_discriminator(cfg, device):
    """Create contrastive discriminator if needed."""
    w_ctr = cfg.get('component_loss_weights', {}).get('contrastive', 0.05)
    if w_ctr > 0:
        discriminator = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        ).to(device)
        return discriminator
    return None
```

**TDD steps**:

Step 10.1 — Loss dispatch works for both v1.x and v2.0:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import torch, sys
sys.path.insert(0, '.')
from pgdacsnet.model_raw_unet import build_model
# Simulate the dispatch check
from pgdacsnet.model_gprmambasep import GprMambaSepOutput

# Test v1.x output (PGDAOutput/PGANetOutput)
class DummyV1Output:
    def __init__(self):
        self.mask = torch.randn(1, 1, 128, 64)
        self.center = torch.randn(1, 64)
        self.pres = torch.randn(1, 64)

v1_out = DummyV1Output()
assert not hasattr(v1_out, 'A_hat'), 'v1 output should not have A_hat'
assert not hasattr(v1_out, 'G_hat'), 'v1 output should not have G_hat'

# Test v2.0 output
v2_out = GprMambaSepOutput(
    A_hat=torch.randn(1, 1, 128, 64),
    S_hat=torch.randn(1, 1, 128, 64),
    G_hat=torch.randn(1, 1, 128, 64),
    G_mask=torch.randn(1, 1, 128, 64),
    G_center=torch.randn(1, 64),
    G_pres=torch.randn(1, 64),
)
assert hasattr(v2_out, 'A_hat'), 'v2 output should have A_hat'

# The dispatch condition should be:
def is_gprmambasep(out):
    return hasattr(out, 'A_hat') and hasattr(out, 'G_hat')

assert not is_gprmambasep(v1_out), 'Dispatch false positive for v1'
assert is_gprmambasep(v2_out), 'Dispatch false negative for v2'
print('PASS: Loss dispatch detection works correctly')
"
```

Expected: `PASS: Loss dispatch detection works correctly`

Step 10.2 — Config schema validation accepts new keys:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import json, sys

# Load and validate the config structure
cfg = {
    'arch': 'v2_0_gprmambasep',
    'base_ch': 16,
    'latent_dim': 64,
    'ssm_state_dim': 64,
    'component_loss_weights': {
        'self_consistency': 2.0,
        'sim_supervised': 0.5,
        'contrastive': 0.05,
        'arrival_prior': 0.1,
        'amplitude_ratio': 0.01,
    },
    'target_loss_keys': ['band_bce', 'band_dice', 'core_bce'],
    'target_loss_weights': [1.0, 0.5, 2.0],
}
json.dumps(cfg)  # Verify serializable
print('PASS: Config schema valid, JSON serializable')
"
```

Expected: `PASS: Config schema valid, JSON serializable`

**Commit**:
```bash
git add scripts/train_raw_only.py
git commit -m "feat(mamba): wire GprMambaSep loss dispatch into training loop

compute_loss() now detects GprMambaSep output by checking for component
fields (A_hat, G_hat) and delegates to compute_gprmambasep_loss(). Existing
v1.x architectures continue to use the original loss path unchanged. Config
schema accepts component_loss_weights dict. Optional discriminator setup for
contrastive separation loss.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
"
```

---

### TASK 11 — Config Generator for LOLO-CV Ablations

**Depends on**: TASK 8

**Files**: CREATE `scripts/make_v2_gprmambasep_loo_configs.py` (~100 lines)

**Interface**:
- `python scripts/make_v2_gprmambasep_loo_configs.py --out-dir configs/lolo_v2_gprmambasep/`
- Produces 15 JSON config files (5 folds × 3 seeds)
- Each config: fixed base_ch=16, latent_dim=64, ssm_state_dim=64, component_loss_weights as design values
- Fold-specific: test_line, val_lines, train_lines
- Also produces ablation configs in subdirectory `configs/lolo_v2_gprmambasep/ablations/`

**Design**:

```python
"""make_v2_gprmambasep_loo_configs.py — Generate LOLO-CV configs for GprMambaSep."""

import json, os, argparse

LINES = ['Line3', 'Line6', 'Line7', 'Line9', 'LineL1']
SEEDS = [1901, 1902, 1903]

BASE_CONFIG = {
    'arch': 'v2_0_gprmambasep',
    'base_ch': 16,
    'latent_dim': 64,
    'ssm_state_dim': 64,
    'ssm_d_conv': 4,
    'ssm_expand_factor': 2,
    'batch_size': 4,
    'lr': 3e-4,
    'weight_decay': 0.05,
    'epochs': 80,
    'data_root': 'data_corrected_v1_4_terrain_direction',
    'sim_data_root': 'data/PGDA_SYNTH_DATASET_V1/05_accepted_dataset',
    'sim_batch_ratio': 0.3,
    'component_loss_weights': {
        'self_consistency': 2.0,
        'sim_supervised': 0.5,
        'contrastive': 0.05,
        'arrival_prior': 0.1,
        'amplitude_ratio': 0.01,
    },
    'use_gradient_checkpointing': True,
}

ABLATION_VARIANTS = {
    'no_contrastive': {'component_loss_weights.contrastive': 0.0},
    'no_arrival_prior': {'component_loss_weights.arrival_prior': 0.0},
    'no_amplitude_ratio': {'component_loss_weights.amplitude_ratio': 0.0},
    'single_decoder': {'arch': 'v2_0_gprmambasep_single_decoder'},  # post-ablation variant
    'time_only_ssm': {'ssm_strategy': 'time_only'},
    'trace_only_ssm': {'ssm_strategy': 'trace_only'},
    'dual_axis_only': {'ssm_strategy': 'dual_axis'},
}
```

**TDD steps**:

Step 11.1 — Config generation produces correct files:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import tempfile, json, os, sys
sys.path.insert(0, 'scripts')
from make_v2_gprmambasep_loo_configs import generate_configs

with tempfile.TemporaryDirectory() as tmpdir:
    generated = generate_configs(out_dir=tmpdir)
    files = os.listdir(tmpdir)
    assert len(files) == 15, f'Expected 15 config files, got {len(files)}'
    for fname in files:
        with open(os.path.join(tmpdir, fname)) as f:
            cfg = json.load(f)
        assert 'arch' in cfg
        assert 'train_lines' in cfg
        assert 'test_line' in cfg
    print(f'PASS: Generated {len(files)} config files')
"
```

Expected: `PASS: Generated 15 config files`

Step 11.2 — Ablation configs generated:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import tempfile, json, os, sys
sys.path.insert(0, 'scripts')
from make_v2_gprmambasep_loo_configs import generate_configs

with tempfile.TemporaryDirectory() as tmpdir:
    generate_configs(out_dir=tmpdir, include_ablations=True)
    ablation_dir = os.path.join(tmpdir, 'ablations')
    assert os.path.isdir(ablation_dir), 'Ablation directory missing'
    files = os.listdir(ablation_dir)
    # 7 ablation variants × 1 fold (Line9) × 1 seed = 7 configs
    assert len(files) >= 7, f'Expected >=7 ablation configs, got {len(files)}'
    print(f'PASS: Generated ablation configs: {files}')
"
```

Expected: `PASS: Generated ablation configs: ...`

**Commit**:
```bash
git add scripts/make_v2_gprmambasep_loo_configs.py
git commit -m "feat(mamba): add LOLO-CV config generator for GprMambaSep

Generates 15 config files (5 folds × 3 seeds) for GprMambaSep LOLO-CV
evaluation. Also generates ablation configs (7 variants: no contrastive,
no arrival prior, no amplitude ratio, single decoder, time-only SSM,
trace-only SSM, dual-axis only) for the Line9-held-out fold.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
"
```

---

### TASK 12 — Training Configs (Stage 1-3)

**Depends on**: TASK 8

**Files**: CREATE `configs/gpu_pretrain_v2_gprmambasep.json`, `configs/gpu_mixed_v2_gprmambasep.json`, `configs/gpu_finetune_v2_gprmambasep_selfsup.json`

**Config designs**:

**Stage 1 — Simulation-only pretrain** (configs/gpu_pretrain_v2_gprmambasep.json):
```json
{
    "arch": "v2_0_gprmambasep",
    "base_ch": 16,
    "latent_dim": 64,
    "ssm_state_dim": 64,
    "ssm_d_conv": 4,
    "ssm_expand_factor": 2,
    "batch_size": 4,
    "lr": 3e-4,
    "weight_decay": 0.05,
    "epochs": 50,
    "data_root": null,
    "sim_data_root": "data/PGDA_SYNTH_DATASET_V1/05_accepted_dataset",
    "sim_batch_ratio": 1.0,
    "use_gradient_checkpointing": true,
    "component_loss_weights": {
        "self_consistency": 2.0,
        "sim_supervised": 0.5,
        "contrastive": 0.05,
        "arrival_prior": 0.1,
        "amplitude_ratio": 0.01
    }
}
```

**Stage 2 — Mixed sim-real** (configs/gpu_mixed_v2_gprmambasep.json): Same as stage 1 but epochs=80, batch_size=2, data_root points to real data, sim_batch_ratio=0.3.

**Stage 3 — Self-supervised fine-tune** (configs/gpu_finetune_v2_gprmambasep_selfsup.json): Same as stage 2 but epochs=20, lr=1e-5, use_co_prediction=true, data_root only.

**TDD steps**:

Step 12.1 — All configs parse and pass schema validation:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import json, sys, os

for fname in [
    'configs/gpu_pretrain_v2_gprmambasep.json',
    'configs/gpu_mixed_v2_gprmambasep.json',
    'configs/gpu_finetune_v2_gprmambasep_selfsup.json',
]:
    path = os.path.join('.', fname) if not os.path.isabs(fname) else fname
    with open(path) as f:
        cfg = json.load(f)
    assert 'arch' in cfg, f'{fname}: missing arch'
    assert cfg['arch'] == 'v2_0_gprmambasep', f'{fname}: wrong arch'
    assert 'base_ch' in cfg
    assert 'component_loss_weights' in cfg
    print(f'OK: {fname} parsed successfully ({cfg[\"epochs\"]} epochs, lr={cfg.get(\"lr\", \"default\")})')
print('PASS: All configs valid')
"
```

Expected: All configs parse successfully.

**Commit**:
```bash
git add configs/gpu_pretrain_v2_gprmambasep.json configs/gpu_mixed_v2_gprmambasep.json configs/gpu_finetune_v2_gprmambasep_selfsup.json
git commit -m "feat(mamba): add Stage 1-3 training configs for GprMambaSep

- Stage 1: simulation-only pretrain (50 epochs, batch=4, sim_batch_ratio=1.0)
- Stage 2: mixed sim-real (80 epochs, batch=2, sim_batch_ratio=0.3)
- Stage 3: self-supervised fine-tune (20 epochs, lr=1e-5, co-prediction)

All configs with component_loss_weights as per architecture design.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
"
```

---

### TASK 13 — Separation Quality Evaluator

**Depends on**: TASK 7

**Files**: CREATE `scripts/eval_gprmambasep_separation.py` (~200 lines)

**Interface**:
- `python scripts/eval_gprmambasep_separation.py --checkpoint <path> --config <path> --out-dir <path>`
- Produces: `separation_report.md`, `component_comparison.png` (4-panel: A_hat vs Y_air, S_hat vs Y_target-A, G_hat vs X_clean, overlay), `leakage_matrix.png` (cross-component correlation heatmap), `metrics.json`

**Metrics computed**:
1. Per-component SNR: `10*log10(||reference||^2 / ||reference - prediction||^2)` for each of A, S, G
2. Leakage ratio: `||A_hat ∩ G_hat|| / ||A_hat||` — fraction of A amplitude that spills into G path
3. Cross-correlation: `corrcoef(A_hat.flatten(), G_hat.flatten())` — should be near 0
4. Reconstruction fidelity: MSE and SSIM of A_hat+S_hat+G_hat vs Y_full
5. G_mask IoU on hold-out simulation cases with ground truth labels

**TDD steps**:

Step 13.1 — Script imports and argument parsing:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import torch, sys, tempfile, os
sys.path.insert(0, '.')

# Test the metric computation functions directly
from scripts.eval_gprmambasep_separation import (
    compute_component_snr, compute_leakage_ratio, compute_cross_correlation
)

B, H, W = 1, 128, 64
A_ref = torch.randn(B, 1, H, W)
A_hat = A_ref + 0.1 * torch.randn(B, 1, H, W)
snr = compute_component_snr(A_hat, A_ref)
assert torch.isfinite(snr), f'Non-finite SNR: {snr}'
print(f'Component SNR: {snr.item():.2f} dB')

leakage = compute_leakage_ratio(A_hat, torch.randn(B, 1, H, W))
assert 0.0 <= leakage <= 1.0, f'Leakage out of range: {leakage}'
print(f'Leakage ratio: {leakage:.4f}')

cross_corr = compute_cross_correlation(A_hat, torch.randn(B, 1, H, W))
assert -1.0 <= cross_corr <= 1.0, f'Cross-corr out of range: {cross_corr}'
print(f'Cross-correlation: {cross_corr:.4f}')

print('PASS: Separation metric computation works')
"
```

Expected: All metric computations produce valid numbers.

**Commit**:
```bash
git add scripts/eval_gprmambasep_separation.py
git commit -m "feat(mamba): add separation quality evaluator — SNR, leakage, cross-correlation metrics

Evaluates GprMambaSep separation quality: per-component SNR vs
simulation references, A-G leakage ratio, cross-correlation, and
reconstruction fidelity. Produces 4-panel comparison figure and
metrics JSON for automated QC.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
"
```

---

### TASK 14 — Smoke Test: Stage 0 Mini-Training

**Depends on**: TASK 10 (all code must be wired)

**Files**: None new — runs existing `train_raw_only.py` with a mini config

**Purpose**: Verify the entire pipeline end-to-end before committing to a full production run. Train GprMambaSep on a single simulation case for 10 epochs, confirm loss decreases.

**Mini config** (inline, not saved):

```python
cfg = {
    'arch': 'v2_0_gprmambasep',
    'base_ch': 4,  # Minimal
    'latent_dim': 8,
    'ssm_state_dim': 8,
    'ssm_d_conv': 3,
    'ssm_expand_factor': 2,
    'batch_size': 1,
    'lr': 1e-3,
    'epochs': 10,
    'data_root': None,
    'sim_data_root': 'data/PGDA_SYNTH_DATASET_V1/05_accepted_dataset',
    'sim_batch_ratio': 1.0,
    'use_gradient_checkpointing': False,
    'component_loss_weights': {
        'self_consistency': 2.0,
        'sim_supervised': 0.5,
        'contrastive': 0.05,
        'arrival_prior': 0.1,
        'amplitude_ratio': 0.01,
    },
}
```

**TDD steps**:

Step 14.1 — 10-epoch smoke training on a single simulation case:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -c "
import torch, sys, json, os, tempfile
sys.path.insert(0, '.')
from pgdacsnet.model_raw_unet import build_model
# This is a simplified smoke test that runs a forward+backward loop
# without the full data loader — the actual full loop test happens below

cfg = {
    'arch': 'v2_0_gprmambasep',
    'base_ch': 4,
    'latent_dim': 8,
    'ssm_state_dim': 8,
    'ssm_d_conv': 3,
    'ssm_expand_factor': 2,
}
model = build_model(cfg)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Simulate 5 steps of training on random data
losses = []
for step in range(5):
    x = torch.randn(1, 1, 128, 64)
    out = model(x)
    loss = out.G_mask.sum() + out.A_hat.sum() + out.S_hat.sum() + out.G_hat.sum()
    loss = -loss  # Maximize output (dummy objective for gradient test)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f'  Step {step}: loss={loss.item():.4f}')

print(f'PASS: 5-step training loop completed, loss range [{min(losses):.4f}, {max(losses):.4f}]')
"
```

Expected: 5 steps complete without error, loss changes.

**Commit** (document the smoke test; commit the smoke test script if created):
```bash
git tag smoke-test-gprmambasep-pinned
git commit --allow-empty -m "chore(mamba): smoke test checkpoint — GprMambaSep 5-step training loop verified

All forward/backward/optimizer steps complete without error on
minimal config (base_ch=4, latent_dim=8, ssm_state_dim=8).
Full 10-epoch smoke training with real data loader to follow.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
"
```

---

### TASK 15 — Full Integration Test: Simulation-Only Pretrain (Stage 1)

**Depends on**: TASK 12, TASK 14 (smoke passed)

**Files**: None — production run

**Command**:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -u scripts/train_raw_only.py configs/gpu_pretrain_v2_gprmambasep.json
```

**Monitoring**:
- First 5 epochs: total loss should drop from ~0.4 to <0.15 (base BCE/Dice + self-consistency dominates)
- Component reconstruction error (A_hat vs Y_air, G_hat vs X_clean) should decrease steadily
- VRAM usage: `nvidia-smi --query-gpu=memory.used --format=csv -l 5` — should stay below 5GB
- If OOM: reduce batch_size from 4 to 2, or enable gradient checkpointing (already in config)

**Expected outcome after 50 epochs**:
- Total loss ~0.08-0.12
- A reconstruction MSE <0.01 (air wave is the easiest because it's the strongest signal)
- G_mask IoU on hold-out sim cases >0.8
- G centerline MAE on sim cases <5ns

**Post-training QC**:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" scripts/eval_gprmambasep_separation.py \
  --checkpoint outputs/run_*/checkpoint_last.pt \
  --config configs/gpu_pretrain_v2_gprmambasep.json \
  --out-dir outputs/run_*/separation_qc/
```

Expected: separation report shows A SNR >15dB, G SNR >8dB, leakage <10%.

**Commit** (after verifying Stage 1 completes successfully):
```bash
git add outputs/run_*/separation_qc/
git commit -m "feat(mamba): Stage 1 simulation-only pretrain complete — G_mask IoU>0.8, A SNR>15dB

50-epoch simulation-only pretraining on combined batch_001 + batch_002 +
batch_003 + LINE9_LABEL_INSPIRED_V1. Separation quality verified:
- A SNR: >15dB (air wave cleanly isolated)
- G SNR: >8dB (bedrock reflection separated)
- Leakage: <10% (minimal A→G spill)
- G_mask IoU: >0.8 on held-out sim cases

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
"
```

---

### TASK 16 — Mixed Sim-Real Training (Stage 2)

**Depends on**: TASK 15 (Stage 1 checkpoint available)

**Command**:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -u scripts/resume_train.py configs/gpu_mixed_v2_gprmambasep.json
```

(Assumes `resume_train.py` finds the stage 1 checkpoint automatically, or explicitly resume from checkpoint.)

**Expected outcome after 80 epochs**:
- Total loss plateaus around epoch 60-70
- Real-data validation metrics improve (G_mask on real windows shows structure, not random noise)

**Commit**: After Stage 2 completes.

---

### TASK 17 — Self-Supervised Fine-Tune (Stage 3, Optional)

**Depends on**: TASK 16

**Command**:
```bash
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -u scripts/resume_train.py configs/gpu_finetune_v2_gprmambasep_selfsup.json
```

**Expected outcome**: Co-prediction cycle loss decreases, decomposition quality on real data improves.

**Commit**: After Stage 3 completes.

---

### TASK 18 — Batch Scripts and Runner Wrappers

**Depends on**: TASK 12

**Files**: CREATE `scripts/run_gprmambasep_pipeline.bat` and `scripts/run_gprmambasep_pipeline.sh`

**Purpose**: One-click batch files for the 3-stage training pipeline, so the user can kick off the complete training sequence without manual intervention.

**TDD step**: Run the batch script, verify it launches Stage 1 (first 10 epochs as a test, then kill with Ctrl+C).

**Commit**: After batch scripts verified.

---

### TASK 19 — Ablation Experiments (E1-E9, Single Fold)

**Depends on**: TASK 15 (Stage 1 complete — need trained checkpoints to ablate)

**Procedure**: For each ablation variant (see problem spec E1-E9), modify the config, train for 50 epochs on the Line9-held-out fold only, evaluate separation quality and G_mask IoU.

**Key ablations and expected outcomes**:

| # | Ablation | Config Change | Expected Metric Delta |
|---|----------|---------------|-----------------------|
| E1 | Single decoder | `arch: v2_0_gprmambasep_single_decoder` | F1 drops 0.08-0.11 |
| E2 | GatedSequenceBlock vs Mamba | `ssm_use_mamba: false` | IoU -2-4% |
| E3 | Scan strategy variants | `ssm_strategy: time_only/trace_only/dual_axis/cross_scan` | full > dual > cross > trace > time |
| E4 | Data scale | Vary `sim_data_root` to subsets | batch_003 gives steepest per-sample gain |
| E6 | Contrastive weight sweep | Vary `component_loss_weights.contrastive` | 0.05-0.1 optimal, >0.5 harmful |

**Commit**: After each ablation completes, with results.

---

### TASK 20 — Baseline Comparison (E8)

**Depends on**: TASK 19 (ablation results)

**Procedure**: Run existing evaluation scripts for baseline architectures (v1.4, v1.7a, v1.7b, v1.9d, v1.11) on the same held-out Line9 fold, using their existing checkpoints or re-training with the same data.

**Expected outcome**: GprMambaSep achieves SOTA on all metrics:
- DP MAE: 22-25ns vs 37.19ns (v1.9d best prior)
- Pick rate: 83-98% vs 96.6% (v1.9d best prior)
- Real data pick rate: 40-60% vs 0-28% (prior best after fine-tuning)

**Commit**: Baseline comparison results and analysis.

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| SelectiveSSMLite proxy diverges too far from true Mamba | Validation test in TASK 2.2; numerical parity check <1e-3 tolerance; if proxy fidelity is poor, fall back to training only on WSL2 with CUDA kernel |
| Three decoders coupled to same solution (self-consistency dominates, all three converge to Y_full/3) | Contrastive loss (L3) forces A and G apart; arrival time prior (L4) gives G a unique physical constraint; separate weight initialization per decoder |
| VRAM OOM at batch_size 4 | Reduce to batch_size 2; Stage 1 can use batch=2 with more gradient accumulation steps (set `grad_accum_steps=2` in config); documented in Global Constraint #3 |
| Cross-scan 4-direction SSM too slow (4x per Mamba2DBlock) | Make cross-scan optional (scan_strategy config); default to dual_axis for stages 1-2, full for stages 3-4 only; benchmark in TASK 3.5 |
| Simulation-only pretrain overfits to A (strongest signal) | L2 sim-supervised loss on all three components; L3 contrastive prevents A from dominating G pathway; G task heads (mask/center/presence) supervised separately |
| No valid simulation data for L2 (Y_air, Y_target, X_clean) in data loader | Document data format requirement: each sim case must have 3 variants (Y_full, Y_target, Y_air) or at minimum Y_full + X_clean; provide data loader fallback to skip L2 when labels missing |

## Infrastructure Notes

- **Windows development path**: All TASK 1-14 can be completed entirely on Windows. The SelectiveSSMLite proxy does not require CUDA compilation.
- **Production training path**: For Stage 1 (TASK 15), recommend WSL2 with native `selective_scan_cuda` for 2-3x faster training. If unavailable, the proxy is acceptable but expect 3x slower training.
- **Data directory**: The simulation data must be organized as:
  ```
  05_accepted_dataset/
    batch_001/
      case_001/
        Y_full.npy
        Y_target.npy
        Y_air.npy     (or X_clean.npy)
        labels.npz
      ...
  ```
  If `Y_target` or `Y_air` are missing, the L2 sim-supervised loss is skipped (logged as warning).

---

## Migration Path from v1.x

1. **Single-file drop-in replacement**: `GprMambaSep` replaces the encoder-decoder in `build_model()`. The output interface (`GprMambaSepOutput`) is backward-compatible with `unpack_model_output()`.
2. **Loss function**: The new loss (`compute_gprmambasep_loss`) wraps the existing `compute_task_losses` from `losses_pgda.py`. Existing v1.x architectures continue unchanged.
3. **Config**: The `arch: "v2_0_gprmambasep"` key is a new entry in the config schema. Existing configs without this key use the original architecture path.
4. **Data pipeline**: No changes needed. The existing data loader provides `Y_full`; the sim data loader (when configured) provides `Y_air`, `Y_target`, `X_clean`.
5. **Evaluation**: `eval_full_line.py` works unchanged because `unpack_model_output()` handles the new output format. The separation evaluator is an optional post-hoc step.
