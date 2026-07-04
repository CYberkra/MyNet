"""
pgdacsnet/model_mamba.py

Mamba-2/SSD components for GprMambaSep:

- SelectiveSSMLite: Pure-PyTorch proxy with input-dependent depthwise conv1d
  modulation that approximates Mamba-2's selection mechanism.
- SelectiveSSMCUDA: Thin wrapper around VMamba's selective_scan_cuda kernel
  with graceful ImportError fallback on Windows.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["SelectiveSSMLite", "SelectiveSSMCUDA", "Mamba2DBlock"]


class SelectiveSSMLite(nn.Module):
    """
    Pure-PyTorch approximation of Mamba-2 SSD selection mechanism.

    Replaces the continuous-time SSM discretization with an input-dependent
    depthwise conv1d where the kernel at position t is modulated by the
    input activation. This captures the essential 'selection' property:
    different tokens get different processing.

    Compared to GatedSequenceBlock (static kernel):
    - Same O(L) complexity with conv1d
    - Content-dependent modulation adds conv_kernel * d_model extra params
    - Acts as a learnable proxy for Mamba-2's A/B/C selection

    Args:
        d_model: Model dimension (input/output channels).
        d_state: SSM state dimension (for interface compatibility with Mamba-2).
        d_conv: Depthwise convolution kernel size.
        expand: Channel expansion factor.
        dt_rank: Delta projection rank (for interface compatibility).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dt_rank = dt_rank
        self.d_inner = int(expand * d_model)

        # 1. Input projection: d_model -> 2 * d_inner (split into signal + gate)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 2. Depthwise conv1d for local context (Mamba-style)
        #    Causal: only look at current + past d_conv-1 positions
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            bias=False,
        )

        # 3. Modulation projection: d_inner -> d_conv
        #    Generates kernel offsets from the gate-branch features
        self.modulation_proj = nn.Linear(self.d_inner, d_conv, bias=False)

        # 4. Base kernel: per-channel learnable base weights, shape (d_inner, d_conv)
        self.base_kernel = nn.Parameter(torch.randn(self.d_inner, d_conv) * 0.02)

        # 5. Output projection: d_inner -> d_model
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with content-adaptive processing.

        Args:
            x: Input tensor of shape (B, C, L) where
                B = batch, C = channels (== d_model), L = sequence length.

        Returns:
            Output tensor of shape (B, C, L) with same dimensions as input.
        """
        B, C, L = x.shape
        assert C == self.d_model, f"SelectiveSSMLite: expected C={self.d_model}, got C={C}"

        # 1. Expand channels: (B, C, L) -> (B, L, 2*d_inner) -> (B, 2*d_inner, L)
        x_proj = self.in_proj(x.transpose(1, 2)).transpose(1, 2)

        # Split: first half = signal path, second half = gate path
        signal, gate_raw = x_proj.chunk(2, dim=1)           # both (B, d_inner, L)

        # 2. Depthwise conv1d on signal branch (causal padding)
        signal_pad = F.pad(signal, (self.d_conv - 1, 0))    # (B, d_inner, L + d_conv - 1)
        signal_conv = self.conv1d(signal_pad)                # (B, d_inner, L)

        # 3. Generate kernel modulation from gate branch
        #    modulation: (B, L, d_conv) -> (B, d_conv, L)
        modulation = self.modulation_proj(gate_raw.transpose(1, 2))
        modulation = modulation.transpose(1, 2)              # (B, d_conv, L)

        # 4. Apply modulated depthwise conv (content-adaptive selection)
        #    Unfold signal_conv into sliding windows
        x_pad = F.pad(signal_conv, (self.d_conv - 1, 0))    # (B, d_inner, L + d_conv - 1)
        x_unfold = x_pad.unfold(2, self.d_conv, 1)           # (B, d_inner, L, d_conv)

        # Build position-dependent kernel
        # base_kernel: (d_inner, d_conv)
        # modulation:  (B, d_conv, L)
        # kernel[b, c, k, t] = base_kernel[c, k] + modulation[b, k, t]
        # Shapes: (1, d_inner, d_conv, 1) + (B, 1, d_conv, L) -> (B, d_inner, d_conv, L)
        kernel = self.base_kernel[None, :, :, None] + modulation[:, None, :, :]

        # Weighted sum: (B, d_inner, L, d_conv) x (B, d_inner, L, d_conv) sum -> (B, d_inner, L)
        signal_modulated = (x_unfold * kernel.transpose(2, 3)).sum(dim=-1)

        # 5. Gate with sigmoid
        gate_val = torch.sigmoid(gate_raw)
        out_gated = signal_modulated * gate_val              # (B, d_inner, L)

        # 6. Project back to d_model
        out = self.out_proj(out_gated.transpose(1, 2))       # (B, L, d_model)
        out = out.transpose(1, 2)                             # (B, d_model, L)

        return out


class SelectiveSSMCUDA(nn.Module):
    """
    Thin wrapper around VMamba's selective_scan_cuda kernel.

    When available (WSL2/Linux with CUDA), delegates to the highly optimized
    fused kernel for exact Mamba-2 SSD computation.

    On Windows, raises a clear ImportError directing the user to use
    SelectiveSSMLite instead.

    Args:
        d_model: Model dimension (input/output channels).
        d_state: SSM state dimension.
        d_conv: Depthwise convolution kernel size.
        expand: Channel expansion factor.
        dt_rank: Delta projection rank.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int = 8,
    ):
        super().__init__()

        # Attempt to import the CUDA kernel — on Windows this raises ImportError
        try:
            from selective_scan import selective_scan_cuda as _scan_fn  # type: ignore[import-untyped]
            self._scan_fn = _scan_fn
        except ImportError:
            raise ImportError(
                "SelectiveSSMCUDA requires the VMamba 'selective_scan_cuda' kernel, "
                "which is only available on WSL2/Linux with CUDA.\n"
                "On Windows, use SelectiveSSMLite instead:\n"
                "  from pgdacsnet.model_mamba import SelectiveSSMLite"
            )

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dt_rank = dt_rank
        self.d_inner = int(expand * d_model)

        # Standard Mamba-2 parameter layout
        # Input projection: d_model -> 2 * d_inner
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Depthwise conv1d for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # SSM discretisation parameters
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner + d_state, bias=False)
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using selective_scan_cuda (WSL2/Linux only).

        Args:
            x: Input tensor of shape (B, C, L).

        Returns:
            Output tensor of shape (B, C, L).
        """
        B, C, L = x.shape
        device = x.device

        # 1. Input projection and split
        x_proj = self.in_proj(x.transpose(1, 2)).transpose(1, 2)
        x_res, gate_raw = x_proj.chunk(2, dim=1)

        # 2. Depthwise conv1d (causal)
        x_pad = F.pad(x_res, (self.d_conv - 1, 0))
        x_conv = self.conv1d(x_pad)

        # 3. Discretisation: delta, A, B, C
        dt_params = self.dt_proj(x_conv.transpose(1, 2))   # (B, L, d_inner + d_state)
        delta = dt_params[..., : self.d_inner]               # (B, L, d_inner)
        B_param = dt_params[..., self.d_inner :]             # (B, L, d_state)
        delta = F.softplus(delta)
        A = -torch.exp(self.A_log)                           # (d_inner, d_state)
        C_param = B_param.clone()                            # simplified; real impl uses separate proj

        # 4. Call the fused CUDA kernel
        # selective_scan_cuda(u, delta, A, B, C, D, delta_bias, ...)
        y = self._scan_fn(
            u=x_conv,
            delta=delta,
            A=A,
            B=B_param,
            C=C_param,
            D=self.D,
            delta_softplus=True,
        )

        # 5. Gate and project back
        out = y * torch.sigmoid(gate_raw)
        out = self.out_proj(out.transpose(1, 2)).transpose(1, 2)
        return out


class Mamba2DBlock(nn.Module):
    """2D spatial SSM block powered by SelectiveSSMLite.

    Applies the content-adaptive SelectiveSSMLite independently along the
    horizontal (W) and vertical (H) axes of a 2D feature map, then merges
    with a learned 1x1 mix plus residual connection.

    Architecture mirrors AxialSSMLiteBlock but replaces GatedSequenceBlock
    with the more expressive SelectiveSSMLite (input-dependent modulation).

    Args:
        channels: Number of input/output channels.
        d_state: SSM state dimension (passed through to SelectiveSSMLite).
        d_conv: Depthwise convolution kernel in SelectiveSSMLite.
        expand: Channel expansion factor for SelectiveSSMLite.
    """

    def __init__(
        self,
        channels: int,
        d_state: int = 32,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.norm_h = nn.GroupNorm(1, channels)
        self.norm_v = nn.GroupNorm(1, channels)
        self.ssm_h = SelectiveSSMLite(
            d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand,
        )
        self.ssm_v = SelectiveSSMLite(
            d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand,
        )
        self.mix = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(1, channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applies SSM along both spatial axes.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, C, H, W) with same dimensions.
        """
        B, C, H, W = x.shape

        # Horizontal SSM: treat each row as a sequence of length W
        # (B, C, H, W) -> (B*H, C, W) -> SSM -> (B*H, C, W) -> (B, C, H, W)
        hx = self.norm_h(x).permute(0, 2, 1, 3).reshape(B * H, C, W)
        hx = self.ssm_h(hx).reshape(B, H, C, W).permute(0, 2, 1, 3)

        # Vertical SSM: treat each column as a sequence of length H
        # (B, C, H, W) -> (B*W, C, H) -> SSM -> (B*W, C, H) -> (B, C, H, W)
        vx = self.norm_v(x).permute(0, 3, 1, 2).reshape(B * W, C, H)
        vx = self.ssm_v(vx).reshape(B, W, C, H).permute(0, 2, 3, 1)

        # Merge and residual
        return x + self.mix(0.5 * (hx + vx))
