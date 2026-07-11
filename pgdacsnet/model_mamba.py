"""
pgdacsnet/model_mamba.py

Mamba-like / SSM-lite components for GprMambaSep.

Important: this module does NOT implement the official Mamba-2 block.
SelectiveSSMLite is a project-local, pure-PyTorch, content-adaptive SSM-lite
proxy. AxialSSMLiteBlock applies it along B-scan time and trace axes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["SelectiveSSMLite", "SelectiveSSMCUDA", "OfficialMamba2Sequence", "AxialSSMLiteBlock", "AxialMamba2Block", "Mamba2DBlock", "make_axial_sequence_block"]


class SelectiveSSMLite(nn.Module):
    """
    Pure-PyTorch Mamba-like / SSM-lite sequence mixer.

    Replaces the continuous-time SSM discretization with an input-dependent
    depthwise conv1d where the kernel at position t is modulated by the
    input activation. This captures the essential 'selection' property:
    different tokens get different processing.

    Compared to GatedSequenceBlock (static kernel):
    - Same O(L) complexity with conv1d
    - Content-dependent modulation adds conv_kernel * d_model extra params
    - Acts as a learnable proxy for content-adaptive SSM mixing

    Args:
        d_model: Model dimension (input/output channels).
        d_state: SSM state dimension (for interface compatibility with SSM-style blocks).
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
    Experimental wrapper around a selective_scan_cuda-style kernel.

    This path is not used by the default GprMambaSep model and is retained
    only for future experiments. It should not be cited as the project using
    official Mamba-2 unless the dependency and numerical interface are audited.

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

        # Experimental selective-scan-style parameter layout
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


class AxialSSMLiteBlock(nn.Module):
    """2D axial SSM-lite block powered by SelectiveSSMLite.

    Applies the content-adaptive SelectiveSSMLite independently along the
    horizontal (W/trace) and vertical (H/time) axes of a B-scan feature map,
    then merges with a learned 1x1 mix plus residual connection. This is a
    Mamba-like proxy, not the official Mamba-2 implementation.

    Args:
        channels: Number of input/output channels.
        d_state: SSM state dimension (passed through to SelectiveSSMLite).
        d_conv: Depthwise convolution kernel in SelectiveSSMLite.
        expand: Channel expansion factor for SelectiveSSMLite.
        bidirectional: Run independent forward and reverse sequence mixers.
    """

    def __init__(
        self,
        channels: int,
        d_state: int = 32,
        d_conv: int = 4,
        expand: int = 2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.norm_h = nn.GroupNorm(1, channels)
        self.norm_v = nn.GroupNorm(1, channels)
        self.bidirectional = bool(bidirectional)
        self.ssm_h = SelectiveSSMLite(
            d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand,
        )
        self.ssm_v = SelectiveSSMLite(
            d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand,
        )
        self.ssm_h_reverse = SelectiveSSMLite(
            d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand,
        ) if self.bidirectional else None
        self.ssm_v_reverse = SelectiveSSMLite(
            d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand,
        ) if self.bidirectional else None
        self.mix = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(1, channels),
            nn.GELU(),
        )

    @staticmethod
    def _mix_bidirectional(forward_mixer: nn.Module, reverse_mixer: nn.Module | None, sequence: torch.Tensor) -> torch.Tensor:
        forward = forward_mixer(sequence)
        if reverse_mixer is None:
            return forward
        reverse = torch.flip(reverse_mixer(torch.flip(sequence, dims=(-1,))), dims=(-1,))
        return 0.5 * (forward + reverse)

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
        hx = self._mix_bidirectional(self.ssm_h, self.ssm_h_reverse, hx).reshape(B, H, C, W).permute(0, 2, 1, 3)

        # Vertical SSM: treat each column as a sequence of length H
        # (B, C, H, W) -> (B*W, C, H) -> SSM -> (B*W, C, H) -> (B, C, H, W)
        vx = self.norm_v(x).permute(0, 3, 1, 2).reshape(B * W, C, H)
        vx = self._mix_bidirectional(self.ssm_v, self.ssm_v_reverse, vx).reshape(B, W, C, H).permute(0, 2, 3, 1)

        # Merge and residual
        return x + self.mix(0.5 * (hx + vx))




class OfficialMamba2Sequence(nn.Module):
    """Optional wrapper around mamba_ssm.Mamba2 for 1D sequences.

    This is disabled unless ``mamba-ssm`` is installed.  It exists so the code
    path is explicit and auditable; requesting official_mamba2 in an environment
    without the dependency raises a clear ImportError rather than silently using
    the SSM-lite proxy.
    """

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2, headdim: int = 16):
        super().__init__()
        d_model, expand, headdim = int(d_model), int(expand), int(headdim)
        if headdim <= 0 or (d_model * expand) % headdim != 0:
            raise ValueError(
                "official_mamba2 requires d_model * expand to be divisible by headdim; "
                f"got d_model={d_model}, expand={expand}, headdim={headdim}"
            )
        try:
            from mamba_ssm import Mamba2  # type: ignore[import-untyped]
        except Exception as exc:  # pragma: no cover - dependency-specific
            raise ImportError(
                "ssm_impl='official_mamba2' requires the official mamba-ssm package. "
                "Install a CUDA/Linux-compatible mamba-ssm build, or use "
                "ssm_impl='ssm_lite'."
            ) from exc
        self.block = Mamba2(
            d_model=d_model, d_state=int(d_state), d_conv=int(d_conv),
            expand=expand, headdim=headdim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mamba_ssm blocks use (B, L, C); project from/to the repository convention (B, C, L).
        return self.block(x.transpose(1, 2)).transpose(1, 2)


class AxialMamba2Block(nn.Module):
    """2D axial mixer using the official Mamba2 sequence block when available."""

    def __init__(self, channels: int, d_state: int = 32, d_conv: int = 4, expand: int = 2,
                 headdim: int = 16, bidirectional: bool = False):
        super().__init__()
        self.norm_h = nn.GroupNorm(1, channels)
        self.norm_v = nn.GroupNorm(1, channels)
        self.bidirectional = bool(bidirectional)
        self.ssm_h = OfficialMamba2Sequence(channels, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim)
        self.ssm_v = OfficialMamba2Sequence(channels, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim)
        self.ssm_h_reverse = OfficialMamba2Sequence(channels, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim) if self.bidirectional else None
        self.ssm_v_reverse = OfficialMamba2Sequence(channels, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim) if self.bidirectional else None
        self.mix = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.GroupNorm(1, channels), nn.GELU())

    @staticmethod
    def _mix_bidirectional(forward_mixer: nn.Module, reverse_mixer: nn.Module | None, sequence: torch.Tensor) -> torch.Tensor:
        forward = forward_mixer(sequence)
        if reverse_mixer is None:
            return forward
        reverse = torch.flip(reverse_mixer(torch.flip(sequence, dims=(-1,))), dims=(-1,))
        return 0.5 * (forward + reverse)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        hx = self.norm_h(x).permute(0, 2, 1, 3).reshape(B * H, C, W)
        hx = self._mix_bidirectional(self.ssm_h, self.ssm_h_reverse, hx).reshape(B, H, C, W).permute(0, 2, 1, 3)
        vx = self.norm_v(x).permute(0, 3, 1, 2).reshape(B * W, C, H)
        vx = self._mix_bidirectional(self.ssm_v, self.ssm_v_reverse, vx).reshape(B, W, C, H).permute(0, 2, 3, 1)
        return x + self.mix(0.5 * (hx + vx))


def make_axial_sequence_block(
    impl: str,
    channels: int,
    d_state: int = 32,
    d_conv: int = 4,
    expand: int = 2,
    headdim: int = 16,
    bidirectional: bool = False,
) -> nn.Module:
    """Factory for the axial sequence mixer used by GprMambaSep.

    ``ssm_lite`` is the default and is always available. ``official_mamba2`` is
    opt-in and requires mamba-ssm; it never falls back silently.
    """
    impl = str(impl or "ssm_lite").lower()
    if impl in ("ssm_lite", "lite", "axial_ssm_lite", "mamba_like"):
        return AxialSSMLiteBlock(channels, d_state=d_state, d_conv=d_conv, expand=expand, bidirectional=bidirectional)
    if impl in ("official_mamba2", "mamba2", "mamba-2"):
        return AxialMamba2Block(channels, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim, bidirectional=bidirectional)
    raise ValueError(f"Unknown axial sequence mixer impl: {impl!r}")


# Backward-compatible alias.  Kept so old checkpoints/configs/tests import,
# but new code should use AxialSSMLiteBlock to avoid claiming official Mamba-2.
Mamba2DBlock = AxialSSMLiteBlock
