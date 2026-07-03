
import torch
from torch import nn
import torch.nn.functional as F


def compress_raw(x, scale=1e-3):
    """Compress direct-wave dynamic range while preserving raw-signal polarity."""
    scale = float(scale)
    if scale <= 0:
        return x
    denominator = torch.log1p(torch.as_tensor(1.0 / scale, dtype=x.dtype, device=x.device))
    return torch.sign(x) * torch.log1p(torch.abs(x) / scale) / denominator


class ConvBlock(nn.Module):
    def __init__(self,a,b):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(1,b),nn.ReLU(inplace=True),nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(1,b),nn.ReLU(inplace=True))
    def forward(self,x): return self.net(x)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class DilatedBottleneck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=1, dilation=1),
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2),
            nn.Conv2d(channels, channels, 3, padding=4, dilation=4),
        ])
        self.mix = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.GroupNorm(1, channels),
            nn.ReLU(inplace=True),
            SEBlock(channels),
        )

    def forward(self, x):
        return self.mix(torch.cat([branch(x) for branch in self.branches], dim=1)) + x


class ConvNeXtBlock(nn.Module):
    """Small ConvNeXt-style block for modern local B-scan texture modeling."""
    def __init__(self, channels, expansion=4, dropout=0.0):
        super().__init__()
        hidden = channels * int(expansion)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 7, padding=3, groups=channels),
            nn.GroupNorm(1, channels),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Conv2d(hidden, channels, 1),
        )
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * 1e-2)

    def forward(self, x):
        return x + self.gamma * self.net(x)


class ConvNeXtStage(nn.Module):
    def __init__(self, in_ch, out_ch, depth=2, dropout=0.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ConvNeXtBlock(out_ch, dropout=dropout) for _ in range(int(depth))])

    def forward(self, x):
        return self.blocks(self.proj(x))


class MetadataFiLM(nn.Module):
    """Stable feature-wise conditioning from terrain/flight metadata channels."""
    def __init__(self, meta_channels, target_channels, hidden=32):
        super().__init__()
        self.meta_channels = int(meta_channels)
        hidden = max(4, int(hidden))
        if self.meta_channels <= 0:
            self.net = None
        else:
            self.net = nn.Sequential(
                nn.Linear(self.meta_channels, hidden),
                nn.GELU(),
                nn.Linear(hidden, target_channels * 2),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, meta):
        if self.net is None or meta is None or meta.shape[1] == 0:
            return x
        pooled = meta.mean(dim=(2, 3))
        gamma, beta = self.net(pooled).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return x * (1.0 + gamma) + beta


class GatedSequenceBlock(nn.Module):
    """Bidirectional depthwise sequence mixer used as a lightweight SSM/Mamba proxy."""
    def __init__(self, channels, kernel_size=31, dropout=0.0):
        super().__init__()
        pad = int(kernel_size) // 2
        self.dw_f = nn.Conv1d(channels, channels, kernel_size, padding=pad, groups=channels)
        self.dw_b = nn.Conv1d(channels, channels, kernel_size, padding=pad, groups=channels)
        self.gate_f = nn.Conv1d(channels, channels * 2, 1)
        self.gate_b = nn.Conv1d(channels, channels * 2, 1)
        self.mix = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

    def branch(self, x, dw, gate):
        a, b = gate(dw(x)).chunk(2, dim=1)
        return a * torch.sigmoid(b)

    def forward(self, x):
        y_f = self.branch(x, self.dw_f, self.gate_f)
        x_rev = torch.flip(x, dims=[-1])
        y_b = torch.flip(self.branch(x_rev, self.dw_b, self.gate_b), dims=[-1])
        return x + self.drop(self.mix(y_f + y_b))


class AxialSSMLiteBlock(nn.Module):
    """Axis-aware long-context block: horizontal traces plus vertical time/depth sequences."""
    def __init__(self, channels, kernel_size=31, dropout=0.0):
        super().__init__()
        self.norm_h = nn.GroupNorm(1, channels)
        self.norm_v = nn.GroupNorm(1, channels)
        self.hseq = GatedSequenceBlock(channels, kernel_size=kernel_size, dropout=dropout)
        self.vseq = GatedSequenceBlock(channels, kernel_size=kernel_size, dropout=dropout)
        self.mix = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(1, channels),
            nn.GELU(),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        hx = self.norm_h(x).permute(0, 2, 1, 3).reshape(b * h, c, w)
        hx = self.hseq(hx).reshape(b, h, c, w).permute(0, 2, 1, 3)
        vx = self.norm_v(x).permute(0, 3, 1, 2).reshape(b * w, c, h)
        vx = self.vseq(vx).reshape(b, w, c, h).permute(0, 2, 3, 1)
        return x + self.mix(0.5 * (hx + vx))


class CrossScanSSMLiteBlock(nn.Module):
    """VMamba/Vim-inspired dependency-light 2D scan mixer."""
    def __init__(self, channels, kernel_size=31, dropout=0.0):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.hseq = GatedSequenceBlock(channels, kernel_size=kernel_size, dropout=dropout)
        self.vseq = GatedSequenceBlock(channels, kernel_size=kernel_size, dropout=dropout)
        self.local = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.gate = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())
        self.mix = nn.Sequential(nn.Conv2d(channels * 3, channels, 1), nn.GroupNorm(1, channels), nn.GELU())

    def forward(self, x):
        b, c, h, w = x.shape
        z = self.norm(x)
        hx = z.permute(0, 2, 1, 3).reshape(b * h, c, w)
        hx = self.hseq(hx).reshape(b, h, c, w).permute(0, 2, 1, 3)
        vx = z.permute(0, 3, 1, 2).reshape(b * w, c, h)
        vx = self.vseq(vx).reshape(b, w, c, h).permute(0, 2, 3, 1)
        y = self.mix(torch.cat([hx, vx, self.local(z)], dim=1))
        return x + self.gate(z) * y


class StripeAttentionBlock(nn.Module):
    """Efficient stripe attention over pooled horizontal and vertical tokens."""
    def __init__(self, channels, heads=4, dropout=0.0):
        super().__init__()
        heads = max(1, min(int(heads), channels))
        while channels % heads != 0 and heads > 1:
            heads -= 1
        self.norm = nn.GroupNorm(1, channels)
        self.attn_h = nn.MultiheadAttention(channels, heads, dropout=float(dropout), batch_first=True)
        self.attn_v = nn.MultiheadAttention(channels, heads, dropout=float(dropout), batch_first=True)
        self.proj = nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.GroupNorm(1, channels), nn.GELU())
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * 1e-2)

    def forward(self, x):
        z = self.norm(x)
        htok = z.mean(dim=2).transpose(1, 2)
        vtok = z.mean(dim=3).transpose(1, 2)
        hctx = self.attn_h(htok, htok, htok, need_weights=False)[0].transpose(1, 2)[:, :, None, :].expand_as(x)
        vctx = self.attn_v(vtok, vtok, vtok, need_weights=False)[0].transpose(1, 2)[:, :, :, None].expand_as(x)
        return x + self.gamma * self.proj(torch.cat([hctx, vctx], dim=1))


class SpectralGatingModule(nn.Module):
    """Fixed learnable spectral gate in the f-k domain.

    Learns a single global spectral mask (not input-dependent) that
    suppresses clutter-associated frequency-wavenumber components.
    Applied uniformly to all inputs — impossible to overfit to a single line.
    ~128 learnable parameters for bottleneck 128x64.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # (1, C, 1, 1) — batch-first for easy expand
        self.mask_logit = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        fft_w = W // 2 + 1
        ml = self.mask_logit  # (1, C, 1, 1)
        if ml.shape[2] != H or ml.shape[3] != fft_w:
            ml = nn.functional.interpolate(ml, size=(H, fft_w), mode='bilinear', align_corners=False)
        gate = torch.sigmoid(ml).expand(B, -1, -1, -1).clone()  # (B, C, H, fft_w)
        gate[:, :, :, 0] = 1.0  # always pass DC
        xf = torch.fft.rfft2(x, norm="ortho")
        x_gated = torch.fft.irfft2(xf * gate[:, :, :xf.shape[2], :xf.shape[3]], s=(H, W), norm="ortho")
        return x + self.gamma.clamp(-1.0, 1.0) * (x_gated - x)


class CenterRefineHead(nn.Module):
    """Sharper center-head candidate for curve extraction experiments."""
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Conv2d(channels, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class RawOnlyConvNeXtUNetV17A(nn.Module):
    """v1.7-A: modern ConvNeXt U-Net baseline without sequence mixer."""
    def __init__(self, base_ch=16, dropout=0.05, input_channels=1):
        super().__init__(); c=base_ch
        self.input_channels=int(input_channels)
        self.e1=ConvNeXtStage(self.input_channels*2,c,depth=2,dropout=dropout); self.p=nn.MaxPool2d(2)
        self.e2=ConvNeXtStage(c,c*2,depth=2,dropout=dropout); self.e3=ConvNeXtStage(c*2,c*4,depth=3,dropout=dropout)
        self.bottleneck=nn.Sequential(ConvNeXtBlock(c*4,dropout=dropout), DilatedBottleneck(c*4))
        self.se2=SEBlock(c*2); self.se1=SEBlock(c)
        self.u2=nn.ConvTranspose2d(c*4,c*2,2,2); self.d2=ConvNeXtStage(c*4,c*2,depth=2,dropout=dropout)
        self.u1=nn.ConvTranspose2d(c*2,c,2,2); self.d1=ConvNeXtStage(c*2,c,depth=2,dropout=dropout)
        self.mask=nn.Conv2d(c,1,1)
        self.center=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.GELU(),nn.Conv2d(c,1,1))
        self.pres=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.GELU(),nn.Conv2d(c,1,1))

    def forward(self,x):
        e1=self.e1(torch.cat((x,torch.abs(x)),dim=1)); e2=self.e2(self.p(e1)); e3=self.bottleneck(self.e3(self.p(e2)))
        d2=self.d2(torch.cat([self.u2(e3),self.se2(e2)],dim=1)); d1=self.d1(torch.cat([self.u1(d2),self.se1(e1)],dim=1))
        pres_map=self.pres(d1)
        lo=max(0,int(round(pres_map.shape[2]*0.35))); hi=max(lo+1,int(round(pres_map.shape[2]*0.75)))
        return self.mask(d1), pres_map[:,:,lo:hi].amax(dim=2), self.center(d1)


class RawTerrainFiLMConvNeXtUNetV18B(nn.Module):
    """v1.8-B: raw-only ConvNeXt U-Net with FiLM terrain/flight conditioning.

    Channel 0 is the compressed raw B-scan. Remaining channels are normalized
    terrain/flight metadata. Metadata modulates internal features but is not
    duplicated through the raw absolute-value branch.
    """
    def __init__(self, base_ch=16, dropout=0.05, input_channels=1, film_hidden=32):
        super().__init__(); c=base_ch
        self.input_channels=int(input_channels)
        self.meta_channels=max(0,self.input_channels-1)
        self.e1=ConvNeXtStage(2,c,depth=2,dropout=dropout); self.p=nn.MaxPool2d(2)
        self.e2=ConvNeXtStage(c,c*2,depth=2,dropout=dropout); self.e3=ConvNeXtStage(c*2,c*4,depth=3,dropout=dropout)
        self.bottleneck=nn.Sequential(ConvNeXtBlock(c*4,dropout=dropout), DilatedBottleneck(c*4))
        self.film1=MetadataFiLM(self.meta_channels,c,film_hidden)
        self.film2=MetadataFiLM(self.meta_channels,c*2,film_hidden)
        self.film3=MetadataFiLM(self.meta_channels,c*4,film_hidden)
        self.filmb=MetadataFiLM(self.meta_channels,c*4,film_hidden)
        self.se2=SEBlock(c*2); self.se1=SEBlock(c)
        self.u2=nn.ConvTranspose2d(c*4,c*2,2,2); self.d2=ConvNeXtStage(c*4,c*2,depth=2,dropout=dropout)
        self.u1=nn.ConvTranspose2d(c*2,c,2,2); self.d1=ConvNeXtStage(c*2,c,depth=2,dropout=dropout)
        self.mask=nn.Conv2d(c,1,1)
        self.center=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.GELU(),nn.Conv2d(c,1,1))
        self.pres=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.GELU(),nn.Conv2d(c,1,1))

    def forward(self,x):
        raw=x[:,:1]
        meta=x[:,1:] if x.shape[1]>1 else None
        e1=self.film1(self.e1(torch.cat((raw,torch.abs(raw)),dim=1)),meta)
        e2=self.film2(self.e2(self.p(e1)),meta)
        e3=self.film3(self.e3(self.p(e2)),meta)
        e3=self.filmb(self.bottleneck(e3),meta)
        d2=self.d2(torch.cat([self.u2(e3),self.se2(e2)],dim=1)); d1=self.d1(torch.cat([self.u1(d2),self.se1(e1)],dim=1))
        pres_map=self.pres(d1)
        lo=max(0,int(round(pres_map.shape[2]*0.35))); hi=max(lo+1,int(round(pres_map.shape[2]*0.75)))
        return self.mask(d1), pres_map[:,:,lo:hi].amax(dim=2), self.center(d1)


class RawOnlyConvNeXtAxialSSMUNetV17B(nn.Module):
    """v1.7-B: ConvNeXt U-Net with axial SSM-lite blocks for long B-scan continuity."""
    def __init__(self, base_ch=16, dropout=0.05, ssm_kernel=31, input_channels=1):
        super().__init__(); c=base_ch
        self.input_channels=int(input_channels)
        self.e1=ConvNeXtStage(self.input_channels*2,c,depth=2,dropout=dropout); self.p=nn.MaxPool2d(2)
        self.e2=ConvNeXtStage(c,c*2,depth=2,dropout=dropout); self.ssm2=AxialSSMLiteBlock(c*2,ssm_kernel,dropout)
        self.e3=ConvNeXtStage(c*2,c*4,depth=2,dropout=dropout)
        self.bottleneck=nn.Sequential(ConvNeXtBlock(c*4,dropout=dropout), AxialSSMLiteBlock(c*4,ssm_kernel,dropout), DilatedBottleneck(c*4))
        self.se2=SEBlock(c*2); self.se1=SEBlock(c)
        self.u2=nn.ConvTranspose2d(c*4,c*2,2,2); self.d2=ConvNeXtStage(c*4,c*2,depth=2,dropout=dropout)
        self.u1=nn.ConvTranspose2d(c*2,c,2,2); self.d1=ConvNeXtStage(c*2,c,depth=2,dropout=dropout)
        self.mask=nn.Conv2d(c,1,1)
        self.center=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.GELU(),nn.Conv2d(c,1,1))
        self.pres=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.GELU(),nn.Conv2d(c,1,1))

    def forward(self,x):
        e1=self.e1(torch.cat((x,torch.abs(x)),dim=1)); e2=self.ssm2(self.e2(self.p(e1))); e3=self.bottleneck(self.e3(self.p(e2)))
        d2=self.d2(torch.cat([self.u2(e3),self.se2(e2)],dim=1)); d1=self.d1(torch.cat([self.u1(d2),self.se1(e1)],dim=1))
        pres_map=self.pres(d1)
        lo=max(0,int(round(pres_map.shape[2]*0.35))); hi=max(lo+1,int(round(pres_map.shape[2]*0.75)))
        return self.mask(d1), pres_map[:,:,lo:hi].amax(dim=2), self.center(d1)


class RawOnlyVMambaLiteUNetV19A(nn.Module):
    """v1.9-A: ConvNeXt U-Net with dependency-light cross-scan SSM blocks."""
    def __init__(self, base_ch=16, dropout=0.05, ssm_kernel=31, input_channels=1):
        super().__init__(); c=base_ch
        self.input_channels=int(input_channels)
        self.e1=ConvNeXtStage(self.input_channels*2,c,depth=2,dropout=dropout); self.p=nn.MaxPool2d(2)
        self.e2=ConvNeXtStage(c,c*2,depth=2,dropout=dropout); self.scan2=CrossScanSSMLiteBlock(c*2,ssm_kernel,dropout)
        self.e3=ConvNeXtStage(c*2,c*4,depth=2,dropout=dropout)
        self.bottleneck=nn.Sequential(CrossScanSSMLiteBlock(c*4,ssm_kernel,dropout), ConvNeXtBlock(c*4,dropout=dropout), DilatedBottleneck(c*4))
        self.se2=SEBlock(c*2); self.se1=SEBlock(c)
        self.u2=nn.ConvTranspose2d(c*4,c*2,2,2); self.d2=ConvNeXtStage(c*4,c*2,depth=2,dropout=dropout)
        self.u1=nn.ConvTranspose2d(c*2,c,2,2); self.d1=ConvNeXtStage(c*2,c,depth=2,dropout=dropout)
        self.mask=nn.Conv2d(c,1,1); self.center=CenterRefineHead(c,dropout)
        self.pres=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.GELU(),nn.Conv2d(c,1,1))

    def forward(self,x):
        e1=self.e1(torch.cat((x,torch.abs(x)),dim=1)); e2=self.scan2(self.e2(self.p(e1))); e3=self.bottleneck(self.e3(self.p(e2)))
        d2=self.d2(torch.cat([self.u2(e3),self.se2(e2)],dim=1)); d1=self.d1(torch.cat([self.u1(d2),self.se1(e1)],dim=1))
        pres_map=self.pres(d1); lo=max(0,int(round(pres_map.shape[2]*0.35))); hi=max(lo+1,int(round(pres_map.shape[2]*0.75)))
        return self.mask(d1), pres_map[:,:,lo:hi].amax(dim=2), self.center(d1)


class RawOnlyUMambaHybridUNetV19B(nn.Module):
    """v1.9-B: U-Mamba-style hybrid with SSM in bottleneck and decoder."""
    def __init__(self, base_ch=16, dropout=0.05, ssm_kernel=47, input_channels=1):
        super().__init__(); c=base_ch
        self.input_channels=int(input_channels)
        self.e1=ConvNeXtStage(self.input_channels*2,c,depth=2,dropout=dropout); self.p=nn.MaxPool2d(2)
        self.e2=ConvNeXtStage(c,c*2,depth=2,dropout=dropout); self.e3=ConvNeXtStage(c*2,c*4,depth=2,dropout=dropout)
        self.bottleneck=nn.Sequential(ConvNeXtBlock(c*4,dropout=dropout), CrossScanSSMLiteBlock(c*4,ssm_kernel,dropout), CrossScanSSMLiteBlock(c*4,ssm_kernel,dropout))
        self.se2=SEBlock(c*2); self.se1=SEBlock(c)
        self.u2=nn.ConvTranspose2d(c*4,c*2,2,2); self.d2=ConvNeXtStage(c*4,c*2,depth=2,dropout=dropout); self.scan_d2=AxialSSMLiteBlock(c*2,ssm_kernel,dropout)
        self.u1=nn.ConvTranspose2d(c*2,c,2,2); self.d1=ConvNeXtStage(c*2,c,depth=2,dropout=dropout)
        self.mask=nn.Conv2d(c,1,1); self.center=CenterRefineHead(c,dropout)
        self.pres=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.GELU(),nn.Conv2d(c,1,1))

    def forward(self,x):
        e1=self.e1(torch.cat((x,torch.abs(x)),dim=1)); e2=self.e2(self.p(e1)); e3=self.bottleneck(self.e3(self.p(e2)))
        d2=self.scan_d2(self.d2(torch.cat([self.u2(e3),self.se2(e2)],dim=1))); d1=self.d1(torch.cat([self.u1(d2),self.se1(e1)],dim=1))
        pres_map=self.pres(d1); lo=max(0,int(round(pres_map.shape[2]*0.35))); hi=max(lo+1,int(round(pres_map.shape[2]*0.75)))
        return self.mask(d1), pres_map[:,:,lo:hi].amax(dim=2), self.center(d1)


class RawOnlyStripeAttentionUNetV19C(nn.Module):
    """v1.9-C: Swin/CSWin-inspired stripe-attention U-Net for layer continuity."""
    def __init__(self, base_ch=16, dropout=0.05, attention_heads=4, input_channels=1):
        super().__init__(); c=base_ch
        self.input_channels=int(input_channels)
        self.e1=ConvNeXtStage(self.input_channels*2,c,depth=2,dropout=dropout); self.p=nn.MaxPool2d(2)
        self.e2=ConvNeXtStage(c,c*2,depth=2,dropout=dropout); self.attn2=StripeAttentionBlock(c*2,attention_heads,dropout)
        self.e3=ConvNeXtStage(c*2,c*4,depth=2,dropout=dropout)
        self.bottleneck=nn.Sequential(StripeAttentionBlock(c*4,attention_heads,dropout), ConvNeXtBlock(c*4,dropout=dropout), DilatedBottleneck(c*4))
        self.se2=SEBlock(c*2); self.se1=SEBlock(c)
        self.u2=nn.ConvTranspose2d(c*4,c*2,2,2); self.d2=ConvNeXtStage(c*4,c*2,depth=2,dropout=dropout)
        self.u1=nn.ConvTranspose2d(c*2,c,2,2); self.d1=ConvNeXtStage(c*2,c,depth=2,dropout=dropout)
        self.mask=nn.Conv2d(c,1,1); self.center=CenterRefineHead(c,dropout)
        self.pres=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.GELU(),nn.Conv2d(c,1,1))

    def forward(self,x):
        e1=self.e1(torch.cat((x,torch.abs(x)),dim=1)); e2=self.attn2(self.e2(self.p(e1))); e3=self.bottleneck(self.e3(self.p(e2)))
        d2=self.d2(torch.cat([self.u2(e3),self.se2(e2)],dim=1)); d1=self.d1(torch.cat([self.u1(d2),self.se1(e1)],dim=1))
        pres_map=self.pres(d1); lo=max(0,int(round(pres_map.shape[2]*0.35))); hi=max(lo+1,int(round(pres_map.shape[2]*0.75)))
        return self.mask(d1), pres_map[:,:,lo:hi].amax(dim=2), self.center(d1)


class RawOnlyMambaVisionHybridUNetV19D(nn.Module):
    """v1.9-D: MambaVision-style ConvNeXt + scan + stripe-attention hybrid."""
    def __init__(self, base_ch=16, dropout=0.05, ssm_kernel=31, attention_heads=4, input_channels=1):
        super().__init__(); c=base_ch
        self.input_channels=int(input_channels)
        self.e1=ConvNeXtStage(self.input_channels*2,c,depth=2,dropout=dropout); self.p=nn.MaxPool2d(2)
        self.e2=ConvNeXtStage(c,c*2,depth=2,dropout=dropout); self.scan2=CrossScanSSMLiteBlock(c*2,ssm_kernel,dropout)
        self.e3=ConvNeXtStage(c*2,c*4,depth=2,dropout=dropout)
        self.bottleneck=nn.Sequential(CrossScanSSMLiteBlock(c*4,ssm_kernel,dropout), StripeAttentionBlock(c*4,attention_heads,dropout), DilatedBottleneck(c*4))
        self.se2=SEBlock(c*2); self.se1=SEBlock(c)
        self.u2=nn.ConvTranspose2d(c*4,c*2,2,2); self.d2=ConvNeXtStage(c*4,c*2,depth=2,dropout=dropout)
        self.u1=nn.ConvTranspose2d(c*2,c,2,2); self.d1=ConvNeXtStage(c*2,c,depth=2,dropout=dropout)
        self.mask=nn.Conv2d(c,1,1); self.center=CenterRefineHead(c,dropout)
        self.pres=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.GELU(),nn.Conv2d(c,1,1))

    def forward(self,x):
        e1=self.e1(torch.cat((x,torch.abs(x)),dim=1)); e2=self.scan2(self.e2(self.p(e1))); e3=self.bottleneck(self.e3(self.p(e2)))
        d2=self.d2(torch.cat([self.u2(e3),self.se2(e2)],dim=1)); d1=self.d1(torch.cat([self.u1(d2),self.se1(e1)],dim=1))
        pres_map=self.pres(d1); lo=max(0,int(round(pres_map.shape[2]*0.35))); hi=max(lo+1,int(round(pres_map.shape[2]*0.75)))
        return self.mask(d1), pres_map[:,:,lo:hi].amax(dim=2), self.center(d1)


class RawOnlyConvNeXtPPUNetV19E(nn.Module):
    """v1.9-E: strong ConvNeXt++ baseline with deeper context and sharper center head."""
    def __init__(self, base_ch=16, dropout=0.05, input_channels=1):
        super().__init__(); c=base_ch
        self.input_channels=int(input_channels)
        self.e1=ConvNeXtStage(self.input_channels*2,c,depth=3,dropout=dropout); self.p=nn.MaxPool2d(2)
        self.e2=ConvNeXtStage(c,c*2,depth=3,dropout=dropout); self.e3=ConvNeXtStage(c*2,c*4,depth=4,dropout=dropout)
        self.bottleneck=nn.Sequential(DilatedBottleneck(c*4), ConvNeXtBlock(c*4,dropout=dropout), DilatedBottleneck(c*4), SEBlock(c*4))
        self.se2=SEBlock(c*2); self.se1=SEBlock(c)
        self.u2=nn.ConvTranspose2d(c*4,c*2,2,2); self.d2=ConvNeXtStage(c*4,c*2,depth=3,dropout=dropout)
        self.u1=nn.ConvTranspose2d(c*2,c,2,2); self.d1=ConvNeXtStage(c*2,c,depth=3,dropout=dropout)
        self.mask=nn.Conv2d(c,1,1); self.center=CenterRefineHead(c,dropout)
        self.pres=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.GELU(),nn.Conv2d(c,1,1))

    def forward(self,x):
        e1=self.e1(torch.cat((x,torch.abs(x)),dim=1)); e2=self.e2(self.p(e1)); e3=self.bottleneck(self.e3(self.p(e2)))
        d2=self.d2(torch.cat([self.u2(e3),self.se2(e2)],dim=1)); d1=self.d1(torch.cat([self.u1(d2),self.se1(e1)],dim=1))
        pres_map=self.pres(d1); lo=max(0,int(round(pres_map.shape[2]*0.35))); hi=max(lo+1,int(round(pres_map.shape[2]*0.75)))
        return self.mask(d1), pres_map[:,:,lo:hi].amax(dim=2), self.center(d1)


class RawOnlySGUSSMUNetV111(nn.Module):
    """v1.11 SG-USSM: v1.9D + Spectral Gating Module.

    Adds a learnable spectral gate in the bottleneck to suppress
    clutter-associated f-k components.  gamma=0 init makes this a
    no-op at the start of training, so it is strictly safe to add.
    """
    def __init__(self, base_ch=16, dropout=0.05, ssm_kernel=31, attention_heads=4, input_channels=1):
        super().__init__()
        c = base_ch
        self.input_channels = int(input_channels)
        # Encoder
        self.e1 = ConvNeXtStage(self.input_channels * 2, c, depth=2, dropout=dropout)
        self.p = nn.MaxPool2d(2)
        self.e2 = ConvNeXtStage(c, c * 2, depth=2, dropout=dropout)
        self.scan2 = CrossScanSSMLiteBlock(c * 2, ssm_kernel, dropout)
        self.e3 = ConvNeXtStage(c * 2, c * 4, depth=2, dropout=dropout)
        # Bottleneck — same as v1.9D
        self.bottleneck = nn.Sequential(
            CrossScanSSMLiteBlock(c * 4, ssm_kernel, dropout),
            StripeAttentionBlock(c * 4, attention_heads, dropout),
            DilatedBottleneck(c * 4),
        )
        # NEW: Spectral Gating Module
        self.sgm = SpectralGatingModule(c * 4)
        # Decoder
        self.se2 = SEBlock(c * 2)
        self.se1 = SEBlock(c)
        self.u2 = nn.ConvTranspose2d(c * 4, c * 2, 2, 2)
        self.d2 = ConvNeXtStage(c * 4, c * 2, depth=2, dropout=dropout)
        self.u1 = nn.ConvTranspose2d(c * 2, c, 2, 2)
        self.d1 = ConvNeXtStage(c * 2, c, depth=2, dropout=dropout)
        # Heads
        self.mask = nn.Conv2d(c, 1, 1)
        self.center = CenterRefineHead(c, dropout)
        self.pres = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.GELU(), nn.Conv2d(c, 1, 1))

    def forward(self, x):
        e1 = self.e1(torch.cat((x, torch.abs(x)), dim=1))
        e2 = self.scan2(self.e2(self.p(e1)))
        e3 = self.bottleneck(self.e3(self.p(e2)))
        e3 = self.sgm(e3)  # Spectral gating — identity at init
        d2 = self.d2(torch.cat([self.u2(e3), self.se2(e2)], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), self.se1(e1)], dim=1))
        pres_map = self.pres(d1)
        lo = max(0, int(round(pres_map.shape[2] * 0.35)))
        hi = max(lo + 1, int(round(pres_map.shape[2] * 0.75)))
        return self.mask(d1), pres_map[:, :, lo:hi].amax(dim=2), self.center(d1)


class RawOnlyUNet(nn.Module):
    def __init__(self,base_ch=16,input_channels=1):
        super().__init__(); c=base_ch
        self.input_channels=int(input_channels)
        self.e1=ConvBlock(self.input_channels*2,c); self.p=nn.MaxPool2d(2); self.e2=ConvBlock(c,c*2); self.e3=ConvBlock(c*2,c*4)
        self.u2=nn.ConvTranspose2d(c*4,c*2,2,2); self.d2=ConvBlock(c*4,c*2); self.u1=nn.ConvTranspose2d(c*2,c,2,2); self.d1=ConvBlock(c*2,c)
        self.mask=nn.Conv2d(c,1,1)
        self.pres=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.ReLU(inplace=True),nn.Conv2d(c,1,1))
    def forward(self,x):
        e1=self.e1(torch.cat((x,torch.abs(x)),dim=1)); e2=self.e2(self.p(e1)); e3=self.e3(self.p(e2));
        d2=self.d2(torch.cat([self.u2(e3),e2],dim=1)); d1=self.d1(torch.cat([self.u1(d2),e1],dim=1));
        pres_map=self.pres(d1)
        lo=max(0,int(round(pres_map.shape[2]*0.35))); hi=max(lo+1,int(round(pres_map.shape[2]*0.75)))
        return self.mask(d1), pres_map[:,:,lo:hi].amax(dim=2)


class RawOnlyUNetV15Light(nn.Module):
    """Light-capacity v1.5 candidate: SE skip attention plus a dilated bottleneck."""
    def __init__(self, base_ch=16, dropout=0.05, input_channels=1):
        super().__init__(); c=base_ch
        self.input_channels=int(input_channels)
        self.e1=ConvBlock(self.input_channels*2,c); self.p=nn.MaxPool2d(2); self.e2=ConvBlock(c,c*2); self.e3=ConvBlock(c*2,c*4)
        self.bottleneck=DilatedBottleneck(c*4)
        self.se2=SEBlock(c*2); self.se1=SEBlock(c)
        self.u2=nn.ConvTranspose2d(c*4,c*2,2,2); self.d2=ConvBlock(c*4,c*2)
        self.u1=nn.ConvTranspose2d(c*2,c,2,2); self.d1=ConvBlock(c*2,c)
        self.drop=nn.Dropout2d(float(dropout)) if float(dropout)>0 else nn.Identity()
        self.mask=nn.Conv2d(c,1,1)
        self.pres=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.ReLU(inplace=True),nn.Conv2d(c,1,1))

    def forward(self,x):
        e1=self.e1(torch.cat((x,torch.abs(x)),dim=1)); e2=self.e2(self.p(e1)); e3=self.bottleneck(self.e3(self.p(e2)))
        d2=self.d2(torch.cat([self.u2(e3),self.se2(e2)],dim=1)); d1=self.d1(torch.cat([self.u1(d2),self.se1(e1)],dim=1))
        d1=self.drop(d1)
        pres_map=self.pres(d1)
        lo=max(0,int(round(pres_map.shape[2]*0.35))); hi=max(lo+1,int(round(pres_map.shape[2]*0.75)))
        return self.mask(d1), pres_map[:,:,lo:hi].amax(dim=2)


class RawOnlyUNetV16Final(nn.Module):
    """Single-model paper candidate with light attention, multiscale context, and center head."""
    def __init__(self, base_ch=16, dropout=0.05, input_channels=1):
        super().__init__(); c=base_ch
        self.input_channels=int(input_channels)
        self.e1=ConvBlock(self.input_channels*2,c); self.p=nn.MaxPool2d(2); self.e2=ConvBlock(c,c*2); self.e3=ConvBlock(c*2,c*4)
        self.bottleneck=nn.Sequential(DilatedBottleneck(c*4), DilatedBottleneck(c*4))
        self.se2=SEBlock(c*2); self.se1=SEBlock(c)
        self.u2=nn.ConvTranspose2d(c*4,c*2,2,2); self.d2=ConvBlock(c*4,c*2)
        self.u1=nn.ConvTranspose2d(c*2,c,2,2); self.d1=ConvBlock(c*2,c)
        self.drop=nn.Dropout2d(float(dropout)) if float(dropout)>0 else nn.Identity()
        self.mask=nn.Conv2d(c,1,1)
        self.center=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.ReLU(inplace=True),nn.Conv2d(c,1,1))
        self.pres=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.ReLU(inplace=True),nn.Conv2d(c,1,1))

    def forward(self,x):
        e1=self.e1(torch.cat((x,torch.abs(x)),dim=1)); e2=self.e2(self.p(e1)); e3=self.bottleneck(self.e3(self.p(e2)))
        d2=self.d2(torch.cat([self.u2(e3),self.se2(e2)],dim=1)); d1=self.d1(torch.cat([self.u1(d2),self.se1(e1)],dim=1))
        d1=self.drop(d1)
        pres_map=self.pres(d1)
        lo=max(0,int(round(pres_map.shape[2]*0.35))); hi=max(lo+1,int(round(pres_map.shape[2]*0.75)))
        return self.mask(d1), pres_map[:,:,lo:hi].amax(dim=2), self.center(d1)


def build_model(cfg):
    arch = str(cfg.get("model_arch", "raw_unet")).lower()
    input_channels = int(cfg.get("input_channels", 1 + len(cfg.get("terrain_feature_names", [])) if cfg.get("use_terrain_features", False) else 1))
    if arch in ("raw_unet", "rawonlyunet", "v1_4"):
        return RawOnlyUNet(cfg["base_ch"], input_channels=input_channels)
    if arch in ("raw_unet_v15_light", "rawonlyunetv15light", "v1_5_light"):
        return RawOnlyUNetV15Light(cfg["base_ch"], dropout=float(cfg.get("model_dropout", 0.05)), input_channels=input_channels)
    if arch in ("raw_unet_v16_final", "rawonlyunetv16final", "v1_6_final"):
        return RawOnlyUNetV16Final(cfg["base_ch"], dropout=float(cfg.get("model_dropout", 0.05)), input_channels=input_channels)
    if arch in ("raw_convnext_unet_v17a", "convnext_unet_v17a", "v1_7a"):
        return RawOnlyConvNeXtUNetV17A(cfg["base_ch"], dropout=float(cfg.get("model_dropout", 0.05)), input_channels=input_channels)
    if arch in ("raw_terrain_film_convnext_unet_v18b", "terrain_film_convnext_unet_v18b", "v1_8b"):
        return RawTerrainFiLMConvNeXtUNetV18B(
            cfg["base_ch"],
            dropout=float(cfg.get("model_dropout", 0.05)),
            input_channels=input_channels,
            film_hidden=int(cfg.get("terrain_film_hidden", 32)),
        )
    if arch in ("raw_convnext_axial_ssm_unet_v17b", "convnext_axial_ssm_unet_v17b", "v1_7b", "pgda_ssmnet_v17b"):
        return RawOnlyConvNeXtAxialSSMUNetV17B(
            cfg["base_ch"],
            dropout=float(cfg.get("model_dropout", 0.05)),
            ssm_kernel=int(cfg.get("ssm_kernel", 31)),
            input_channels=input_channels,
        )
    if arch in ("v1_9a_vmamba_lite", "raw_vmamba_lite_unet_v19a", "vmamba_lite"):
        return RawOnlyVMambaLiteUNetV19A(cfg["base_ch"], dropout=float(cfg.get("model_dropout", 0.05)), ssm_kernel=int(cfg.get("ssm_kernel", 31)), input_channels=input_channels)
    if arch in ("v1_9b_umamba_hybrid", "raw_umamba_hybrid_unet_v19b", "umamba_hybrid"):
        return RawOnlyUMambaHybridUNetV19B(cfg["base_ch"], dropout=float(cfg.get("model_dropout", 0.05)), ssm_kernel=int(cfg.get("ssm_kernel", 47)), input_channels=input_channels)
    if arch in ("v1_9c_stripe_attention", "raw_stripe_attention_unet_v19c", "stripe_attention"):
        return RawOnlyStripeAttentionUNetV19C(cfg["base_ch"], dropout=float(cfg.get("model_dropout", 0.05)), attention_heads=int(cfg.get("attention_heads", 4)), input_channels=input_channels)
    if arch in ("v1_9d_mambavision_hybrid", "raw_mambavision_hybrid_unet_v19d", "mambavision_hybrid"):
        return RawOnlyMambaVisionHybridUNetV19D(cfg["base_ch"], dropout=float(cfg.get("model_dropout", 0.05)), ssm_kernel=int(cfg.get("ssm_kernel", 31)), attention_heads=int(cfg.get("attention_heads", 4)), input_channels=input_channels)
    if arch in ("v1_9e_convnext_pp", "raw_convnext_pp_unet_v19e", "convnext_pp"):
        return RawOnlyConvNeXtPPUNetV19E(cfg["base_ch"], dropout=float(cfg.get("model_dropout", 0.05)), input_channels=input_channels)
    if arch in ("v1_11_sguussm", "sguussm", "v11"):
        return RawOnlySGUSSMUNetV111(
            cfg["base_ch"],
            dropout=float(cfg.get("model_dropout", 0.05)),
            ssm_kernel=int(cfg.get("ssm_kernel", 31)),
            attention_heads=int(cfg.get("attention_heads", 4)),
            input_channels=input_channels,
        )
    raise ValueError(f"Unknown model_arch: {cfg.get('model_arch')}")
