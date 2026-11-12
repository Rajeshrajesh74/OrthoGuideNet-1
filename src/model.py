"""OrthoGuideNet: ASBS -> Spectral/Spatial DualMamba -> ASSF -> BAR -> multi-task decoder.

The Mamba-style selective state-space block is implemented in pure PyTorch,
so no `mamba_ssm` installation is needed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Selective state-space (Mamba-style) block, pure PyTorch (no mamba_ssm install needed) ----
class SelectiveSSM(nn.Module):
    def __init__(self, d, d_state=8, expand=2, d_conv=3):
        super().__init__()
        di = expand * d; self.di, self.n = di, d_state; self.dt_rank = max(1, d // 16)
        self.in_proj = nn.Linear(d, 2*di)
        self.conv = nn.Conv1d(di, di, d_conv, padding=d_conv-1, groups=di)
        self.x_proj = nn.Linear(di, self.dt_rank + 2*d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, di)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state+1).float()).repeat(di, 1))
        self.D = nn.Parameter(torch.ones(di))
        self.out_proj = nn.Linear(di, d)
    def forward(self, x):                                   # (B, L, d)
        B, L, _ = x.shape
        xi, z = self.in_proj(x).chunk(2, -1)
        xi = F.silu(self.conv(xi.transpose(1, 2))[..., :L].transpose(1, 2))
        dt, Bm, Cm = self.x_proj(xi).split([self.dt_rank, self.n, self.n], -1)
        dt = F.softplus(self.dt_proj(dt))                   # (B, L, di)
        A = -torch.exp(self.A_log)                          # (di, n)
        dA = torch.exp(dt.unsqueeze(-1) * A)                # (B, L, di, n)
        dBx = dt.unsqueeze(-1) * Bm.unsqueeze(2) * xi.unsqueeze(-1)
        h = x.new_zeros(B, self.di, self.n); ys = []
        for t in range(L):                                  # selective scan
            h = dA[:, t] * h + dBx[:, t]
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, 1) + self.D * xi
        return self.out_proj(y * F.silu(z))

class BiMamba(nn.Module):                                   # forward + backward scan (spectral branch)
    def __init__(self, d):
        super().__init__(); self.norm = nn.LayerNorm(d); self.ssm = SelectiveSSM(d)
    def forward(self, x):
        B = x.shape[0]; u = self.norm(x)
        y = self.ssm(torch.cat([u, u.flip(1)], 0))
        return x + y[:B] + y[B:].flip(1)

class SpatialMamba(nn.Module):                              # 4-direction scan: row/col x fwd/bwd (spatial branch)
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d); self.ssm = SelectiveSSM(d)
        self.ffn = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2*d), nn.GELU(), nn.Linear(2*d, d))
    def forward(self, x):                                   # (B, C, h, w)
        B, C, h, w = x.shape
        t = self.norm(x.flatten(2).transpose(1, 2))         # (B, hw, C) row-major
        tc = t.view(B, h, w, C).transpose(1, 2).reshape(B, h*w, C)   # column-major
        y = self.ssm(torch.cat([t, t.flip(1), tc, tc.flip(1)], 0))
        y1 = y[:B] + y[B:2*B].flip(1)
        y2 = (y[2*B:3*B] + y[3*B:].flip(1)).view(B, w, h, C).transpose(1, 2).reshape(B, h*w, C)
        t = t + (y1 + y2) / 4
        t = t + self.ffn(t)
        return t.transpose(1, 2).reshape(B, C, h, w)

# ---- Adaptive Spectral Band Selection (concrete / Gumbel-softmax selector) ----
class ASBS(nn.Module):
    def __init__(self, n_bands, k):
        super().__init__()
        self.nb, self.k = n_bands, k
        self.logits = nn.Parameter(torch.randn(k, n_bands) * 0.01)
        self.tau = 5.0
    def weights(self):
        if self.training:
            g = -torch.log(-torch.log(torch.rand_like(self.logits).clamp(1e-9, 1-1e-9)))
            w = F.softmax((self.logits + g) / self.tau, -1)
        else:
            w = F.one_hot(self.logits.argmax(-1), self.nb).float()
        idx = (w * torch.arange(self.nb, device=w.device)).sum(-1)
        order = idx.detach().argsort()                      # keep selected bands in wavelength order
        return w[order], idx[order] / (self.nb - 1)
    def forward(self, x):                                   # (B, nb, H, W) -> (B, k, H, W)
        w, idx = self.weights()
        return torch.einsum('kb,nbhw->nkhw', w, x), idx
    def regulariser(self):
        p = F.softmax(self.logits / self.tau, -1)
        ent = -(p * torch.log(p + 1e-9)).sum(-1).mean()     # push every selector to a single band
        coll = F.relu(p.sum(0) - 1).mean()                  # avoid picking the same band twice
        return 0.01 * ent + 1.0 * coll
    @torch.no_grad()
    def selected(self): return sorted(set(self.logits.argmax(-1).tolist()))

# ---- Spectral DualMamba branch ----
class SpectralBranch(nn.Module):
    def __init__(self, d=32, out_ch=64, depth=2, pool=8):
        super().__init__()
        self.pool = pool
        self.val = nn.Linear(1, d)
        self.pos = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))  # wavelength embedding
        self.blocks = nn.ModuleList([BiMamba(d) for _ in range(depth)])
        self.norm = nn.LayerNorm(d); self.score = nn.Linear(d, 1)
        self.proj = nn.Sequential(nn.Conv2d(d, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU())
    def forward(self, x, idx):
        x = F.avg_pool2d(x, self.pool); B, K, h, w = x.shape
        t = x.permute(0, 2, 3, 1).reshape(B*h*w, K, 1)
        t = self.val(t) + self.pos(idx[:, None])[None]
        for b in self.blocks: t = b(t)
        t = self.norm(t); a = F.softmax(self.score(t), 1)    # attention pooling over bands
        f = (a * t).sum(1).view(B, h, w, -1).permute(0, 3, 1, 2)
        return self.proj(f)

# ---- Adaptive Spectral-Spatial Fusion ----
class ASSF(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(2*c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(), nn.Conv2d(c, c, 1), nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(2*c, c//4, 1), nn.ReLU(), nn.Conv2d(c//4, c, 1), nn.Sigmoid())
        self.out = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU())
    def forward(self, fspec, fspat):
        cat = torch.cat([fspec, fspat], 1)
        g = self.gate(cat)
        return self.out((g * fspec + (1 - g) * fspat) * (1 + self.ca(cat)))

# ---- Boundary-Aware Refinement ----
class BAR(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.edge = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(), nn.Conv2d(c, 1, 1))
        self.fuse = nn.Sequential(nn.Conv2d(2*c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU())
    def forward(self, f):
        hp = (f - F.avg_pool2d(f, 3, 1, 1)).abs()           # parameter-free high-pass boundary cue
        b = self.edge(hp)
        att = torch.sigmoid(b)
        return f + self.fuse(torch.cat([f * att, hp], 1)), b

def cbr(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU())

class OrthoGuideNet(nn.Module):
    def __init__(self, n_bands, k=32, c=64, n_classes=7, p_drop=0.2):
        super().__init__()
        self.asbs = ASBS(n_bands, k)
        self.s1, self.s2, self.s3 = cbr(k, 32, 2), cbr(32, 64, 2), cbr(64, c, 2)   # /2, /4, /8
        self.spat = nn.Sequential(SpatialMamba(c), SpatialMamba(c))
        self.spec = SpectralBranch(out_ch=c)
        self.assf = ASSF(c)
        self.up1 = cbr(c + 64, 64); self.up2 = cbr(64 + 32, 48)
        self.bar = BAR(48)
        self.drop = nn.Dropout2d(p_drop)                    # also used for MC-dropout uncertainty
        self.seg_head = nn.Conv2d(48, n_classes, 1)
        self.bnd_head = nn.Conv2d(48, 1, 1)
        self.log_vars = nn.Parameter(torch.zeros(2))        # learnable task uncertainties (loss weighting)
    def forward(self, x):
        H, W = x.shape[-2:]
        xs, idx = self.asbs(x)
        f1 = self.s1(xs); f2 = self.s2(f1); f3 = self.s3(f2)
        fspat = self.spat(f3)
        fspec = self.spec(xs, idx)
        f = self.assf(fspec, fspat)
        f = self.up1(torch.cat([F.interpolate(f, scale_factor=2, mode='bilinear', align_corners=False), f2], 1))
        f = self.up2(torch.cat([F.interpolate(f, scale_factor=2, mode='bilinear', align_corners=False), f1], 1))
        f, b_aux = self.bar(f)
        f = self.drop(f)
        up = lambda t: F.interpolate(t, size=(H, W), mode='bilinear', align_corners=False)
        return up(self.seg_head(f)), up(self.bnd_head(f) + b_aux)


def build_model(n_bands, cfg):
    """Instantiate OrthoGuideNet from the configuration."""
    return OrthoGuideNet(n_bands, k=cfg.model.k_bands, c=cfg.model.embed_dim,
                         n_classes=cfg.n_classes, p_drop=cfg.model.dropout)
