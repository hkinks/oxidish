"""Vision policy: IMPALA-ResNet trunk on the 256 px view stack + small trunk on the 128 px wide stack.

Inputs  view u8[B,2,256,256], wide u8[B,2,128,128] (current + previous decision frame), converted on the GPU.
Outputs heat  [B,H,H] logits over the view (H = 64: one cell = 4 view pixels; argmax = cursor target),
        split, eject logits [B], value [B] (for M4).

The first stage keeps full resolution (3x3 conv at 256 px, then a 2x2 max pool), so the 2-3 px size differences that
decide eat/flee reach the deeper layers. Max pooling keeps the strongest edge response.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.a = nn.Conv2d(c, c, 3, padding=1)
        self.b = nn.Conv2d(c, c, 3, padding=1)
        nn.init.zeros_(self.b.weight)
        nn.init.zeros_(self.b.bias)

    def forward(self, x):
        return x + self.b(F.relu(self.a(F.relu(x))))


class Stage(nn.Module):
    """conv -> 2x2 max pool -> n residual blocks."""

    def __init__(self, cin, cout, n_res):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.res = nn.Sequential(*[Res(cout) for _ in range(n_res)])

    def forward(self, x):
        return self.res(F.max_pool2d(self.conv(x), 2))


class VisionNet(nn.Module):
    def __init__(self, width=1.0, heat=64, px=256, gru=0):
        super().__init__()
        self.gru = gru
        w = [max(8, int(round(c * width / 8)) * 8) for c in (16, 32, 64, 128)]
        self.px, self.heat = px, heat
        self.s1 = Stage(2, w[0], 1)            # 256 -> 128
        self.s2 = Stage(w[0], w[1], 2)         # 128 -> 64
        self.s3 = Stage(w[1], w[2], 2)         # 64 -> 32
        self.s4 = Stage(w[2], w[3], 1)         # 32 -> 16
        v = [w[0], w[1], w[2]]                 # wide trunk, 128 -> 64 -> 32 -> 16
        self.w1 = Stage(2, v[0], 1)
        self.w2 = Stage(v[0], v[1], 1)
        self.w3 = Stage(v[1], v[2], 1)
        d = w[2]
        self.fuse = nn.Conv2d(w[3] + v[2], d, 1)
        self.l3 = nn.Conv2d(w[2], d, 1)
        self.l2 = nn.Conv2d(w[1], d, 1)
        self.dec3 = nn.Conv2d(d, d, 3, padding=1)
        self.dec2 = nn.Conv2d(d, d, 3, padding=1)
        self.out = nn.Conv2d(d, 1, 1)
        self.spatial = nn.Conv2d(w[3] + v[2], 8, 1)
        fdim = 8 * 16 * 16 + 2 * (w[3] + v[2])
        self.mlp = nn.Sequential(nn.Linear(fdim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.head = nn.Linear(256, 3)         # split, eject, value
        self.register_buffer("scale", torch.tensor(1 / 255.0))
        if gru:
            # M5 memory: a GRU cell on the 256-d feature. Its output enters the split/eject/value head and (as a channel bias at the
            # 16x16 decoder input) the heat map, through zero-initialised projections: a fresh GRU model == the non-recurrent one.
            self.cell = nn.GRUCell(256, gru)
            self.g_head = nn.Linear(gru, 256)
            self.g_dec = nn.Linear(gru, w[2])
            for q in (self.g_head, self.g_dec):
                nn.init.zeros_(q.weight)
                nn.init.zeros_(q.bias)

    def encode(self, view, wide):
        x = (view.to(torch.float32) * self.scale).contiguous(memory_format=torch.channels_last)
        y = (wide.to(torch.float32) * self.scale).contiguous(memory_format=torch.channels_last)
        f2 = self.s2(self.s1(x))
        f3 = self.s3(f2)
        f4 = self.s4(f3)
        g = self.w3(self.w2(self.w1(y)))
        z = F.relu(torch.cat([f4, g], 1))
        h0 = F.relu(self.fuse(z))
        s = F.relu(self.spatial(z)).flatten(1)
        pooled = torch.cat([z.mean((2, 3)), z.amax((2, 3))], 1)
        feat = self.mlp(torch.cat([s, pooled], 1))
        return dict(f2=f2, f3=f3, h0=h0, feat=feat)

    def decode(self, e, g=None):
        h = e["h0"]
        if g is not None:
            h = h + self.g_dec(g)[:, :, None, None]
        h = F.interpolate(h, scale_factor=2, mode="nearest") + self.l3(F.relu(e["f3"]))
        h = F.relu(self.dec3(F.relu(h)))
        h = F.interpolate(h, scale_factor=2, mode="nearest") + self.l2(F.relu(e["f2"]))
        h = F.relu(self.dec2(F.relu(h)))
        return self.out(h).squeeze(1)

    def heads(self, feat, g=None):
        o = self.head(feat + self.g_head(g)) if g is not None else self.head(feat)
        return o[:, 0], o[:, 1], o[:, 2]

    def forward_rec(self, view, wide, h):
        """One decision with hidden state h [B, gru] (zeros = new life) -> heat, split, eject, value, h_new."""
        e = self.encode(view, wide)
        with torch.autocast("cuda", enabled=False):   # fp32 recurrence: bf16 rounding drifts between rollout and BPTT replay
            hn = self.cell(e["feat"].float(), h.float())
        heat = self.decode(e, hn)
        sp, ej, v = self.heads(e["feat"], hn)
        return heat, sp, ej, v, hn

    def forward(self, view, wide):
        if self.gru:   # stateless use (hidden state = zeros): heat, split, eject, value
            B = view.shape[0]
            return self.forward_rec(view, wide, torch.zeros(B, self.gru, device=view.device, dtype=torch.float32))[:4]
        e = self.encode(view, wide)
        sp, ej, v = self.heads(e["feat"])
        return self.decode(e), sp, ej, v

    def _old_forward(self, view, wide):
        x = (view.to(torch.float32) * self.scale).contiguous(memory_format=torch.channels_last)
        y = (wide.to(torch.float32) * self.scale).contiguous(memory_format=torch.channels_last)
        f2 = self.s2(self.s1(x))
        f3 = self.s3(f2)
        f4 = self.s4(f3)
        g = self.w3(self.w2(self.w1(y)))
        z = F.relu(torch.cat([f4, g], 1))
        h = F.relu(self.fuse(z))
        h = F.interpolate(h, scale_factor=2, mode="nearest") + self.l3(F.relu(f3))
        h = F.relu(self.dec3(F.relu(h)))
        h = F.interpolate(h, scale_factor=2, mode="nearest") + self.l2(F.relu(f2))
        h = F.relu(self.dec2(F.relu(h)))
        heat = self.out(h).squeeze(1)          # [B,64,64]
        s = F.relu(self.spatial(z)).flatten(1)
        pooled = torch.cat([z.mean((2, 3)), z.amax((2, 3))], 1)
        o = self.head(self.mlp(torch.cat([s, pooled], 1)))
        return heat, o[:, 0], o[:, 1], o[:, 2]


def decode_target(heat, px=256):
    """Heat logits [B,H,H] -> pixel target [B,2] (u, v): argmax cell refined by a 3x3 soft-argmax around the peak."""
    B, H, _ = heat.shape
    cell = px / H
    idx = heat.flatten(1).argmax(1)
    iy, ix = idx // H, idx % H
    p = F.pad(heat.float(), (1, 1, 1, 1), value=-1e4)
    off = torch.arange(-1, 2, device=heat.device)
    ys = iy[:, None] + 1 + off[None, :]
    xs = ix[:, None] + 1 + off[None, :]
    patch = p[torch.arange(B, device=heat.device)[:, None, None], ys[:, :, None], xs[:, None, :]]  # [B,3,3]
    wgt = torch.softmax(patch.flatten(1), 1).view(B, 3, 3)
    dx = (wgt.sum(1) * off[None, :]).sum(1)
    dy = (wgt.sum(2) * off[None, :]).sum(1)
    u = (ix.float() + 0.5 + dx) * cell
    v = (iy.float() + 0.5 + dy) * cell
    return torch.stack([u, v], 1).clamp(0, px)


def heat_target(tu, tv, H=64, px=256, sigma=1.0):
    """Soft Gaussian target distribution [B,H,H] around the teacher pixel (cell centres at (i+0.5) * px/H)."""
    cell = px / H
    c = (torch.arange(H, device=tu.device, dtype=torch.float32) + 0.5) * cell
    gx = torch.exp(-0.5 * ((c[None, :] - tu[:, None]) / (sigma * cell)) ** 2)
    gy = torch.exp(-0.5 * ((c[None, :] - tv[:, None]) / (sigma * cell)) ** 2)
    t = gy[:, :, None] * gx[:, None, :]
    return t / t.sum((1, 2), keepdim=True).clamp_min(1e-9)
