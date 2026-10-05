"""Entity policy: a transformer over the visible objects (sim/entities.py tokens) with a pointer
move head. The interface matches rl.model.Policy (dists / logp_entropy / act / binary_eps), so the
trainers use either; `obs = "entity"` tells them which observation to feed.

Moves: 17 + NT logits = 16 headings, stay, and "head toward token j" for every token (a pointer:
query from the agent's state vs a key per token; padding tokens are masked). A pointer move aims at
the token's exact position at full speed (the mouse goes far beyond it along that direction), so
"go to the nearest pellet" is choosing a token, not drawing a heading out of pixels. Split and eject
are 2-way heads on the pooled state. The critic has its own encoder (value regression must not
reshape the policy's features).

Encoder: per-token Linear(FD -> d) plus a state token from the 16 scalars, `layers` pre-LN
transformer layers (d, `heads` heads, 2d feed-forward), padding masked; pooled = the state token's
output and the mean of the valid tokens, then an MLP to `width`.
"""

import math

import torch
from torch import nn
from torch.distributions import Categorical


class EntityEncoder(nn.Module):
    def __init__(self, fd: int, s: int, d: int, layers: int, heads: int, width: int):
        super().__init__()
        self.tok = nn.Linear(fd, d)
        self.state = nn.Linear(s, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(2 * d, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())

    def forward(self, ent, scal):
        valid = ent[..., 0] > 0.5                                     # [N,NT]
        x = torch.cat([self.state(scal)[:, None], self.tok(ent)], 1)  # [N,1+NT,d]
        pad = torch.cat([torch.zeros_like(valid[:, :1]), ~valid], 1)
        x = self.norm(self.tf(x, src_key_padding_mask=pad))
        tokens = x[:, 1:]
        vf = valid.unsqueeze(-1).to(x.dtype)
        mean = (tokens * vf).sum(1) / vf.sum(1).clamp(min=1)
        return self.mlp(torch.cat([x[:, 0], mean], -1)), tokens, valid


class EntityPolicy(nn.Module):
    def __init__(self, n_tok: int, fd: int, s: int = 16, d: int = 128, layers: int = 2, heads: int = 4,
                 width: int = 256, kind: str = "entity", **_):
        super().__init__()
        self.config = dict(kind="entity", n_tok=n_tok, fd=fd, s=s, d=d, layers=layers, heads=heads, width=width,
                           heads_out=(17 + n_tok, 2, 2), follow=None)
        self.obs = "entity"
        self.follow = None
        self.n_tok, self.d = n_tok, d
        self.enc = EntityEncoder(fd, s, d, layers, heads, width)
        self.critic = EntityEncoder(fd, s, d, layers, heads, width)
        self.move = nn.Linear(width, 17)
        self.query = nn.Linear(width, d)
        self.key = nn.Linear(d, d)
        self.split = nn.Linear(width, 2)
        self.eject = nn.Linear(width, 2)
        self.value = nn.Linear(width, 1)
        with torch.no_grad():                    # start calm: near-uniform moves, rarely split / eject
            for m in (self.move, self.query):
                m.weight.mul_(0.1)
            for h in (self.split, self.eject):
                h.weight.mul_(0.01)
                h.bias.copy_(torch.tensor([3.0, 0.0]))

    def forward(self, ent, scal):
        h, tokens, valid = self.enc(ent, scal)
        ptr = torch.einsum("nd,ntd->nt", self.query(h), self.key(tokens)) / math.sqrt(self.d)
        ptr = ptr.float().masked_fill(~valid, -1e9)
        moves = torch.cat([self.move(h).float(), ptr], -1)
        hv, _, _ = self.critic(ent, scal)
        return [moves, self.split(h), self.eject(h)], self.value(hv).squeeze(-1)

    binary_eps: float = 0.0
    EXPLORE_MIN_LOG_MASS = math.log(70) / 8

    def dists(self, ent, scal, amp: bool = False):
        if amp:
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                logits, value = self(ent, scal)
            logits, value = [l.float() for l in logits], value.float()
        else:
            logits, value = self(ent.float(), scal.float())
        out = [Categorical(logits=logits[0])]
        eps = self.binary_eps * (scal[:, 0] >= self.EXPLORE_MIN_LOG_MASS).float().unsqueeze(-1)
        for l in logits[1:]:
            if self.binary_eps > 0:
                out.append(Categorical(probs=(1 - eps) * l.softmax(-1) + eps / 2))
            else:
                out.append(Categorical(logits=l))
        return out, value

    def logp_entropy(self, dists, acts):
        logp = sum(d.log_prob(acts[:, j]) for j, d in enumerate(dists))
        ent = sum(d.entropy() for d in dists)
        return logp, ent

    @torch.no_grad()
    def act(self, ent, scal, greedy: bool = False, amp: bool = False):
        dists, value = self.dists(ent, scal, amp)
        acts = torch.stack([d.probs.argmax(-1) if greedy else d.sample() for d in dists], -1)
        logp, _ = self.logp_entropy(dists, acts)
        return acts, logp, value
