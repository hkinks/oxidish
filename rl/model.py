"""Actor-critic CNN: grid + scalars -> logits for (move, split, eject) and a value.

Shapes are configurable (plain bot: rl.obs; guided bot: rl.guided) and stored in the
checkpoint as `config`. With `follow` set, move index `follow` means "execute the
champion's action": the split/eject heads then don't act, so they are excluded from
that step's log-probability and entropy.
"""

import math

import torch
from torch import nn
from torch.distributions import Categorical

from .actions import HEADS as PLAIN_HEADS
from .obs import C as PLAIN_C, G, S as PLAIN_S


DEFAULT_CONVS = ((32, 1), (64, 2), (64, 2))


class Policy(nn.Module):
    """Architecture genes (rl/evo.py mutates them; defaults = the original net):
    convs: (channels, stride) per 3x3 conv layer; pool: average-pool the input grid by this
    factor first; in_mask: grid channels the net sees (others are dropped, None = all);
    s_mask: likewise for the scalars; shared: policy and value share the conv trunk."""

    def __init__(self, width: int = 256, c: int = PLAIN_C, s: int = PLAIN_S,
                 heads: tuple = PLAIN_HEADS, follow: int | None = None, g: int = G,
                 convs: tuple = DEFAULT_CONVS, pool: int = 1, in_mask: tuple | None = None,
                 s_mask: tuple | None = None, shared: bool = False):
        super().__init__()
        convs = tuple(tuple(x) for x in convs)
        self.config = dict(width=width, c=c, s=s, heads=tuple(heads), follow=follow, g=g)
        if convs != DEFAULT_CONVS or pool != 1 or in_mask is not None or s_mask is not None or shared:
            self.config.update(convs=convs, pool=pool, in_mask=in_mask, s_mask=s_mask, shared=shared)
        self.follow = follow
        self.pool = pool
        self.shared = shared
        self.register_buffer("in_idx", torch.tensor(in_mask if in_mask is not None else range(c)), persistent=False)
        self.register_buffer("s_idx", torch.tensor(s_mask if s_mask is not None else range(s)), persistent=False)
        cin, sin = len(self.in_idx), len(self.s_idx)

        def trunk():
            layers, ch = [], cin
            for out, stride in convs:
                layers += [nn.Conv2d(ch, out, 3, stride=stride, padding=1), nn.ReLU()]
                ch = out
            return nn.Sequential(*layers, nn.Flatten())

        with torch.no_grad():
            flat = trunk()(torch.zeros(1, cin, g // pool, g // pool)).shape[1]

        def body():
            return nn.Sequential(nn.Linear(flat + sin, width), nn.ReLU(),
                                 nn.Linear(width, width), nn.ReLU())

        self.conv, self.body = trunk(), body()
        self.heads = nn.ModuleList(nn.Linear(width, n) for n in heads)
        # Separate critic: value regression (large, noisy targets) must not reshape the
        # features the policy relies on.
        self.critic_conv, self.critic_body = (None if shared else trunk()), body()
        self.value = nn.Linear(width, 1)

    def init_follow(self, bias: float = 7.0) -> None:
        """Start as the champion: FOLLOW gets a large logit, other moves ~uniform below it,
        and on the rare own moves split/eject default firmly to "no" (a random split halves us)."""
        with torch.no_grad():
            self.heads[0].weight.mul_(0.01)
            self.heads[0].bias.zero_()
            self.heads[0].bias[self.follow] = bias
            for head in self.heads[1:]:
                head.weight.mul_(0.01)
                head.bias.copy_(torch.tensor([bias, 0.0]))

    def forward(self, grid: torch.Tensor, scal: torch.Tensor):
        if len(self.in_idx) != grid.shape[1]:
            grid = grid.index_select(1, self.in_idx)
        if len(self.s_idx) != scal.shape[1]:
            scal = scal.index_select(1, self.s_idx)
        if self.pool > 1:
            grid = nn.functional.avg_pool2d(grid, self.pool)
        f = self.conv(grid)
        h = self.body(torch.cat([f, scal], dim=1))
        fv = f if self.shared else self.critic_conv(grid)
        hv = self.critic_body(torch.cat([fv, scal], dim=1))
        return [head(h) for head in self.heads], self.value(hv).squeeze(-1)

    # Exploration floor for the rare binary actions (split, eject): while our mass is at
    # least EXPLORE_MIN_MASS each keeps >= binary_eps/2 probability, so training keeps
    # trying them where they can matter (small cells can't afford random ejects).
    # 0 at evaluation time.
    binary_eps: float = 0.0
    EXPLORE_MIN_LOG_MASS = math.log(70) / 8   # scal[0] = log(total mass) / 8; splits need a 35-mass piece

    def dists(self, grid, scal, amp: bool = False):
        """amp: run the network in bfloat16 (2-3x faster training on the GPU); the
        distributions and value are still built in float32, so PPO ratios stay exact."""
        if amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, value = self(grid, scal)
            logits, value = [l.float() for l in logits], value.float()
        else:
            logits, value = self(grid, scal)
        out = [Categorical(logits=logits[0])]
        eps = self.binary_eps * (scal[:, 0] >= self.EXPLORE_MIN_LOG_MASS).float().unsqueeze(-1)
        for l in logits[1:]:
            if self.binary_eps > 0:
                out.append(Categorical(probs=(1 - eps) * l.softmax(-1) + eps / 2))
            else:
                out.append(Categorical(logits=l))
        return out, value

    def logp_entropy(self, dists, acts):
        """Joint log-prob and entropy; split/eject don't count on FOLLOW steps."""
        own = torch.ones_like(acts[:, 0], dtype=torch.float32)
        if self.follow is not None:
            own = (acts[:, 0] != self.follow).float()
        logp = dists[0].log_prob(acts[:, 0])
        ent = dists[0].entropy()
        for j, d in enumerate(dists[1:], start=1):
            logp = logp + own * d.log_prob(acts[:, j])
            ent = ent + own * d.entropy()
        return logp, ent

    @torch.no_grad()
    def act(self, grid, scal, greedy: bool = False, amp: bool = False):
        dists, value = self.dists(grid, scal, amp)
        acts = torch.stack([d.probs.argmax(-1) if greedy else d.sample() for d in dists], -1)
        logp, _ = self.logp_entropy(dists, acts)
        return acts, logp, value
