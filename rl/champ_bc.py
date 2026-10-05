"""Fine-tune an entity net toward the champions' moves (rl/champ_data.py), anchored to itself.

    uv run python -m rl.champ_bc --init runs/rl/ffa1/eval/g002_m3.pt --out runs/rl/champ_bc.pt

Loss per batch: -log p(champion's move) - log p(split label) + beta * KL(original || new) on the same
states (keeps what the net already does well where the champion disagrees only mildly). Reports the
held-out move accuracy (exact heading of 16, or the pointer the net picks decoded to the same heading
is NOT counted: headings only) before and after.
"""

import argparse

import torch

from rl.common import ROOT
from rl.ppo_sim import from_checkpoint, frozen, setup_cuda


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--init", required=True)
    p.add_argument("--data", default=str(ROOT / "runs" / "rl" / "champ_data.pt"))
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta", type=float, default=0.5, help="KL anchor to the initial net")
    p.add_argument("--vram-gb", type=float, default=4.0)
    args = p.parse_args()
    setup_cuda(args.vram_gb)
    dev = "cuda"
    d = torch.load(args.data, map_location=dev)
    ent, scal, act = d["ent"], d["scal"].float(), d["act"]
    n = len(act)
    perm = torch.randperm(n, device=dev)
    val, tr = perm[: n // 20], perm[n // 20:]
    model = from_checkpoint(args.init, dev)
    anchor = frozen(from_checkpoint(args.init, dev))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    def accuracy() -> float:
        model.eval()
        hits = 0
        with torch.no_grad():
            for i in range(0, len(val), 512):
                ix = val[i:i + 512]
                dists, _ = model.dists(ent[ix], scal[ix])
                hits += (dists[0].probs.argmax(-1) == act[ix, 0]).sum().item()
        model.train()
        return hits / len(val)

    print(f"{n} samples; held-out move accuracy before {accuracy():.1%}", flush=True)
    for step in range(1, args.steps + 1):
        ix = tr[torch.randint(0, len(tr), (args.batch,), device=dev)]
        dists, _ = model.dists(ent[ix], scal[ix])
        with torch.no_grad():
            adists, _ = anchor.dists(ent[ix], scal[ix])
        nll = -(dists[0].log_prob(act[ix, 0]).mean() + dists[1].log_prob(act[ix, 1]).mean())
        kl = sum(torch.distributions.kl_divergence(a, b) for a, b in zip(adists, dists)).mean()
        loss = nll + args.beta * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        if step % 100 == 0:
            print(f"  step {step}: nll {nll.item():.3f}  kl {kl.item():.3f}  held-out accuracy {accuracy():.1%}",
                  flush=True)
    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    ck["model"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(ck, args.out)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
