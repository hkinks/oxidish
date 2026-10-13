"""Average the PPO training log in blocks of N prints: python ppo_sum.py logs/vision/p1.jsonl [N]"""
import json
import sys

rs = [json.loads(l) for l in open(sys.argv[1])]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
for i in range(0, len(rs), N):
    b = rs[i:i + N]
    f = lambda k: sum(r.get(k, 0) for r in b) / len(b)
    print(f"it {b[0]['it']:5d}-{b[-1]['it']:5d} lead {f('lead'):.3f} deaths/h {f('deaths_h'):5.1f} mass {f('mass'):6.0f} "
          f"rew {f('rew'):.4f} vl {f('vl'):.2f} kl {f('kl'):.4f} ent {f('ent'):.2f} ce {f('ce'):.2f} agree {f('agree'):.2f} clipf {f('clipf'):.2f}")
