"""Export a checkpoint for M3: TorchScript and ONNX (inputs u8 view[B,2,256,256], wide[B,2,128,128];
outputs heat[B,64,64] logits, split, eject logits, value).

    python rl/vision/export.py runs/vision/bc2/final.pt runs/vision/export/vision_bc
Decoding (M3 must do the same): cell = argmax heat; u,v = (cell + 0.5 + 3x3 soft-argmax offset) * 4 px (see
model.decode_target); split = sigmoid(split) > split_thr (stored in <out>.json), eject = sigmoid(eject) > 0.5.
"""
import json
import os
import sys

import numpy as np
import torch

from evalval import load_model


class RecWrap(torch.nn.Module):
    """Recurrent export (M5): inputs view, wide, h [B, gru]; outputs heat, split, eject, value, h_out."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, view, wide, h):
        return self.m.forward_rec(view, wide, h)


def main(ckpt, out):
    model, ck = load_model(ckpt, device="cpu")
    model = model.float().eval().to(memory_format=torch.contiguous_format)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    v = torch.randint(0, 255, (2, 2, 256, 256), dtype=torch.uint8)
    w = torch.randint(0, 255, (2, 2, 128, 128), dtype=torch.uint8)
    gru = getattr(model, "gru", 0)
    h = torch.randn(2, gru) * 0.5 if gru else None
    if gru:
        model = RecWrap(model).eval()
    args = (v, w, h) if gru else (v, w)
    with torch.no_grad():
        ref = model(*args)
        ts = torch.jit.trace(model, args)
        ts.save(out + ".pt")
        got = ts(*args)
    print("torchscript max diff", max((a - b).abs().max().item() for a, b in zip(ref, got)))
    ins = ["view", "wide"] + (["h"] if gru else [])
    outs = ["heat", "split", "eject", "value"] + (["h_out"] if gru else [])
    torch.onnx.export(model, args, out + ".onnx", input_names=ins, output_names=outs, opset_version=17,
                      dynamic_axes={k: {0: "batch"} for k in ins + outs}, dynamo=False)
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(out + ".onnx", providers=["CPUExecutionProvider"])
        feed = {"view": v.numpy(), "wide": w.numpy()}
        if gru:
            feed["h"] = h.numpy()
        o = sess.run(None, feed)
        print("onnx max diff", max(np.abs(a.numpy() - b).max() for a, b in zip(ref, o)))
    except ImportError:
        print("onnxruntime not installed; ONNX parity not checked")
    json.dump(dict(cfg=ck["cfg"], split_thr=ck.get("split_thr", 0.5), px=256, heat=64, metrics=ck.get("metrics")),
              open(out + ".json", "w"), indent=1)
    print("saved", out + ".pt/.onnx/.json")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
