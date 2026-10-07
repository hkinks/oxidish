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


def main(ckpt, out):
    model, ck = load_model(ckpt, device="cpu")
    model = model.float().eval().to(memory_format=torch.contiguous_format)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    v = torch.randint(0, 255, (2, 2, 256, 256), dtype=torch.uint8)
    w = torch.randint(0, 255, (2, 2, 128, 128), dtype=torch.uint8)
    with torch.no_grad():
        ref = model(v, w)
        ts = torch.jit.trace(model, (v, w))
        ts.save(out + ".pt")
        got = ts(v, w)
    print("torchscript max diff", max((a - b).abs().max().item() for a, b in zip(ref, got)))
    torch.onnx.export(model, (v, w), out + ".onnx", input_names=["view", "wide"],
                      output_names=["heat", "split", "eject", "value"], opset_version=17,
                      dynamic_axes={k: {0: "batch"} for k in ("view", "wide", "heat", "split", "eject", "value")},
                      dynamo=False)
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(out + ".onnx", providers=["CPUExecutionProvider"])
        o = sess.run(None, {"view": v.numpy(), "wide": w.numpy()})
        print("onnx max diff", max(np.abs(a.numpy() - b).max() for a, b in zip(ref, o)))
    except ImportError:
        print("onnxruntime not installed; ONNX parity not checked")
    json.dump(dict(cfg=ck["cfg"], split_thr=ck.get("split_thr", 0.5), px=256, heat=64, metrics=ck.get("metrics")),
              open(out + ".json", "w"), indent=1)
    print("saved", out + ".pt/.onnx/.json")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
