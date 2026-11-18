"""Export a trained GRU checkpoint to demo/model_weights.js for the in-browser demo.

Weights are stored as float16 (base64) to keep the file small (~1.8 MB for the 256x2 GRU);
the page decodes them to Float32Arrays. Also writes demo/reference.json (PyTorch logits for a
test prompt) so that the JavaScript implementation can be checked against PyTorch.

Usage:  python demo/export_weights.py [checkpoint-name]   (default: E6_gru_long if present, else E1_gru)
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.models import CharRNN  # noqa: E402


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else ("E6_gru_long" if (ROOT / "checkpoints/E6_gru_long.pt").exists() else "E1_gru")
    ck = torch.load(ROOT / "checkpoints" / f"{name}.pt"); c = ck["config"]; stoi = ck["stoi"]
    assert c["cell"] == "gru", "the browser demo implements the GRU cell"
    model = CharRNN(len(stoi), "gru", c["emb_dim"], c["hidden"], c["layers"], 0.0)
    model.load_state_dict(ck["state_dict"]); model.eval()
    tensors = {}
    for k, v in model.state_dict().items():
        a = v.numpy().astype(np.float16)
        tensors[k] = {"shape": list(a.shape), "b64": base64.b64encode(a.tobytes()).decode()}
    itos = [None] * len(stoi)
    for ch, i in stoi.items():
        itos[i] = ch
    res = json.loads((ROOT / "results" / ("final/E6_gru_long.json" if name == "E6_gru_long" else f"runs/{name}.json")).read_text())
    meta = {"name": name, "hidden": c["hidden"], "layers": c["layers"], "emb": c["emb_dim"], "vocab": itos,
            "test_bpc": round(res["test_bpc"], 3), "steps": c["steps"], "params": res["params"]}
    out = ROOT / "demo" / "model_weights.js"
    out.write_text("window.MODEL_META = " + json.dumps(meta) + ";\nwindow.MODEL_TENSORS = " + json.dumps(tensors) + ";\n")
    # reference logits (float16-rounded weights, so JS should match closely)
    with torch.no_grad():
        for k, v in model.state_dict().items():
            v.copy_(v.half().float())
        prompt = "It is a truth universally acknowledged"
        logits, _ = model(torch.tensor([[stoi[ch] for ch in prompt]]))
    (ROOT / "demo" / "reference.json").write_text(json.dumps({"prompt": prompt, "logits": logits[0, -1].tolist()}))
    print(f"exported {name}: {out.stat().st_size / 1e6:.2f} MB, test BPC {meta['test_bpc']}")


if __name__ == "__main__":
    main()
