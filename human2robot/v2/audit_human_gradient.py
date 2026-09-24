"""Replay initial human minibatch to verify a nonzero shared-encoder gradient.
This is an audit, not a claim of improved downstream performance.
"""

import argparse, json, hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from v2.policy import Policy
from v2.train import load_human

p = argparse.ArgumentParser()
p.add_argument("--human")
p.add_argument("--out")
a = p.parse_args()
torch.set_num_threads(3)
human = load_human(a.human, "cpu")
result = []
for seed in [17, 29]:
    for condition in ["explicit", "implicit_joint"]:
        torch.manual_seed(seed)
        model = Policy()
        digest = hashlib.sha256(
            b"".join(v.detach().numpy().tobytes() for v in model.state_dict().values())
        ).hexdigest()
        idx = torch.tensor(
            np.random.default_rng(seed + 777).integers(len(human["images"]), size=32)
        )
        motion, image = model.human(human["images"][idx])
        if condition == "explicit":
            mask = human["mask"][idx]
            loss = (
                0.1
                * ((motion - human["motion"][idx] / 0.1).square() * mask).sum()
                / mask.sum().clamp_min(1)
            )
        else:
            cur = F.interpolate(
                human["images"][idx].float().permute(0, 3, 1, 2) / 255,
                (32, 32),
                mode="area",
            )
            future = F.interpolate(
                human["future"][idx].float().permute(0, 3, 1, 2) / 255,
                (32, 32),
                mode="area",
            )
            loss = F.mse_loss(image, future - cur)
        loss.backward()
        norm = (
            sum(
                float(p.grad.square().sum())
                for p in model.encoder.parameters()
                if p.grad is not None
            )
            ** 0.5
        )
        assert norm > 0 and np.isfinite(norm)
        result.append(
            {
                "seed": seed,
                "condition": condition,
                "weighted_human_loss": loss.item(),
                "encoder_gradient_l2": norm,
                "initial_hash": digest,
                "audit": "CPU replay of first human minibatch; not performance evidence",
            }
        )
Path(a.out).write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
