"""从 PDCC 加载 ImageBaseClusterDistancePlusGatingModel 与权重。"""
from __future__ import annotations

import os
import sys
from collections import OrderedDict
from typing import Any, Dict

import torch


def clean_state_dict(state_dict: Dict[str, Any]) -> OrderedDict:
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v
    return new_state_dict


def ensure_pdcc_on_path(pdcc_root: str) -> None:
    root = os.path.abspath(pdcc_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def build_model(cfg: dict, device: torch.device) -> torch.nn.Module:
    ensure_pdcc_on_path(cfg["pdcc_root"])
    from lib.model.CMEL import ImageBaseClusterDistancePlusGatingModel  # noqa: WPS433

    model = ImageBaseClusterDistancePlusGatingModel(
        in_chan=int(cfg.get("in_chan", 3)),
        num_experts=int(cfg["num_experts"]),
        nlabels=int(cfg["num_class"]),
        num_iterations=int(cfg.get("num_iterations", 20)),
        cluster_init_type=str(cfg.get("cluster_init_type", "kmeans++")),
        k=int(cfg.get("k", 8)),
    )
    model.to(device)
    ckpt = cfg["checkpoint_path"]
    if ckpt and os.path.isfile(ckpt):
        try:
            blob = torch.load(ckpt, map_location=device, weights_only=False)
        except TypeError:
            blob = torch.load(ckpt, map_location=device)
        if isinstance(blob, dict) and "state_dict" in blob:
            state = blob["state_dict"]
        elif isinstance(blob, dict) and "model" in blob:
            state = blob["model"]
        else:
            state = blob
        model.load_state_dict(clean_state_dict(state), strict=False)
    else:
        raise FileNotFoundError(f"未找到权重: {ckpt}")

    model.if_init = True
    model.eval()
    return model


def resolve_device(cfg: dict) -> torch.device:
    d = cfg.get("device")
    if d:
        return torch.device(d)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
