"""单样本推理（仅返回预测类别索引，不对外暴露概率）。"""
from __future__ import annotations

from typing import Any, Dict, Union

import numpy as np
import torch


@torch.inference_mode()
def predict_one(model: torch.nn.Module, x: Union[torch.Tensor, "np.ndarray"]) -> Dict[str, Any]:
    device = next(model.parameters()).device
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = x.to(device, non_blocking=True)
    out = model(x)
    fused = out["output"]
    pred = int(fused.argmax(dim=1).view(-1)[0].item())
    return {
        "predicted_class": pred,
        "pseudo_expert_index": int(out["pseudo_labels"].view(-1)[0].item())
        if out["pseudo_labels"].numel() > 0
        else -1,
    }
