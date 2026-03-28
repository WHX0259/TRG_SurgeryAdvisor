"""ONNX Runtime 推理（CPU/GPU EP），与 torch 版 predict 输出字段对齐。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def load_onnx_session(
    model_path: str,
    providers: Optional[List[str]] = None,
) -> Any:
    import onnxruntime as ort

    if providers is None:
        providers = ["CPUExecutionProvider"]
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=so, providers=providers)


def predict_onnx(session: Any, x_np: np.ndarray) -> Dict[str, Any]:
    """x_np: (1,C,H,W) float32"""
    name = session.get_inputs()[0].name
    fused, _ = session.run(None, {name: x_np.astype(np.float32)})
    pred = int(np.argmax(fused, axis=-1).squeeze())
    return {
        "predicted_class": pred,
        "pseudo_expert_index": -1,
    }
