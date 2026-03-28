"""Gradio / CLI 共用的单次推理管线。"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

from PIL import Image

from src.preprocess import preprocess_pair
from src.runtime import backend_name
from src.visualize import highlight_region


def run_inference(
    original: Image.Image,
    seg: Image.Image,
    engine: Any,
    cfg: Dict[str, Any],
    input_mode: Optional[str] = None,
) -> Tuple[Image.Image, str, str, Dict[str, Any]]:
    """
    返回: (高亮图, markdown 文本, JSON 字符串, 结构化 dict)
    """
    image_size = int(cfg.get("image_size", 224))
    mode = (input_mode if input_mode is not None else cfg.get("input_mode", "masked")) or "masked"
    x, _gray_u8, mask_bin = preprocess_pair(
        original,
        seg,
        image_size=image_size,
        input_mode=mode,
        in_chan=int(cfg.get("in_chan", 3)),
        use_cmel_normalize=bool(cfg.get("use_cmel_normalize", True)),
    )
    if backend_name(cfg) == "onnx":
        from src.onnx_backend import predict_onnx

        result = predict_onnx(engine, x)
    else:
        from src.predict import predict_one

        result = predict_one(engine, x)
    class_names = list(cfg.get("class_names", ["类别0", "类别1"]))
    pred_idx = int(result["predicted_class"])
    label = class_names[pred_idx] if pred_idx < len(class_names) else f"类别{pred_idx}"

    vis = highlight_region(original, mask_bin)
    table: Dict[str, Any] = {
        "backend": backend_name(cfg),
        "predicted_class": pred_idx,
        "predicted_label": label,
        "input_mode_used": mode,
        "image_size": image_size,
    }
    if result.get("pseudo_expert_index", -1) >= 0:
        table["pseudo_expert_index"] = result["pseudo_expert_index"]
    if "_training_meta" in cfg:
        table["training_meta"] = cfg["_training_meta"]

    md = (
        f"### 预测结果：{label}\n\n"
        "说明：以上为模型输出的类别标签，不能替代临床决策。"
    )

    return vis, md, json.dumps(table, ensure_ascii=False, indent=2), table
