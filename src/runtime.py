"""按配置加载 PyTorch 模型或 ONNX Runtime 会话。"""
from __future__ import annotations

import os
from typing import Any, Dict

def load_engine(cfg: Dict[str, Any]) -> Any:
    backend = str(cfg.get("backend", "torch")).lower().strip()
    if backend == "onnx":
        from src.onnx_backend import load_onnx_session

        path = cfg.get("onnx_model_path") or cfg.get("onnx_path")
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(
                "backend=onnx 时请在 config.yaml 中设置有效的 onnx_model_path（指向 .onnx 文件）"
            )
        providers = cfg.get("onnx_providers")
        if providers is None:
            providers = ["CPUExecutionProvider"]
        return load_onnx_session(path, list(providers))

    from src.model_loader import build_model, resolve_device

    return build_model(cfg, resolve_device(cfg))


def backend_name(cfg: Dict[str, Any]) -> str:
    return str(cfg.get("backend", "torch")).lower().strip()
