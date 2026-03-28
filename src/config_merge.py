"""合并训练时保存的 config.json（Multi_Modal_MoE/PDCC 输出目录），便于与 validate 对齐 img_size 等。"""
from __future__ import annotations

import json
import os
from typing import Any, Dict

import yaml


def load_app_config(yaml_path: str) -> Dict[str, Any]:
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tj = cfg.get("training_config_json")
    if not tj or not os.path.isfile(tj):
        return cfg

    with open(tj, encoding="utf-8") as f:
        train_cfg = json.load(f)

    meta = {
        "model_type": train_cfg.get("model_type"),
        "fold": train_cfg.get("fold"),
        "img_size": train_cfg.get("img_size"),
        "slice_path": train_cfg.get("slice_path"),
    }
    cfg["_training_meta"] = meta

    if cfg.get("sync_image_size_from_training_json", True) and train_cfg.get("img_size") is not None:
        cfg["image_size"] = int(train_cfg["img_size"])

    return cfg
