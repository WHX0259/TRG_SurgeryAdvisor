#!/usr/bin/env python3
"""命令行单次推理：原图 + 掩膜 → 高亮图 + result.json（与 Gradio 共用管线）。"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _default_config_path() -> str:
    env = os.environ.get("TRG_CONFIG")
    if env and os.path.isfile(env):
        return env
    for name in ("config.yaml", "config.example.yaml"):
        p = os.path.join(ROOT, name)
        if os.path.isfile(p):
            return p
    raise SystemExit("未找到 config.yaml，请复制 config.example.yaml 或设置 TRG_CONFIG")


def main() -> None:
    parser = argparse.ArgumentParser(description="TRG SurgeryAdvisor 命令行推理")
    parser.add_argument("--config", default=None, help="YAML 配置路径，默认 config.yaml / TRG_CONFIG")
    parser.add_argument("--orig", required=True, help="原图 PNG/JPG")
    parser.add_argument("--mask", required=True, help="分割掩膜 PNG")
    parser.add_argument("--out", default=os.path.join(ROOT, "outputs"), help="输出目录")
    parser.add_argument("--mode", choices=["masked", "full"], default=None, help="覆盖配置中的 input_mode")
    args = parser.parse_args()

    cfg_path = args.config or _default_config_path()
    from src.config_merge import load_app_config
    from src.pipeline import run_inference
    from src.runtime import load_engine

    cfg = load_app_config(cfg_path)
    engine = load_engine(cfg)

    os.makedirs(args.out, exist_ok=True)
    from PIL import Image

    original = Image.open(args.orig)
    seg = Image.open(args.mask)
    vis, _md, jstr, _table = run_inference(original, seg, engine, cfg, input_mode=args.mode)
    stem = os.path.splitext(os.path.basename(args.orig))[0]
    vis_path = os.path.join(args.out, f"{stem}_highlight.png")
    json_path = os.path.join(args.out, f"{stem}_result.json")
    vis.save(vis_path)
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(jstr)
    print(vis_path)
    print(json_path)


if __name__ == "__main__":
    main()
