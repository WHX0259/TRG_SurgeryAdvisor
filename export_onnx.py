#!/usr/bin/env python3
"""
将 ImageBase 推理子图导出为 ONNX，并可选 ONNX Runtime 动态 INT8 量化（权重）。

需要：PyTorch + PDCC（与日常训练环境相同）。导出完成后，推理机可仅安装 onnxruntime 使用 CPU。

剪枝：结构化剪枝会改变计算图，需重新训练；本脚本不做「伪剪枝」。轻量部署请用 --quantize。
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _default_config() -> str:
    for name in ("config.yaml", "config.example.yaml"):
        p = os.path.join(ROOT, name)
        if os.path.isfile(p):
            return p
    raise SystemExit("未找到 config.yaml")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="YAML 配置")
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "onnx_export"), help="输出目录")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--quantize", action="store_true", help="生成 ONNX Runtime 动态量化 INT8 模型")
    parser.add_argument("--verify", action="store_true", help="对比 Wrapper 与完整模型 fused 输出（max abs diff）")
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F

    from src.config_merge import load_app_config
    from src.inference_wrapper import ImageBaseInferenceWrapper
    from src.model_loader import build_model, resolve_device

    cfg_path = args.config or _default_config()
    cfg = load_app_config(cfg_path)
    os.makedirs(args.out_dir, exist_ok=True)

    fp32_path = os.path.join(args.out_dir, "trg_imagebase_infer_fp32.onnx")
    quant_path = os.path.join(args.out_dir, "trg_imagebase_infer_int8.onnx")

    device = torch.device("cpu")
    full = build_model(cfg, device)
    full.eval()
    wrapper = ImageBaseInferenceWrapper(full).eval()

    if args.verify:
        c = int(cfg.get("in_chan", 3))
        h = w = int(cfg.get("image_size", 224))
        x = torch.randn(1, c, h, w, device=device)
        with torch.no_grad():
            out_full = full(x)["output"]
            out_wrap, _ = wrapper(x)
            d = (F.softmax(out_full, dim=1) - F.softmax(out_wrap, dim=1)).abs().max().item()
        print(f"[verify] max abs diff (softmax space): {d:.6e}")

    dummy = torch.randn(1, int(cfg.get("in_chan", 3)), int(cfg.get("image_size", 224)), int(cfg.get("image_size", 224)))
    torch.onnx.export(
        wrapper,
        dummy,
        fp32_path,
        input_names=["image"],
        output_names=["fused_logits", "expert_outputs"],
        dynamic_axes={
            "image": {0: "batch"},
            "fused_logits": {0: "batch"},
            "expert_outputs": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"已导出 FP32 ONNX: {fp32_path}")

    if args.quantize:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
        except ImportError as e:
            raise SystemExit("量化需要: pip install onnx onnxruntime") from e

        quantize_dynamic(
            model_input=fp32_path,
            model_output=quant_path,
            weight_type=QuantType.QInt8,
        )
        print(f"已导出动态量化 INT8: {quant_path}")

    print("\n在 config.yaml 中设置:")
    print('  backend: onnx')
    print(f'  onnx_model_path: "{quant_path if args.quantize else fp32_path}"')


if __name__ == "__main__":
    main()
