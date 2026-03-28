#!/usr/bin/env python3
"""
TRG 手术辅助决策演示界面（Gradio）。
运行:  TRG_CONFIG=/path/to/config.yaml python app_gradio.py
默认读取项目内 config.yaml；若不存在则尝试 config.example.yaml。
"""
from __future__ import annotations

import os
import sys

# 保证可导入 src
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import gradio as gr
from PIL import Image

from src.config_merge import load_app_config
from src.pipeline import run_inference
from src.runtime import load_engine

_ENGINE = None
_CFG = None


def _config_path() -> str:
    env = os.environ.get("TRG_CONFIG")
    if env and os.path.isfile(env):
        return env
    for name in ("config.yaml", "config.example.yaml"):
        p = os.path.join(ROOT, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "未找到配置文件。请复制 config.example.yaml 为 config.yaml 并填写路径，"
        "或设置环境变量 TRG_CONFIG=/绝对路径/config.yaml"
    )


def get_runtime():
    global _ENGINE, _CFG
    if _CFG is None:
        _CFG = load_app_config(_config_path())
    if _ENGINE is None:
        _ENGINE = load_engine(_CFG)
    return _ENGINE, _CFG


def infer(original: Image.Image, seg: Image.Image, input_mode: str):
    if original is None or seg is None:
        return None, "请同时上传原图与分割掩膜 PNG。", "{}"
    engine, cfg = get_runtime()
    mode = input_mode or cfg.get("input_mode", "masked")
    vis, md, jstr, _ = run_inference(original, seg, engine, cfg, input_mode=mode)
    return vis, md, jstr


def main():
    _, cfg = get_runtime()
    class_names = cfg.get("class_names", ["类别0", "类别1"])
    desc = (
        "### 使用说明\n"
        "1. 上传 **原图 PNG** 与 **分割掩膜 PNG**（单通道或 RGB 均可，内部转灰度）。\n"
        "2. 选择输入模式：**masked** = 灰度原图×掩膜；**full** = 仅用整幅灰度图（与 `PDCC/main.py` 中 `model(batch['image'])` 更接近）。\n"
        "3. 若配置了 `training_config_json`，会自动按训练时的 **img_size** 缩放（见 `docs/TRAINING_ALIGNMENT.md`）。\n"
        "4. 右侧为 **目标区域高亮**；下方仅展示 **预测类别**（不显示概率数值，非诊断结论）。\n\n"
        f"当前配置类别：{class_names}\n"
        f"**推理后端**：`{cfg.get('backend', 'torch')}`（`onnx` 时无需 GPU / 无需 PyTorch 运行界面，仅需 onnxruntime）\n"
    )
    with gr.Blocks(title="TRG 手术辅助演示") as demo:
        gr.Markdown("# 超声/影像区域分割 + 深度学习辅助分析（演示系统）")
        gr.Markdown(desc)
        with gr.Row():
            img_orig = gr.Image(type="pil", label="原图 PNG")
            img_mask = gr.Image(type="pil", label="分割掩膜 PNG")
        mode_in = gr.Radio(
            choices=["masked", "full"],
            value=str(cfg.get("input_mode", "masked")),
            label="模型输入模式",
        )
        btn = gr.Button("分析", variant="primary")
        with gr.Row():
            out_vis = gr.Image(type="pil", label="目标区域高亮")
        out_md = gr.Markdown()
        out_json = gr.Code(label="结构化结果 JSON", language="json")
        btn.click(infer, inputs=[img_orig, img_mask, mode_in], outputs=[out_vis, out_md, out_json])
    host = os.environ.get("TRG_HOST", "0.0.0.0")
    port = int(os.environ.get("TRG_PORT", "7860"))
    demo.launch(server_name=host, server_port=port)


if __name__ == "__main__":
    main()
