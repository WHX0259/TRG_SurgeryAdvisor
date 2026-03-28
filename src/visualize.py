"""原图 + 掩膜区域高亮（用于界面展示，不参与模型推理）。"""
from __future__ import annotations

import numpy as np
from PIL import Image


def highlight_region(
    original: Image.Image,
    mask_bin: np.ndarray,
    highlight_color: tuple[int, int, int] = (255, 80, 80),
    alpha: float = 0.42,
) -> Image.Image:
    """
    original: 任意尺寸 RGB
    mask_bin: 与预处理后掩膜同形状的 {0,255}，此处传入与模型输入同尺寸的 (H,H)
    """
    rgb = np.array(original.convert("RGB"), dtype=np.float32)
    h, w = rgb.shape[:2]
    if mask_bin.shape != (h, w):
        from PIL import Image as PImage

        m = PImage.fromarray(mask_bin, mode="L").resize((w, h), resample=PImage.Resampling.NEAREST)
        mask_bin = np.array(m, dtype=np.uint8)

    m = (mask_bin.astype(np.float32) / 255.0)[..., None]
    color = np.zeros_like(rgb)
    color[..., 0] = highlight_color[0]
    color[..., 1] = highlight_color[1]
    color[..., 2] = highlight_color[2]
    out = rgb * (1.0 - alpha * m) + color * (alpha * m)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")
