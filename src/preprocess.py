"""与 PDCC/lib/dataset/ceus.py 对齐的灰度图 + 掩膜预处理（Resize/Pad 策略一致）。"""
from __future__ import annotations

import numpy as np
from PIL import Image


def _pil_to_gray_u8(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("L"), dtype=np.uint8)


def _pil_to_mask_u8(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("L"), dtype=np.uint8)


def _pad_to_min_size(gray: np.ndarray, mask: np.ndarray, min_side: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = gray.shape[:2]
    if h >= min_side and w >= min_side:
        return gray, mask
    pad_h = max(0, min_side - h)
    pad_w = max(0, min_side - w)
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    gray = np.pad(gray, ((top, bottom), (left, right)), mode="constant", constant_values=0)
    mask = np.pad(mask, ((top, bottom), (left, right)), mode="constant", constant_values=0)
    return gray, mask


def _pad_to_min_size_rgb(rgb: np.ndarray, mask: np.ndarray, min_side: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = rgb.shape[:2]
    if h >= min_side and w >= min_side:
        return rgb, mask
    pad_h = max(0, min_side - h)
    pad_w = max(0, min_side - w)
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    rgb = np.pad(rgb, ((top, bottom), (left, right), (0, 0)), mode="constant", constant_values=0)
    mask = np.pad(mask, ((top, bottom), (left, right)), mode="constant", constant_values=0)
    return rgb, mask


def _resize_square(gray: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return gray, mask


def _resize_square_rgb(rgb: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return rgb, mask


def preprocess_pair(
    original: Image.Image,
    seg_mask: Image.Image,
    image_size: int = 256,
    input_mode: str = "masked",
    in_chan: int = 3,
    use_cmel_normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回:
      - model_input: (1, C, H, H) float32，C=1 或 3（须与权重 conv1 输入通道一致）
      - gray_display: (H,H) uint8，便于调试或缩略展示
      - mask_bin: (H,H) uint8 {0,255} 用于叠加高亮
    """
    mask = _pil_to_mask_u8(seg_mask)

    if in_chan == 3:
        rgb = np.array(original.convert("RGB"), dtype=np.uint8)
        if rgb.shape[:2] != mask.shape[:2]:
            import cv2

            mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        rgb, mask = _pad_to_min_size_rgb(rgb, mask, image_size)
        rgb, mask = _resize_square_rgb(rgb, mask, image_size)

        mask_f = (mask.astype(np.float32) / 255.0) if mask.max() > 1 else mask.astype(np.float32)
        mask_f = np.clip(mask_f, 0.0, 1.0)
        x = rgb.astype(np.float32) / 255.0
        if input_mode == "masked":
            x = x * mask_f[..., np.newaxis]
        elif input_mode != "full":
            raise ValueError("input_mode 须为 'masked' 或 'full'")

        if use_cmel_normalize:
            mean = np.array([0.1591, 0.1591, 0.1591], dtype=np.float32)
            std = np.array([0.2593, 0.2593, 0.2593], dtype=np.float32)
            x = (x - mean) / std

        arr = np.transpose(x, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
        gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)
    elif in_chan == 1:
        gray = _pil_to_gray_u8(original)
        if gray.shape != mask.shape:
            import cv2

            mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)

        gray, mask = _pad_to_min_size(gray, mask, image_size)
        gray, mask = _resize_square(gray, mask, image_size)

        mask_f = (mask.astype(np.float32) / 255.0) if mask.max() > 1 else mask.astype(np.float32)
        mask_f = np.clip(mask_f, 0.0, 1.0)
        gray_f = gray.astype(np.float32) / 255.0

        if input_mode == "masked":
            x = gray_f * mask_f
        elif input_mode == "full":
            x = gray_f
        else:
            raise ValueError("input_mode 须为 'masked' 或 'full'")

        if use_cmel_normalize:
            mean, std = 0.1591, 0.2593
            x = (x - mean) / std

        arr = x.astype(np.float32)[np.newaxis, np.newaxis, :, :]
    else:
        raise ValueError("in_chan 仅支持 1 或 3")

    mask_bin = (mask_f >= 0.5).astype(np.uint8) * 255
    return arr, gray, mask_bin
