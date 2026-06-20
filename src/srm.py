from __future__ import annotations

import cv2
import numpy as np


SRM_KERNELS = np.array(
    [
        [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
        [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    ],
    dtype=np.float32,
)

SRM_KERNELS[0] /= 4.0
SRM_KERNELS[1] /= 12.0
SRM_KERNELS[2] /= 2.0


def compute_srm_residual(image_rgb: np.ndarray) -> np.ndarray:
    """Return a 3-channel SRM-style high-pass residual image in uint8 RGB."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    channels = []
    for kernel in SRM_KERNELS:
        filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
        filtered = np.clip(filtered, -20, 20)
        filtered = ((filtered + 20) / 40 * 255).astype(np.uint8)
        channels.append(filtered)
    return np.stack(channels, axis=-1)
