from __future__ import annotations

import cv2
import numpy as np


def compute_fft_spectrum(image_rgb: np.ndarray) -> np.ndarray:
    """Return a 3-channel log-amplitude FFT spectrum image in uint8 RGB."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    fft = np.fft.fft2(gray)
    shifted = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(shifted))
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    spectrum = magnitude.astype(np.uint8)
    return np.repeat(spectrum[:, :, None], 3, axis=-1)
