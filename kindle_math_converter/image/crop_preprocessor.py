"""
Crop pre-processing for equation images before pix2tex recognition.

Applied only for scanned (VISUAL_PDF) sources. pix2tex was trained on clean
LaTeX-rendered images; feeding it raw scan crops produces degraded output.

Pipeline:
  1. Grayscale conversion
  2. CLAHE contrast normalization (per-region, not page-level)
  3. Gaussian denoise (3×3 kernel)
  4. Adaptive threshold → clean binary image
  5. Invert if background is dark
  6. Tight crop (remove white margins, add 4 px padding)

All steps are guarded — any failure returns the original bytes unchanged.
"""
import io

import numpy as np


def preprocess_crop_for_ocr(img_bytes: bytes) -> bytes:
    """
    Returns preprocessed PNG bytes for a scanned equation crop.
    Falls back to the original bytes on any error.
    """
    try:
        import cv2
        from PIL import Image

        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        arr = np.array(img, dtype=np.uint8)

        h, w = arr.shape
        if w < 10 or h < 5:
            return img_bytes  # too small to process

        # CLAHE — local contrast normalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        arr = clahe.apply(arr)

        # Gaussian denoise
        arr = cv2.GaussianBlur(arr, (3, 3), 0)

        # Adaptive threshold → binary
        binary = cv2.adaptiveThreshold(
            arr, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=8,
        )

        # Invert if background is dark (ink-on-dark scans)
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)

        # Tight crop: find bounding box of ink (non-white) pixels
        rows_with_ink = np.any(binary < 255, axis=1)
        cols_with_ink = np.any(binary < 255, axis=0)

        if rows_with_ink.any() and cols_with_ink.any():
            r_idx = np.where(rows_with_ink)[0]
            c_idx = np.where(cols_with_ink)[0]
            pad = 4
            rmin = max(0, r_idx[0] - pad)
            rmax = min(h - 1, r_idx[-1] + pad)
            cmin = max(0, c_idx[0] - pad)
            cmax = min(w - 1, c_idx[-1] + pad)
            binary = binary[rmin : rmax + 1, cmin : cmax + 1]

        out = Image.fromarray(binary)
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()

    except Exception:
        return img_bytes  # safe fallback: return original unmodified
