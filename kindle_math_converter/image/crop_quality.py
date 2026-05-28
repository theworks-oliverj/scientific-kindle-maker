"""
Crop quality analysis for equation images.

Three fast checks (numpy only, no model) that run before and after pix2tex
to detect truncated crops and implausible recognition output:

  check_boundary_ink  — detects ink touching crop edges (truncated crop)
  delimiter_imbalance — counts unmatched (, [, \\left vs ), ], \\right in LaTeX
  recognition_sparsity — pixel area per LaTeX char (high = likely bad crop)
"""
import io

import numpy as np


def check_boundary_ink(img_bytes: bytes, margin_px: int = 3) -> dict[str, bool]:
    """
    Returns {"top", "bottom", "left", "right"}: True if dark ink exists
    within margin_px pixels of that edge.

    A True result means the crop is likely truncated on that side — the formula
    extends beyond the bbox and the full character was not captured.
    Falls back to all-False on any error so callers can always use the result.
    """
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        arr = np.array(img, dtype=np.uint8)
        h, w = arr.shape
        if h <= 2 * margin_px or w <= 2 * margin_px:
            return {"top": False, "bottom": False, "left": False, "right": False}
        # Ink = pixels significantly darker than white (< 200 on 0–255 grayscale)
        # Using 200 rather than 128 to be sensitive to light but present ink.
        threshold = 200
        return {
            "top":    bool(np.any(arr[:margin_px, :] < threshold)),
            "bottom": bool(np.any(arr[-margin_px:, :] < threshold)),
            "left":   bool(np.any(arr[:, :margin_px] < threshold)),
            "right":  bool(np.any(arr[:, -margin_px:] < threshold)),
        }
    except Exception:
        return {"top": False, "bottom": False, "left": False, "right": False}


def delimiter_imbalance(latex: str) -> int:
    """
    Returns (open_count - close_count) for visible delimiters: (, [, and \\left/\\right.

    A non-zero result strongly suggests the crop cut off a matching bracket.
    Positive = more openers → closing bracket was cut off (bottom/right of crop).
    Negative = more closers → opening bracket was cut off (top/left of crop).

    LaTeX grouping braces { } are intentionally excluded — they are structural
    LaTeX syntax and frequently unbalanced in valid sub-expressions.
    """
    parens  = latex.count("(")       - latex.count(")")
    squares = latex.count("[")       - latex.count("]")
    lr      = latex.count(r"\left")  - latex.count(r"\right")
    return parens + squares + lr


def recognition_sparsity(crop_bytes: bytes, latex: str) -> float:
    """
    Returns crop pixel-area divided by LaTeX character count.

    High values (> 5 000 px²/char) indicate a large crop produced very short
    LaTeX — typical of a truncated or partially-blank crop where pix2tex only
    saw fragments of the equation.
    Returns 0.0 on any error.
    """
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(crop_bytes))
        w, h = img.size
        return (w * h) / max(len(latex.strip()), 1)
    except Exception:
        return 0.0
