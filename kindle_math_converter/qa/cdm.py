"""
CDM (Character Detection Matching) metric implementation.

Full CDM (as in the paper):
  - Render predicted and ground-truth LaTeX to images.
  - Detect characters as objects in each image.
  - Match by visual feature + spatial position.
  - F1 score over matched/unmatched characters.

Simplified CDM (initial implementation — SSIM proxy):
  - Use SSIM (structural similarity index) between source crop and
    rendered PNG as a proxy score.
  - SSIM is a reasonable proxy for simple and medium-complexity equations.
  - It degrades for very large or very sparse equations.
  - Replace with full CDM when accuracy becomes a priority.
  - scikit-image provides SSIM: skimage.metrics.structural_similarity

The simplified version is acceptable for the initial build because:
  1. It is directionally correct — clearly wrong renders score near 0.
  2. It is fast — no secondary model needed.
  3. It is transparent — easy to reason about failures.
  4. Full CDM can be swapped in without changing the pipeline contract.
"""
import io

from PIL import Image


def compute_cdm_score(
    source_crop_bytes: bytes,
    rendered_png_bytes: bytes,
    method: str = "ssim",
) -> float:
    """
    Returns a score in [0, 1]. Higher is better.
    method: "ssim" (simplified) | "cdm" (full, not yet implemented)
    """
    if method == "ssim":
        return _ssim_score(source_crop_bytes, rendered_png_bytes)
    raise NotImplementedError(f"CDM method '{method}' not implemented")


def _ssim_score(a_bytes: bytes, b_bytes: bytes) -> float:
    import numpy as np
    from skimage.metrics import structural_similarity  # type: ignore

    a = np.array(Image.open(io.BytesIO(a_bytes)).convert("L"))
    b = np.array(Image.open(io.BytesIO(b_bytes)).convert("L"))

    # Resize b to match a if needed (rendered may be different resolution)
    if a.shape != b.shape:
        b_img = Image.fromarray(b).resize((a.shape[1], a.shape[0]), Image.LANCZOS)
        b = np.array(b_img)

    score, _ = structural_similarity(a, b, full=True)
    return float(score)
