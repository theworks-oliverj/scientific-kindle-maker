"""
Model Manager — downloads and caches all ML model weights on first run.

Models used:
  1. DocLayout-YOLO  (layout detection)
     HuggingFace: juliozhao/DocLayout-YOLO-DocStructBench
     File: doclayout_yolo_docstructbench_imgsz1024.pt
     Loaded directly via YOLOv10(weights_path) — not from_pretrained(),
     which fails because it tries to fetch yolov10n.pt as a base model.

  2. YOLOv8-MFD  (math formula detection — inline + display)
     HuggingFace: opendatalab/PDF-Extract-Kit-1.0
     File: models/MFD/yolov8_mix.pt
     Classes: 0 = embedding (inline), 1 = isolated (display)

  3. pix2tex / LaTeX-OCR  (math expression recognition → LaTeX)
     Weights auto-managed by the pix2tex package (~/.cache/pix2tex/).

  4. PaddleOCR  (text OCR for scanned pages)
     Version 3.x API — use_textline_orientation replaces use_angle_cls,
     show_log parameter was removed entirely.

All weights are stored in ~/.kindle_converter/models/ by default.
Override with the KINDLE_CONVERTER_MODELS_DIR environment variable.

HuggingFace downloads happen once at setup time only.
All inference runs 100% locally — no data leaves your machine.
"""

import os
import shutil
from pathlib import Path
from typing import Any, Optional

from .observability.logger import get_logger

log = get_logger("model_manager")

DEFAULT_MODELS_DIR = Path.home() / ".kindle_converter" / "models"

# DocLayout-YOLO
DOCLAYOUT_REPO_ID = "juliozhao/DocLayout-YOLO-DocStructBench"
DOCLAYOUT_FILENAME = "doclayout_yolo_docstructbench_imgsz1024.pt"

# YOLOv8-MFD (PDF-Extract-Kit)
MFD_REPO_ID   = "opendatalab/PDF-Extract-Kit"
MFD_FILE_PATH = "models/MFD/weights.pt"

# PDF-Extract-Kit class-ID → formula type
MFD_CLASS_MAP = {
    0: "inline_formula",   # "embedding"
    1: "display_formula",  # "isolated"
}


def get_models_dir() -> Path:
    env = os.environ.get("KINDLE_CONVERTER_MODELS_DIR")
    return Path(env) if env else DEFAULT_MODELS_DIR


def _hf_download(repo_id: str, filename: str, local_dir: Path, repo_type: str = "model") -> Path:
    """
    Downloads a single file from HuggingFace Hub to local_dir.
    repo_type: "model" (default) or "dataset" — must match how the repo is registered on HF.
    """
    from huggingface_hub import hf_hub_download  # type: ignore

    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        local_dir=str(local_dir),
    )
    return Path(local_path)


# ---------------------------------------------------------------------------
# Individual model loaders
# ---------------------------------------------------------------------------

def load_layout_model(models_dir: Optional[Path] = None) -> Any:
    """
    Downloads (if needed) and loads DocLayout-YOLO.

    Uses direct weight download + YOLOv10(path) instead of from_pretrained().
    from_pretrained() fails because after downloading the DocLayout weights it
    then looks for yolov10n.pt (the ultralytics base model) which is separate.
    Loading the weights file directly bypasses that lookup entirely.
    """
    try:
        from doclayout_yolo import YOLOv10  # type: ignore
    except ImportError as exc:
        log.error("layout_import_failed", error=str(exc), hint="pip install doclayout-yolo")
        return None

    mdir = models_dir or get_models_dir()
    mdir.mkdir(parents=True, exist_ok=True)
    local_weights = mdir / "doclayout_yolo.pt"

    if not local_weights.exists():
        log.info("downloading_layout_model", repo=DOCLAYOUT_REPO_ID, file=DOCLAYOUT_FILENAME)
        try:
            downloaded = _hf_download(DOCLAYOUT_REPO_ID, DOCLAYOUT_FILENAME, mdir / "_hf_cache")
            shutil.copy2(str(downloaded), str(local_weights))
            log.info("layout_model_downloaded", path=str(local_weights))
        except Exception as exc:
            log.error("layout_model_download_failed", error=str(exc))
            return None

    try:
        model = YOLOv10(str(local_weights))
        log.info("layout_model_loaded", path=str(local_weights))
        return model
    except Exception as exc:
        log.error("layout_model_load_failed", path=str(local_weights), error=str(exc))
        return None


def load_mfd_model(models_dir: Optional[Path] = None) -> Any:
    """
    Downloads (if needed) and loads the YOLOv8-MFD formula detection model.
    Source: opendatalab/PDF-Extract-Kit-1.0 → models/MFD/yolov8_mix.pt
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        log.error("mfd_import_failed", error=str(exc), hint="pip install ultralytics")
        return None

    mdir = models_dir or get_models_dir()
    mdir.mkdir(parents=True, exist_ok=True)
    local_weights = mdir / "mfd_yolov8_mix.pt"

    if not local_weights.exists():
        log.info("downloading_mfd_model", repo=MFD_REPO_ID, file=MFD_FILE_PATH)
        try:
            # hf_hub_download returns the exact path of the downloaded file,
            # which may be inside a subdirectory matching the filename path.
            downloaded = _hf_download(MFD_REPO_ID, MFD_FILE_PATH, mdir / "_hf_cache")
            shutil.copy2(str(downloaded), str(local_weights))
            log.info("mfd_model_downloaded", path=str(local_weights))
        except Exception as exc:
            err = str(exc)
            hint = (
                "Visit https://huggingface.co/datasets/opendatalab/PDF-Extract-Kit-1.0, "
                "accept the license, then run `huggingface-cli login`."
                if "401" in err or "Unauthorized" in err or "authentication" in err.lower()
                else "Check your internet connection or run `huggingface-cli login`."
            )
            log.error("mfd_model_download_failed", error=err, hint=hint)
            return None

    try:
        model = YOLO(str(local_weights))
        log.info("mfd_model_loaded", path=str(local_weights))
        return model
    except Exception as exc:
        log.error("mfd_model_load_failed", path=str(local_weights), error=str(exc))
        return None


def load_pix2tex_model() -> Any:
    """
    Loads pix2tex (LaTeX-OCR) for math expression recognition.
    Weights (~1.4 GB) are downloaded from HuggingFace on first instantiation
    and cached by the pix2tex package itself (~/.cache/pix2tex/).
    All subsequent calls load from local cache — no network needed.
    """
    try:
        from pix2tex.cli import LatexOCR  # type: ignore

        log.info("loading_pix2tex")
        model = LatexOCR()
        log.info("pix2tex_loaded")
        return model
    except ImportError as exc:
        log.error("pix2tex_import_failed", error=str(exc), hint="pip install pix2tex>=0.1.2")
        return None
    except Exception as exc:
        log.error("pix2tex_load_failed", error=str(exc))
        return None


def load_paddleocr_model() -> Any:
    """
    Instantiates PaddleOCR 3.x.

    API changes from 2.x to 3.x:
      - use_angle_cls  → use_textline_orientation  (deprecated, renamed)
      - show_log       → removed entirely
      - lang           → still supported
    """
    try:
        from paddleocr import PaddleOCR  # type: ignore

        log.info("loading_paddleocr")
        model = PaddleOCR(
            use_textline_orientation=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            lang="en",
        )
        log.info("paddleocr_loaded")
        return model
    except Exception as exc:
        log.error("paddleocr_load_failed", error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Aggregate loader
# ---------------------------------------------------------------------------

def load_all_models(models_dir: Optional[Path] = None) -> dict[str, Any]:
    """
    Loads all models. Returns a dict with keys:
      "layout"    — DocLayout-YOLO (or None)
      "formula"   — YOLOv8-MFD    (or None)
      "pix2tex"   — pix2tex model  (or None)
      "paddleocr" — PaddleOCR     (or None)

    None values mean the model failed to load. The pipeline degrades
    gracefully: missing formula model → no equations detected;
    missing pix2tex → all equations go to fallback.
    """
    mdir = models_dir or get_models_dir()
    log.info("loading_all_models", models_dir=str(mdir))

    models = {
        "layout":    load_layout_model(mdir),
        "formula":   load_mfd_model(mdir),
        "pix2tex":   load_pix2tex_model(),
        "paddleocr": load_paddleocr_model(),
    }

    loaded  = [k for k, v in models.items() if v is not None]
    missing = [k for k, v in models.items() if v is None]
    log.info("models_ready", loaded=loaded, missing=missing)

    if missing:
        log.warning(
            "some_models_unavailable",
            missing=missing,
            hint="Install missing packages from requirements.txt and re-run.",
        )

    return models
