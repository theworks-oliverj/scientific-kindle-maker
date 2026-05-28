# Kindle Math Converter

Converts academic PDFs and EPUBs into Kindle-compatible EPUB3 files with
correctly rendered mathematical equations as scalable vector graphics (SVG).

---

## Table of Contents

1. [How it works — the short version](#1-how-it-works--the-short-version)
2. [System requirements](#2-system-requirements)
3. [Installation](#3-installation)
4. [Usage](#4-usage)
5. [Understanding the output files](#5-understanding-the-output-files)
6. [Configuration reference](#6-configuration-reference)
7. [The pipeline — what happens inside](#7-the-pipeline--what-happens-inside)
8. [Troubleshooting](#8-troubleshooting)
9. [Known limitations](#9-known-limitations)

---

## 1. How it works — the short version

Most PDF-to-Kindle converters break on math. They either skip equations entirely,
render them as blurry low-resolution images, or produce raw LaTeX text the Kindle
can't display.

This tool does four things differently:

1. **Detects** every equation on every page using a trained AI model (YOLOv8-MFD).
2. **Recognises** each equation as LaTeX using another AI model (pix2tex).
3. **Validates** the result by compiling the LaTeX and comparing the rendered output
   to the original — equations that look wrong get a repair attempt.
4. **Renders** each equation to SVG using a real LaTeX engine (tectonic + dvisvgm),
   and embeds it inline in the EPUB — so equations scale with font size, respect
   dark mode, and never appear blurry.

---

## 2. System requirements

- **macOS** (Apple Silicon or Intel). Linux works with minor path differences.
- **Python 3.10 or later.**
- **~5 GB free disk space** for model weights and tools.
- **Internet connection** for the first run (to download models).
- No GPU required — everything runs on CPU.

---

## 3. Installation

Run these commands in order. Each step must complete without errors before
moving to the next.

### Step 1 — Install Homebrew (macOS package manager)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

After it finishes, it prints a "Next steps" block. Run the two `eval` lines it
shows — they add `brew` to your PATH. Then verify:

```bash
brew --version
```

### Step 2 — Install system binaries

```bash
brew install tectonic dvisvgm poppler
```

Verify each one installed:

```bash
tectonic --version
dvisvgm --version
pdftoppm -v        # part of poppler
```

### Step 3 — Navigate to the project root

```bash
cd "Scientific Kindle Maker"
```

All subsequent commands are run from this directory.

### Step 4 — Install CPU PyTorch

This must be installed before the other packages. If you install it after,
pip may have already pulled a 3 GB GPU build that won't uninstall cleanly.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Step 5 — Install Python dependencies

```bash
pip install -r kindle_math_converter/requirements.txt
```

This takes a few minutes. If you see any red error lines, see
[Troubleshooting → Installation errors](#installation-errors).

### Step 6 — Download AI model weights

```bash
python main.py download-models
```

This downloads ~2–4 GB of model weights on the first run and caches them in
`~/.kindle_converter/models/`. Subsequent runs skip the download entirely.

Expected output:
```
Downloading model weights
  Cache dir: /Users/you/.kindle_converter/models

  Loaded:  layout, formula, pix2tex, paddleocr
```

If any model shows as "Missing", see
[Troubleshooting → Model download failures](#model-download-failures).

---

## 4. Usage

All commands are run from the `Scientific Kindle Maker/` directory.

### Convert a PDF

```bash
python main.py convert path/to/paper.pdf
```

### Convert an EPUB

```bash
python main.py convert path/to/book.epub
```

### Common options

```bash
# Save output to a specific folder
python main.py convert paper.pdf --output-dir ./my_books

# See detailed per-equation logging while it runs
python main.py convert paper.pdf --verbose

# Open the HTML inspection report automatically when done
python main.py convert paper.pdf --open-report

# Lower the quality threshold (accept more equations as-is, fewer flagged)
python main.py convert paper.pdf --cdm-threshold 0.80

# Use a different body font size for SVG scaling (default is 10pt)
python main.py convert paper.pdf --font-size 11.0

# Skip epubcheck (if Java is not installed)
python main.py convert paper.pdf --no-epubcheck
```

### Full options reference

```
python main.py convert --help

Options:
  --output-dir PATH       Output directory.              [default: ./output]
  --verbose               Detailed per-equation logging in terminal.
  --cdm-threshold FLOAT   Quality pass threshold (0–1).  [default: 0.88]
  --font-size FLOAT       Body font size in pt.          [default: 10.0]
  --mathpix-id TEXT       Mathpix App ID (optional fallback recogniser).
  --mathpix-key TEXT      Mathpix App Key.
  --open-report           Open HTML report in browser on completion.
  --dump-cache            Write cache contents to JSON for debugging.
  --no-epubcheck          Skip EPUB validation (if Java unavailable).
```

### Re-download models

```bash
python main.py download-models

# Save to a custom location
python main.py download-models --models-dir /path/to/models
```

### Terminal summary after a successful run

```
Complete
  Source:     griffiths_em.pdf
  Output:     ./output/griffiths_em.epub
  Equations:  312 total
              287 passed  (92.0%)
               19 repaired (6.1%)
                4 fallback  (1.3%)
                2 flagged   (0.6%)
  Cache:       89 hits / 223 misses (28.5% hit rate)
  Duration:    4m 22s
  Report:     ./output/griffiths_em_report.html
```

If any equations are flagged:
```
  ⚠  2 equations could not be rendered.
     Open the report and review: ./output/griffiths_em_report.html
```

---

## 5. Understanding the output files

Every conversion produces four files in the output directory
(default `./output/`):

### `{title}.epub`

The converted book. Send this to your Kindle via the
[Send to Kindle](https://www.amazon.com/sendtokindle) website or app.
Amazon converts it to their internal KFX format automatically.

### `{title}_report.html`

Open this in any browser. It contains:

- A summary table of every pipeline stage (duration, status, any errors).
- A searchable, sortable table of every equation in the document showing:
  - The source image crop (what the equation looked like in the original).
  - The recognised LaTeX.
  - The CDM quality score (how closely the rendered SVG matches the original).
  - The confidence gate: **pass**, **repaired**, **fallback**, or **flagged**.
- For flagged equations: the source crop and raw LaTeX so you can correct them manually.
- The full error log.

Use this file to diagnose any quality issues.

### `{title}_result.json`

Machine-readable summary of the pipeline run — useful if you want to script
batch processing or check results programmatically.

### `{title}_pipeline.log`

Newline-delimited JSON log. Every stage start/end and every equation event is
recorded here with timestamps. Useful for deep debugging.

---

## 6. Configuration reference

### Quality threshold (`--cdm-threshold`)

The CDM score is a measure of how closely the rendered SVG matches the source
equation image. It ranges from 0 (completely wrong) to 1 (pixel-perfect).

| Score | What happens |
|---|---|
| ≥ threshold (default 0.88) | Equation passes as-is |
| threshold − 0.18 to threshold | Repair attempted (common transcription errors fixed) |
| Below repair threshold | Sent to fallback (Mathpix if configured, otherwise flagged) |

**Raise the threshold** (e.g. `--cdm-threshold 0.92`) if you need higher
accuracy and don't mind more equations being flagged.

**Lower the threshold** (e.g. `--cdm-threshold 0.78`) if too many equations
are being flagged and the results look visually correct anyway.

### Font size (`--font-size`)

SVG equation dimensions are expressed in `em` units so they scale when the
Kindle user adjusts font size. The conversion uses the body font size in points
as the divisor. Most books use 10pt or 11pt body text. If equations appear
too large or small on the Kindle, adjust this value.

### Mathpix fallback (optional)

Mathpix is a commercial API for equation recognition. If configured, equations
that fail the local pipeline are sent to Mathpix instead of being flagged.

```bash
export MATHPIX_APP_ID=your_app_id
export MATHPIX_APP_KEY=your_app_key
python main.py convert paper.pdf
```

Or pass them directly:

```bash
python main.py convert paper.pdf --mathpix-id your_id --mathpix-key your_key
```

### Model weights location

By default, weights are stored in `~/.kindle_converter/models/`.
Override with an environment variable:

```bash
export KINDLE_CONVERTER_MODELS_DIR=/Volumes/ExternalDrive/models
python main.py download-models
```

---

## 7. The pipeline — what happens inside

Understanding the stages helps interpret the HTML report and diagnose problems.

```
Input file
    │
    ▼
Stage 1 — Classifier
    Reads the file and determines: is this a LaTeX PDF, a scanned PDF,
    or an EPUB? Detects math fonts, scanned pages, column layout.
    ► Fatal if the file is encrypted, empty, or unrecognised format.
    │
    ▼
Stage 2 — Extraction  (three variants)
    2A Digital PDF:  Extracts text blocks and their positions via PyMuPDF.
    2B Scanned PDF:  Rasterises each page to a 300 DPI grayscale image.
    2C EPUB/HTML:    Parses MathML elements and equation images from HTML.
    │
    ▼
Stage 3 — Detection
    Runs YOLOv8-MFD on each page image. Draws a bounding box around every
    equation. Classifies each as inline (within a sentence) or display
    (its own centred line). Crops and stores each equation image.
    ► Fatal if the model cannot run at all.
    │
    ▼
Stage 4 — Reading Order
    Sorts all content regions into the correct reading sequence.
    Handles two-column layouts. Extracts equation numbers like "(3.14)".
    │
    ▼
Stage 5A — Text OCR  (scanned PDFs only)
    Runs PaddleOCR on text block regions to extract body text.
    Skipped entirely for digital PDFs (text already extracted in Stage 2A).
    │
    ▼
Stage 5B — Formula Recognition
    Runs pix2tex on each equation crop image.
    pix2tex outputs a LaTeX string for each equation.
    For EPUB MathML: converts MathML to LaTeX directly (no AI needed).
    ► Degraded per equation if pix2tex fails on one crop.
    │
    ▼
Stage 6 — Validation and Repair
    For each equation:
      1. Compiles the LaTeX with tectonic.
      2. Compares the compiled output to the source crop (SSIM score).
      3. If the score is below the threshold, applies heuristic repair rules
         (fixes common pix2tex transcription errors like \Sum → \sum).
      4. Assigns a confidence gate: PASS, REPAIR, FALLBACK, or FLAGGED.
    │
    ▼
Stage 7 — Routing
    Splits equations into two groups:
    PASS/REPAIR → Stage 8A (local SVG rendering)
    FALLBACK    → Stage 8B (Mathpix API or flag for review)
    │
    ▼
Stage 8A — SVG Rendering
    Compiles each passing equation with tectonic (→ DVI format).
    Converts DVI to SVG with dvisvgm using --no-fonts (critical: prevents
    a known Kindle rendering bug where font sizes get corrupted).
    Checks the session cache first — identical equations skip rendering.
    ► Failed equations routed to Stage 8B.
    │
    ▼
Stage 8B — Fallback
    If Mathpix is configured: sends the equation image to Mathpix API.
    Otherwise: marks the equation as flagged. A visible placeholder
    [EQ:eq_3_12] appears in the EPUB so the reader knows something is missing.
    │
    ▼
Stage 9 — SVG Post-Processing
    Applies four Kindle-specific fixes to every SVG:
    1. Removes font-size declarations (prevents KDP px→rem corruption).
    2. Converts dimensions from pt to em (scales with Kindle font size).
    3. Replaces fill="black" with fill="currentColor" (dark mode support).
    4. Strips XML declaration (invalid inside HTML5 inline SVG).
    │
    ▼
Stage 10 — EPUB Assembly
    Builds a valid EPUB3 ZIP file. SVGs are embedded inline in XHTML —
    never as <object> tags (KindleGen rejects those). Runs epubcheck
    validation if Java is available.
    ► Fatal if assembly fails (a partial EPUB is not useful).
    │
    ▼
Stage 11 — Output
    Copies the EPUB to the output directory. Writes the HTML report,
    JSON result, and pipeline log.
```

---

## 8. Troubleshooting

### Installation errors

**`ERROR: ResolutionImpossible` during `pip install`**

This means two packages require incompatible versions of a shared dependency.
The most common cause is PyTorch being installed after other packages.
Fix: create a fresh virtual environment and follow the install steps in order.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r kindle_math_converter/requirements.txt
```

---

**`zsh: command not found: brew`**

Homebrew is not installed or not on your PATH. Install it:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
Then run the `eval` lines it prints before trying `brew` again.

---

**`zsh: command not found: tectonic`**

tectonic didn't install or isn't on PATH. Try:
```bash
brew reinstall tectonic
which tectonic   # should print a path like /opt/homebrew/bin/tectonic
```

---

**`paddlepaddle` or `paddleocr` fails to install**

PaddlePaddle has strict Python version requirements. Check your version:
```bash
python3 --version   # needs to be 3.8–3.11
```
If you're on Python 3.12+, pin to Python 3.11 for this project using
`pyenv` or `conda`.

---

### Model download failures

**One or more models show as "Missing" after `download-models`**

Check which package is missing and reinstall it:
```bash
# pix2tex missing
pip install pix2tex>=0.1.2

# doclayout-yolo missing
pip install doclayout-yolo>=0.0.2

# ultralytics (needed for YOLOv8-MFD) missing
pip install ultralytics
```

Then re-run:
```bash
python main.py download-models
```

---

**`HfHubHTTPError` or download hangs**

HuggingFace is rate-limiting or temporarily unavailable. Wait a few minutes
and retry. If the error persists, set a HuggingFace token:
```bash
pip install huggingface_hub
huggingface-cli login
```

---

### Conversion errors

**`Stage 1 failed: ['E002']` — Encrypted PDF**

The PDF is password-protected. Remove the password using Preview (macOS):
File → Export as PDF (this creates an unencrypted copy).

---

**`Stage 3 failed` — Formula detection failed**

The formula detection model (YOLOv8-MFD) could not run. Most likely cause:
`ultralytics` is not installed or the weights file is corrupted.

```bash
pip install ultralytics
python main.py download-models
```

If the weights file is corrupted, delete it and re-download:
```bash
rm ~/.kindle_converter/models/mfd_yolov8_mix.pt
python main.py download-models
```

---

**`tectonic: command not found` during conversion**

tectonic was installed but the current shell session doesn't see it.
Run `brew install tectonic` again, then start a new terminal tab.

---

**Many equations flagged — CDM scores are low**

The pix2tex recognition is producing LaTeX that doesn't match the source.
Common causes and fixes:

| Cause | Fix |
|---|---|
| Source PDF is low resolution | Try rasterising at higher DPI (source quality issue, nothing to fix) |
| Equations use unusual symbol packages | Lower `--cdm-threshold` to 0.75–0.80 |
| Handwritten equations | Not supported — this tool is for typeset documents only |
| Very large or sparse equations | SSIM scoring degrades for these; lower the threshold |

---

**EPUB opens on Kindle but equations are invisible**

This is almost always the KDP dark mode bug. It means the SVG has
hardcoded `fill="black"` that wasn't caught by Stage 9. Check the report:
if the SVG preview in the report renders correctly, the issue is Kindle-side.
Make sure you're sending the file via Send to Kindle (not sideloading via USB),
as Amazon's conversion pipeline applies the KFX fixes.

---

**EPUB opens on Kindle but equations are very large or very small**

The `--font-size` value doesn't match the body text of the converted document.
Try `--font-size 11.0` for 11pt books, or `--font-size 9.0` for compact books.

---

**`epubcheck` reports errors**

If you have Java installed, epubcheck runs automatically. Errors (not warnings)
will fail the assembly stage. The most common cause is malformed SVG from an
unusual LaTeX package. Check the HTML report for which equations caused it.

To skip epubcheck while debugging:
```bash
python main.py convert paper.pdf --no-epubcheck
```

---

**Conversion is very slow**

Normal speeds on CPU:
- 10–30 seconds per page for scanned PDFs (OCR + detection + recognition)
- 2–5 seconds per page for digital PDFs
- 5–15 seconds per equation for tectonic rendering

If it's much slower, check that you're not accidentally running GPU torch
on a machine that expects CPU. Also check `--verbose` output to see which
stage is the bottleneck.

---

### Reading the HTML report

Open `{title}_report.html` in any browser. Key things to look for:

- **Stage table**: any stage showing ✗ is where processing stopped or degraded.
- **Equation table**: sort by CDM score ascending to see the worst-scoring
  equations first. These are most likely to look wrong in the final output.
- **Flagged rows** (highlighted red): these have a placeholder in the EPUB.
  Click "View" to see the source crop and the raw LaTeX — you can correct the
  LaTeX manually and re-run if needed.
- **Error log** at the bottom: each error has an error code (E001–E901)
  and the stage it came from.

---

## 9. Known limitations

**Handwritten equations** — not supported. The recognition model (pix2tex)
is trained on typeset academic documents.

**Very complex multi-line equation arrays** — `\begin{align}` blocks that
span many lines sometimes get split across multiple detected regions or
recognised incorrectly. Check the report for these and lower the threshold
if needed.

**Tables containing equations** — table detection is handled by DocLayout-YOLO
but equation-in-table cases can be missed. If a table in your source document
contains formulas, they may not be captured.

**Right-to-left text** — not tested. Documents with Arabic or Hebrew body text
alongside equations may have incorrect reading order.

**Colour equations** — equations with coloured symbols (e.g. highlighted steps
in a textbook) lose their colour. Stage 9 replaces all black fills with
`currentColor` but does not preserve other colours.

**Password-protected PDFs** — not supported (error E002). Remove the password
before converting.

**Scanned PDFs with skew > 5°** — the deskew step handles minor rotation but
heavily skewed scans will produce poor detection results.
