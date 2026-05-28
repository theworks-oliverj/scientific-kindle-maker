"""
Visual diff utilities for the HTML report.
Generates side-by-side comparison images for flagged equations.
"""
import io

from PIL import Image, ImageDraw


def make_side_by_side_diff(
    source_crop_bytes: bytes,
    rendered_svg: str | None,
    width: int = 600,
) -> bytes:
    """
    Returns a PNG of source crop | rendered SVG side by side.
    If rendered_svg is None, shows source crop | placeholder.
    Used in the HTML report for flagged equations.
    """
    half = width // 2

    source_img = Image.open(io.BytesIO(source_crop_bytes)).convert("RGB")
    source_img.thumbnail((half, half * 2), Image.LANCZOS)

    if rendered_svg is not None:
        rendered_img = _svg_to_pil(rendered_svg, half)
    else:
        rendered_img = _placeholder_image(half, source_img.height)

    total_height = max(source_img.height, rendered_img.height)
    canvas = Image.new("RGB", (width, total_height), color=(255, 255, 255))
    canvas.paste(source_img, (0, 0))
    canvas.paste(rendered_img, (half, 0))

    # Divider line
    draw = ImageDraw.Draw(canvas)
    draw.line([(half, 0), (half, total_height)], fill=(200, 200, 200), width=1)

    out = io.BytesIO()
    canvas.save(out, format="PNG")
    return out.getvalue()


def _placeholder_image(width: int, height: int) -> Image.Image:
    img = Image.new("RGB", (width, max(height, 40)), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "No render available", fill=(150, 150, 150))
    return img


def _svg_to_pil(svg: str, max_width: int) -> Image.Image:
    """
    Attempts to rasterize an SVG to PIL Image for the diff view.
    Falls back to a placeholder if cairosvg is not available.
    """
    try:
        import cairosvg  # type: ignore
        png_bytes = cairosvg.svg2png(bytestring=svg.encode(), output_width=max_width)
        return Image.open(io.BytesIO(png_bytes)).convert("RGB")
    except ImportError:
        return _placeholder_image(max_width, 80)
