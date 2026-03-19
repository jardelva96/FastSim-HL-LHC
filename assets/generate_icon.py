"""Generate the FastSim HL-LHC application icon and logo.

Run this script once to create:
  - assets/icon.ico   (Windows executable icon)
  - assets/logo.png   (README / dashboard logo)
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ASSETS_DIR = Path(__file__).resolve().parent
SIZE = 512
ICON_SIZES = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]

# Palette -- CMS-inspired colours.
BG_DARK = (15, 23, 42)         # Dark navy
BLUE_BRIGHT = (59, 130, 246)   # Electric blue
ORANGE_WARM = (249, 115, 22)   # Warm orange
CYAN_GLOW = (34, 211, 238)     # Cyan accent
WHITE = (255, 255, 255)


def draw_calorimeter_grid(draw: ImageDraw.ImageDraw, cx: int, cy: int, radius: int) -> None:
    """Draw concentric rings representing calorimeter layers."""
    n_layers = 6
    for i in range(n_layers):
        r = radius * (i + 1) / n_layers
        alpha = int(255 * (0.3 + 0.7 * (i / (n_layers - 1))))
        colour = (*BLUE_BRIGHT[:2], min(255, BLUE_BRIGHT[2] + i * 15), alpha)

        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw.ellipse(bbox, outline=colour[:3], width=2)

        # Draw cell divisions on each ring.
        n_cells = 16
        for j in range(n_cells):
            angle = 2 * math.pi * j / n_cells
            x_inner = cx + (r - radius / n_layers) * math.cos(angle)
            y_inner = cy + (r - radius / n_layers) * math.sin(angle)
            x_outer = cx + r * math.cos(angle)
            y_outer = cy + r * math.sin(angle)
            draw.line(
                [(x_inner, y_inner), (x_outer, y_outer)],
                fill=(*BLUE_BRIGHT, 80),
                width=1,
            )


def draw_shower_particles(draw: ImageDraw.ImageDraw, cx: int, cy: int, radius: int) -> None:
    """Draw particle shower streaks emanating from centre."""
    n_particles = 12
    for i in range(n_particles):
        angle = 2 * math.pi * i / n_particles + 0.15
        length = radius * (0.5 + 0.5 * ((i * 7 + 3) % 5) / 4)

        x_end = cx + length * math.cos(angle)
        y_end = cy + length * math.sin(angle)

        colour = ORANGE_WARM if i % 3 == 0 else CYAN_GLOW
        width = 3 if i % 3 == 0 else 2
        draw.line([(cx, cy), (x_end, y_end)], fill=colour, width=width)

        # Energy deposit dot at the end.
        dot_r = 4 + (i % 3) * 2
        draw.ellipse(
            [x_end - dot_r, y_end - dot_r, x_end + dot_r, y_end + dot_r],
            fill=colour,
        )


def draw_centre_glow(draw: ImageDraw.ImageDraw, cx: int, cy: int) -> None:
    """Draw a glowing interaction point at the centre."""
    for r in range(20, 0, -2):
        alpha = int(255 * (1.0 - r / 20))
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            fill=(*WHITE, alpha),
        )


def draw_text_label(img: Image.Image, text: str) -> None:
    """Draw the project name at the bottom of the logo."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except OSError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    x = (SIZE - tw) // 2
    y = SIZE - 60
    draw.text((x, y), text, fill=WHITE, font=font)


def generate() -> None:
    """Create icon.ico and logo.png."""
    img = Image.new("RGBA", (SIZE, SIZE), (*BG_DARK, 255))
    draw = ImageDraw.Draw(img, "RGBA")

    cx, cy = SIZE // 2, SIZE // 2 - 20
    radius = 190

    draw_calorimeter_grid(draw, cx, cy, radius)
    draw_shower_particles(draw, cx, cy, radius)
    draw_centre_glow(draw, cx, cy)
    draw_text_label(img, "FastSim HL-LHC")

    # Save logo PNG.
    logo_path = ASSETS_DIR / "logo.png"
    img.save(str(logo_path), "PNG")
    print(f"logo salvo em: {logo_path}")

    # Save ICO with multiple sizes.
    ico_path = ASSETS_DIR / "icon.ico"
    ico_images = []
    for size in ICON_SIZES:
        resized = img.resize(size, Image.Resampling.LANCZOS)
        ico_images.append(resized)
    ico_images[0].save(
        str(ico_path),
        format="ICO",
        sizes=ICON_SIZES,
        append_images=ico_images[1:],
    )
    print(f"icone salvo em: {ico_path}")


if __name__ == "__main__":
    generate()
