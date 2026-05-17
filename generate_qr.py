#!/usr/bin/env python3
"""
Generate a high-contrast PNG QR code for the live Gradio demo URL.

Use this the morning of the defence: launch your Gradio cell in Colab,
copy the .gradio.live URL it prints, paste it as the URL below, run this
script, and drop the resulting PNG into Slide 12 of the deck.

Usage:
    python generate_qr.py                      # uses the URL set in the script
    python generate_qr.py <url>                # override via CLI
    python generate_qr.py <url> <output.png>   # custom output filename

Requires: qrcode[pil]  (install with: pip install "qrcode[pil]")
"""

import sys
from pathlib import Path

try:
    import qrcode
    from qrcode.constants import ERROR_CORRECT_H
except ImportError:
    print("Missing dependency. Install with:  pip install 'qrcode[pil]'")
    sys.exit(1)

# -------- DEFAULT URL --------
# Replace this with the gradio.live URL you get on defence morning.
DEFAULT_URL = "https://edb6fad0e84fe1bd62.gradio.live"

# -------- CONFIG --------
DEFAULT_OUTPUT = "demo_qr.png"

# Project palette: navy QR on white background — matches the slide deck.
FILL_COLOR = "#0A1F44"      # navy (matches slide theme)
BACK_COLOR = "#FFFFFF"      # white for max scan contrast
BOX_SIZE   = 14             # pixels per QR module — big for projection
BORDER     = 4              # quiet zone (modules); 4 is the QR standard


def make_qr(url: str, out_path: str) -> None:
    qr = qrcode.QRCode(
        version=None,                       # auto-fit
        error_correction=ERROR_CORRECT_H,   # 30% — survives projector glare
        box_size=BOX_SIZE,
        border=BORDER,
    )
    qr.add_data(url)
    qr.make(fit=True)

    img = qr.make_image(fill_color=FILL_COLOR, back_color=BACK_COLOR)
    img.save(out_path)

    print(f"✓ QR generated for: {url}")
    print(f"✓ Saved to:         {Path(out_path).resolve()}")
    print(f"✓ Module size:      {qr.modules_count}×{qr.modules_count}")
    print(f"✓ Image size:       {img.size[0]}×{img.size[1]} px")
    print()
    print("Test it: open the PNG, scan with your phone camera. It should "
          "redirect straight to the Gradio demo.")


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    out = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT
    make_qr(url, out)
