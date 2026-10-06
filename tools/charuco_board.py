"""Generate the printable ChArUco board used by tools/calib_intrinsics.py.

    python tools/charuco_board.py                       # A3, 7x5 squares of 55 mm
    python tools/charuco_board.py --paper A2 --cols 9 --rows 7

Writes a PDF at exact physical size (plus a PNG preview). Print it at 100 % /
"actual size" with fit-to-page OFF, on matte paper, and glue it flat to foam
board. Then check the printed 100 mm ruler and measure one square: pass the
measured size to `calib_intrinsics.py solve --square-mm` for the record (the lens
parameters themselves don't depend on it, only the marker:square ratio does).

The board is built by tracker3d.lens.make_board, the same function the detector
uses, so the printout and the detector can never disagree.
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracker3d.lens import BOARD_DEFAULTS, make_board   # noqa: E402

PAPER_MM = {"A4": (297.0, 210.0), "A3": (420.0, 297.0), "A2": (594.0, 420.0)}  # landscape
PX_PER_SQUARE = 650          # ~300 dpi at 55 mm; the PDF dpi is set from this exactly


def render_page(cols, rows, square_mm, marker_mm, dictionary, paper):
    """Return (page image, dpi). The board is centred; a 100 mm ruler and the
    board parameters go in the margins."""
    page_w_mm, page_h_mm = PAPER_MM[paper]
    board_w_mm, board_h_mm = cols * square_mm, rows * square_mm
    if board_w_mm + 10 > page_w_mm or board_h_mm + 16 > page_h_mm:
        raise SystemExit(f"A {cols}x{rows} board of {square_mm} mm squares "
                         f"({board_w_mm:.0f}x{board_h_mm:.0f} mm) does not fit {paper} "
                         "with margins for the ruler. Use fewer/smaller squares or bigger paper.")
    # Integer pixels per square, and the dpi that makes that exactly square_mm.
    px_sq = PX_PER_SQUARE
    dpi = px_sq / square_mm * 25.4
    mm = dpi / 25.4                                   # pixels per mm
    page = np.full((int(round(page_h_mm * mm)), int(round(page_w_mm * mm))), 255, np.uint8)

    board = make_board(cols, rows, square_mm, marker_mm, dictionary)
    img = board.generateImage((cols * px_sq, rows * px_sq), marginSize=0, borderBits=1)
    y0 = (page.shape[0] - img.shape[0]) // 2
    x0 = (page.shape[1] - img.shape[1]) // 2
    page[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img

    # 100 mm ruler with 10 mm ticks, below the board.
    ry = y0 + img.shape[0] + int(3 * mm)
    rx = x0
    th = max(2, int(0.3 * mm))
    cv2.line(page, (rx, ry), (rx + int(round(100 * mm)), ry), 0, th)
    for i in range(11):
        x = rx + int(round(i * 10 * mm))
        cv2.line(page, (x, ry), (x, ry + int((3 if i % 5 == 0 else 1.5) * mm)), 0, th)
    font_scale = 0.09 * mm
    cv2.putText(page, "100 mm - check with a ruler", (rx + int(round(104 * mm)), ry + int(2.5 * mm)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, 0, max(1, th // 2), cv2.LINE_AA)
    label = (f"ChArUco {cols}x{rows} squares | square {square_mm:g} mm | marker {marker_mm:g} mm | "
             f"{dictionary} | print at 100% (no fit-to-page)")
    cv2.putText(page, label, (x0, y0 - int(2 * mm)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 0,
                max(1, th // 2), cv2.LINE_AA)
    return page, dpi


def main():
    d = BOARD_DEFAULTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=d["cols"])
    ap.add_argument("--rows", type=int, default=d["rows"])
    ap.add_argument("--square-mm", type=float, default=d["square_mm"])
    ap.add_argument("--marker-mm", type=float, default=d["marker_mm"])
    ap.add_argument("--dict", default=d["dictionary"])
    ap.add_argument("--paper", default="A3", choices=sorted(PAPER_MM))
    ap.add_argument("--out", default=None, help="output PDF path")
    args = ap.parse_args()

    page, dpi = render_page(args.cols, args.rows, args.square_mm, args.marker_mm, args.dict,
                            args.paper)
    out = args.out or (f"charuco_{args.cols}x{args.rows}_{args.square_mm:g}mm_"
                       f"{args.paper}.pdf")
    from PIL import Image
    Image.fromarray(page).save(out, "PDF", resolution=dpi)
    png = os.path.splitext(out)[0] + ".png"
    cv2.imwrite(png, page)
    print(f"Wrote {out} ({args.paper} landscape, {dpi:.1f} dpi) and preview {png}")
    print("Print at 100% / actual size (fit-to-page OFF), then measure one square.")


if __name__ == "__main__":
    main()
