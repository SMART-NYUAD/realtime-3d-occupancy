"""Generate the README diagrams as light + dark SVGs.

    python docs/diagrams/build.py        # writes docs/diagrams/<name>-{light,dark}.svg

Pure Python, no dependencies. Edit a number or label here and re-run.
"""
import math
import os
from html import escape

OUT = os.path.dirname(os.path.abspath(__file__))

SANS = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

THEMES = {
    "light": dict(
        bg="#FAF9F5", border="#E6E3D8", surface="#FFFFFF", stroke="#D6D2C4",
        text="#1F1E1D", muted="#6F6C64", faint="#A9A59A", grid="#EDEAE1",
        clay="#D97757", clay_fill="#F7E4DB", clay_deep="#B5553A",
        slate="#5A82B4", slate_fill="#E4ECF5",
        olive="#6E8B4E", olive_fill="#E8EEDF",
        amber="#C4953A", amber_fill="#F5EBD6",
    ),
    "dark": dict(
        bg="#1C1B19", border="#34322E", surface="#252422", stroke="#46443E",
        text="#EEECE4", muted="#A3A097", faint="#6F6C65", grid="#2D2C29",
        clay="#E3876A", clay_fill="#3B2921", clay_deep="#EFA08A",
        slate="#82A7D3", slate_fill="#222E3C",
        olive="#98B377", olive_fill="#28311F",
        amber="#D9AE5C", amber_fill="#372D1A",
    ),
}


# ---------------------------------------------------------------- primitives
def T(x, y, s, t, size=13, weight=400, color="text", anchor="start", mono=False,
      spacing=None, italic=False):
    fam = MONO if mono else SANS
    ls = f' letter-spacing="{spacing}"' if spacing else ""
    it = ' font-style="italic"' if italic else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{fam}" font-size="{size}" '
            f'font-weight="{weight}" fill="{t[color]}" text-anchor="{anchor}"{ls}{it}>'
            f'{escape(s)}</text>')


def R(x, y, w, h, t, fill="surface", stroke="stroke", r=10, sw=1, dash=None, opacity=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    o = f' fill-opacity="{opacity}"' if opacity is not None else ""
    f = t[fill] if fill in t else fill
    s = t[stroke] if stroke in t else stroke
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" '
            f'fill="{f}"{o} stroke="{s}" stroke-width="{sw}"{d}/>')


def L(x1, y1, x2, y2, t, color="muted", sw=1.5, arrow=False, dash=None):
    m = f' marker-end="url(#ah-{color})"' if arrow else ""
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{t[color]}" stroke-width="{sw}" stroke-linecap="round"{m}{d}/>')


def P(d, t, color="muted", sw=1.5, arrow=False, fill="none", dash=None, opacity=None):
    m = f' marker-end="url(#ah-{color})"' if arrow else ""
    dd = f' stroke-dasharray="{dash}"' if dash else ""
    f = t[fill] if fill in t else fill
    o = f' fill-opacity="{opacity}"' if opacity is not None else ""
    return (f'<path d="{d}" fill="{f}"{o} stroke="{t[color]}" stroke-width="{sw}" '
            f'stroke-linecap="round" stroke-linejoin="round"{m}{dd}/>')


def C(x, y, r, t, fill="clay", stroke=None, sw=1.5):
    f = t[fill] if fill in t else fill
    s = f' stroke="{t[stroke]}" stroke-width="{sw}"' if stroke else ""
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{f}"{s}/>'


def poly(pts, t, fill, stroke, opacity=0.2, sw=1.5, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    s = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return (f'<polygon points="{s}" fill="{t[fill]}" fill-opacity="{opacity}" '
            f'stroke="{t[stroke]}" stroke-width="{sw}" stroke-linejoin="round"{d}/>')


def svg(w, h, body, t):
    markers = "".join(
        f'<marker id="ah-{c}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M1,1.5 L8.5,5 L1,8.5" '
        f'fill="none" stroke="{t[c]}" stroke-width="1.6" stroke-linecap="round" '
        f'stroke-linejoin="round"/></marker>'
        for c in ("muted", "faint", "clay", "slate", "olive"))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}"><defs>{markers}</defs>'
            f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="14" '
            f'fill="{t["bg"]}" stroke="{t["border"]}"/>' + "".join(body) + "</svg>")


def eyebrow(x, y, s, t, color="muted", anchor="start"):
    return T(x, y, s.upper(), t, size=10.5, weight=600, color=color, anchor=anchor,
             spacing="1.4")


def step_card(x, y, w, h, t, num, title, lines, module=None, accent="clay",
              border="stroke", fill="surface"):
    b = [R(x, y, w, h, t, fill=fill, stroke=border),
         R(x + 16, y + 16, 18, 3, t, fill=accent, stroke=accent, r=1.5, sw=0),
         T(x + 16, y + 40, num, t, size=11, color="muted", mono=True),
         T(x + 16, y + 62, title, t, size=15, weight=600)]
    for i, ln in enumerate(lines):
        b.append(T(x + 16, y + 86 + 17 * i, ln, t, size=12, color="muted"))
    if module:
        b.append(L(x + 16, y + h - 32, x + w - 16, y + h - 32, t, color="grid", sw=1))
        b.append(T(x + 16, y + h - 13, module, t, size=11, color="faint", mono=True))
    return "".join(b)


# ---------------------------------------------------------------- diagrams
def pipeline(t):
    W, H = 1290, 290
    b = ['<g transform="translate(0,-40)">']
    cy = 185
    # cameras
    b.append(eyebrow(36, 86, "Cameras", t, color="faint"))
    for i, name in enumerate(("yi01", "yi04", "yi05")):
        y = 115 + 50 * i
        b.append(R(36, y, 96, 40, t, r=8))
        b.append(C(52, y + 20, 3.5, t, fill="slate"))
        b.append(T(62, y + 24.5, name, t, size=12.5, mono=True))
        b.append(P(f"M132,{y + 20} C146,{y + 20} 142,{cy} 154,{cy}", t, color="faint"))
    b.append(L(154, cy, 164, cy, t, color="faint", arrow=True))

    cards = [
        ("01", "Decode", ["GStreamer + NVDEC", "NTP timestamps"], "gst_stream.py", "slate"),
        ("02", "Align", ["1 s buffer / camera", "same capture instant"], "sync.py", "slate"),
        ("03", "Detect", ["YOLO11-pose", "TensorRT, batched"], "detector.py", "clay"),
        ("04", "Track", ["ByteTrack", "per camera"], "detector.py", "clay"),
        ("05", "Localize", ["feet from pose", "pixels → meters"], "geometry.py", "clay"),
        ("06", "Fuse", ["inverse-variance", "+ blob overlap"], "fusion.py", "olive"),
    ]
    x0, cw, step = 168, 140, 162
    for i, (n, title, lines, mod, acc) in enumerate(cards):
        x = x0 + i * step
        b.append(step_card(x, 110, cw, 150, t, n, title, lines, mod, accent=acc))
        if i < len(cards) - 1:
            b.append(L(x + cw + 4, cy, x + step - 5, cy, t, arrow=True))
    # band labels
    for (a, z, label, col) in ((0, 1, "Ingest", "slate"), (2, 4, "Perception", "clay"),
                               (5, 5, "Fusion", "olive")):
        xa, xz = x0 + a * step, x0 + z * step + cw
        b.append(eyebrow(xa, 86, label, t, color=col))
        b.append(L(xa, 96, xz, 96, t, color=col, sw=2))
    # outputs
    xo = x0 + 5 * step + cw
    b.append(eyebrow(1154, 86, "Output", t, color="faint"))
    for i, name in enumerate(("Floor map", "Camera mosaic", "MQTT")):
        y = 115 + 50 * i
        b.append(R(1154, y, 108, 40, t, r=8))
        b.append(T(1208, y + 24.5, name, t, size=12.5, anchor="middle"))
        b.append(P(f"M{xo + 4},{cy} C{xo + 22},{cy} {xo + 14},{y + 20} {1150},{y + 20}",
                   t, color="faint", arrow=True))
    b.append(T(W / 2, 300, "One pass per aligned frame set   ·   ~15 ms detection   ·   "
               "20–30 FPS with 3 cameras on Jetson AGX Thor", t, size=12, color="muted",
               anchor="middle"))
    b.append("</g>")
    return svg(W, H, b, t)


def calibration(t):
    W, H = 1160, 300
    b = []
    cards = [
        ("STEP 1", "Configure", ["add name + source"], "config.yaml", "slate", "stroke", "surface"),
        ("STEP 2", "Click pairs", ["≥ 5 spread floor points"], "calibrate.py", "slate", "stroke", "surface"),
        ("CHECK", "Leave-one-out", ["error < 0.3 m ?"], None, "amber", "amber", "amber_fill"),
        ("STEP 3", "Fine-tune live", ["drag until dots agree"], "recalibrate.py", "clay", "stroke", "surface"),
        ("STEP 4", "Track", ["positions in meters"], "track.py", "olive", "stroke", "surface"),
    ]
    x0, cw, step, y, h = 42, 180, 224, 56, 140
    for i, (n, title, lines, mod, acc, bd, fl) in enumerate(cards):
        x = x0 + i * step
        b.append(step_card(x, y, cw, h, t, n, title, lines, mod, accent=acc, border=bd, fill=fl))
        if i < len(cards) - 1:
            b.append(L(x + cw + 5, y + h / 2, x + step - 6, y + h / 2, t, arrow=True))
    b.append(T(x0 + 2 * step + cw + 22, y + h / 2 - 8, "yes", t, size=11, color="olive",
               anchor="middle", weight=600))
    # loop back
    xa, xb = x0 + 2 * step + cw / 2, x0 + step + cw / 2
    b.append(P(f"M{xa},{y + h + 4} C{xa},{y + h + 42} {xb},{y + h + 42} {xb},{y + h + 7}",
               t, color="amber", arrow=True, dash="4 4"))
    b.append(T((xa + xb) / 2, y + h + 50, "no — add or move points", t, size=12,
               color="amber", anchor="middle", weight=600))
    b.append(T(W / 2, 278, "Done once per camera placement. The sidecar JSON stores the image "
               "size, so a stream at another resolution is rescaled automatically.",
               t, size=12, color="muted", anchor="middle"))
    return svg(W, H, b, t)


def foot_ladder(t):
    W, H = 1100, 450
    b = []
    # figure
    b.append(R(32, 40, 270, 370, t, r=12))
    b.append(eyebrow(52, 66, "COCO keypoints", t))
    fx = 167
    head, sh, hp, kn, an = (fx, 104), 140, 222, 292, 356
    b.append(R(fx - 58, 78, 116, 290, t, fill="none", stroke="faint", r=4, dash="3 4"))
    b.append(T(fx + 52, 92, "box", t, size=10.5, color="faint", anchor="end", mono=True))
    body = t["muted"]
    segs = [((fx - 30, sh), (fx + 30, sh)), ((fx, sh), (fx, hp)), ((fx - 18, hp), (fx + 18, hp)),
            ((fx - 18, hp), (fx - 20, kn)), ((fx + 18, hp), (fx + 20, kn)),
            ((fx - 20, kn), (fx - 22, an)), ((fx + 20, kn), (fx + 22, an)),
            ((fx - 30, sh), (fx - 40, 212)), ((fx + 30, sh), (fx + 40, 212))]
    for (a, z) in segs:
        b.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{z[0]}" y2="{z[1]}" stroke="{body}" '
                 f'stroke-width="2" stroke-linecap="round" stroke-opacity="0.55"/>')
    b.append(f'<circle cx="{head[0]}" cy="{head[1]}" r="17" fill="none" stroke="{body}" '
             f'stroke-width="2" stroke-opacity="0.55"/>')
    for (x, y, c) in ((fx - 30, sh, "clay"), (fx + 30, sh, "clay"),
                      (fx - 18, hp, "clay"), (fx + 18, hp, "clay"),
                      (fx - 20, kn, "amber"), (fx + 20, kn, "amber"),
                      (fx - 22, an, "olive"), (fx + 22, an, "olive")):
        b.append(C(x, y, 5.5, t, fill=c, stroke="surface", sw=2))
    for (y, s) in ((sh, "shoulders"), (hp, "hips"), (kn, "knees"), (an, "ankles")):
        b.append(T(fx + 70, y + 4, s, t, size=11, color="muted", mono=True, anchor="start"))

    # table
    tx = 340
    b.append(eyebrow(tx, 66, "Source", t))
    b.append(eyebrow(tx + 120, 66, "How the foot is placed", t))
    b.append(T(tx + 440, 66, "UNCERTAINTY σ  ·  SHARE OF BOX HEIGHT", t, size=10.5, weight=600, color="muted", spacing="1.4"))
    rows = [
        ("ankles", "midpoint of both ankles", 4, "olive"),
        ("ankle", "the one visible ankle", 7, "olive"),
        ("knees", "knee + (knee − hip)", 9, "amber"),
        ("hips", "hip + 1.7 × torso length", 20, "clay"),
        ("box", "bottom-center of the box", 50, "clay_deep"),
    ]
    for i, (src, rule, pct, col) in enumerate(rows):
        y = 112 + 58 * i
        b.append(L(tx, y - 30, W - 36, y - 30, t, color="grid", sw=1))
        b.append(C(tx + 6, y - 4, 5, t, fill=col))
        b.append(T(tx + 20, y + 1, src, t, size=14, mono=True, weight=500))
        b.append(T(tx + 120, y + 1, rule, t, size=13))
        bw = max(6, pct / 50 * 200)
        b.append(R(tx + 440, y - 10, 200, 12, t, fill="grid", stroke="grid", r=6, sw=0))
        b.append(R(tx + 440, y - 10, bw, 12, t, fill=col, stroke=col, r=6, sw=0))
        b.append(T(tx + 652, y + 1, f"≈ {pct} %", t, size=12, color="muted", mono=True))
    b.append(L(tx, 112 + 58 * 5 - 30, W - 36, 112 + 58 * 5 - 30, t, color="grid", sw=1))
    b.append(T(tx, 398, "Tried top to bottom — the first rule whose keypoints are visible "
               "(≥ kp_conf) wins.", t, size=12, color="muted"))
    b.append(T(tx, 417, "σ travels with the point, so fusion trusts a visible ankle far more "
               "than an extrapolated one.", t, size=12, color="muted"))
    return svg(W, H, b, t)


def sync(t):
    W, H = 1160, 370
    b = []
    lat = {"yi01": 120, "yi04": 430, "yi05": 430}
    phase = {"yi01": 0, "yi04": 11, "yi05": 22}
    span, period = 600, 33

    def panel(px, title, sub, aligned):
        out = [R(px, 32, 520, 300, t, r=12),
               T(px + 24, 64, title, t, size=15, weight=600),
               T(px + 24, 84, sub, t, size=12, color="muted")]
        ax0, ax1 = px + 92, px + 496
        sx = (ax1 - ax0) / span

        def X(ms_before_now):
            return ax1 - ms_before_now * sx

        t_star = 450
        picks = []
        for i, cam in enumerate(lat):
            y = 128 + 50 * i
            out.append(T(px + 24, y + 4, cam, t, size=12.5, mono=True))
            out.append(L(ax0, y, ax1, y, t, color="grid", sw=1))
            # in-flight latency
            out.append(R(X(lat[cam]), y - 9, lat[cam] * sx, 18, t, fill="slate_fill",
                         stroke="slate_fill", r=4, sw=0))
            out.append(T(X(lat[cam] / 2), y + 4, f"in flight {lat[cam]} ms" if lat[cam] > 200 else f"{lat[cam]} ms", t, size=10.5,
                         color="slate", anchor="middle"))
            avail = [m for m in range(span - phase[cam], 0, -period) if m >= lat[cam]]
            for m in avail:
                out.append(L(X(m), y - 6, X(m), y + 6, t, color="faint", sw=1.5))
            pick = min(avail) if not aligned else min(avail, key=lambda m: abs(m - t_star))
            picks.append(pick)
            out.append(C(X(pick), y, 6, t, fill="clay", stroke="surface", sw=2))
        # axis
        out.append(L(ax0, 262, ax1, 262, t, color="stroke", sw=1))
        out.append(T(ax0, 280, "−600 ms", t, size=10.5, color="faint", mono=True))
        out.append(T(ax1, 280, "now", t, size=10.5, color="faint", mono=True, anchor="end"))
        if aligned:
            out.append(L(X(t_star), 114, X(t_star), 250, t, color="clay", sw=1.2, dash="4 4"))
            out.append(T(X(t_star), 108, "common capture instant", t, size=10.5,
                         color="clay", anchor="middle", weight=600))
        lo, hi = X(max(picks)), X(min(picks))
        spread = max(picks) - min(picks)
        yb = 300
        if hi - lo < 8:
            lo, hi = lo - 4, hi + 4
        out.append(P(f"M{lo},{yb - 6} L{lo},{yb} L{hi},{yb} L{hi},{yb - 6}", t,
                     color="clay", sw=1.5))
        label = f"frames {spread} ms apart" if not aligned else f"frames ≈ {spread} ms apart"
        out.append(T((lo + hi) / 2, yb + 18, label, t, size=12, weight=600, color="clay",
                     anchor="middle"))
        return "".join(out)

    b.append(panel(32, "Latest frame from each camera",
                   "A moving person is seen at different moments → two dots", False))
    b.append(panel(608, "NTP-aligned  (sync.enabled: true)",
                   "Every camera contributes the frame captured at the same instant", True))
    b.append(T(W / 2, 355, "Frames carry RTCP capture timestamps on the Jetson's NTP clock, so "
               "they are matched by when they were captured, not when they arrived.",
               t, size=12, color="muted", anchor="middle"))
    return svg(W, H, b, t)


def _clip(subject, clipper):
    """Sutherland–Hodgman convex clip (both polygons counter-clockwise or both cw)."""
    def inside(p, a, z):
        return (z[0] - a[0]) * (p[1] - a[1]) - (z[1] - a[1]) * (p[0] - a[0]) >= 0

    def inter(p1, p2, a, z):
        x1, y1, x2, y2 = *p1, *p2
        x3, y3, x4, y4 = *a, *z
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        u = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return (x1 + u * (x2 - x1), y1 + u * (y2 - y1))

    out = subject
    for i in range(len(clipper)):
        a, z = clipper[i], clipper[(i + 1) % len(clipper)]
        inp, out = out, []
        for j in range(len(inp)):
            cur, prev = inp[j], inp[j - 1]
            if inside(cur, a, z):
                if not inside(prev, a, z):
                    out.append(inter(prev, cur, a, z))
                out.append(cur)
            elif inside(prev, a, z):
                out.append(inter(prev, cur, a, z))
    return out


def _trap(cam, target, d_near, d_far, k=0.085):
    ux, uy = target[0] - cam[0], target[1] - cam[1]
    n = math.hypot(ux, uy)
    ux, uy = ux / n, uy / n
    px, py = -uy, ux

    def at(d, s):
        hw = k * d * s
        return (cam[0] + ux * d + px * hw, cam[1] + uy * d + py * hw)

    return [at(d_near, 1), at(d_far, 1), at(d_far, -1), at(d_near, -1)], (ux, uy)


def _ccw(poly_):
    a = sum(poly_[i][0] * poly_[(i + 1) % len(poly_)][1] - poly_[(i + 1) % len(poly_)][0] * poly_[i][1]
            for i in range(len(poly_)))
    return poly_ if a > 0 else poly_[::-1]


def blob(t):
    W, H = 1100, 480
    b = []
    fx0, fy0, fx1, fy1 = 32, 32, 740, 448
    b.append(R(fx0, fy0, fx1 - fx0, fy1 - fy0, t, r=12))
    for gx in range(fx0 + 60, fx1, 60):
        b.append(L(gx, fy0 + 1, gx, fy1 - 1, t, color="grid", sw=1))
    for gy in range(fy0 + 60, fy1, 60):
        b.append(L(fx0 + 1, gy, fx1 - 1, gy, t, color="grid", sw=1))
    b.append(eyebrow(52, 58, "Floor, top-down", t))

    camA, camB, F = (110, 400), (690, 100), (370, 250)
    dA = math.hypot(F[0] - camA[0], F[1] - camA[1])
    dB = math.hypot(F[0] - camB[0], F[1] - camB[1])
    trapA, _ = _trap(camA, F, dA - 34, dA + 16)
    over = 150
    trapB, uB = _trap(camB, F, dB + over - 290, dB + over + 16)
    N = (F[0] + uB[0] * over, F[1] + uB[1] * over)
    inter = _clip(_ccw(trapA), _ccw(trapB))

    # desk between B and the person
    dx, dy = camB[0] + uB[0] * (dB - 205), camB[1] + uB[1] * (dB - 205)
    b.append(R(dx - 42, dy - 20, 84, 40, t, fill="grid", stroke="stroke", r=4))
    b.append(T(dx, dy + 4, "desk", t, size=11, color="muted", anchor="middle", mono=True))

    # sight lines
    for cam, trap in ((camA, trapA), (camB, trapB)):
        for p in (trap[1], trap[2]):
            b.append(L(cam[0], cam[1], p[0], p[1], t, color="faint", sw=1, dash="2 5"))

    b.append(poly(trapB, t, "slate", "slate", opacity=0.14))
    b.append(poly(trapA, t, "olive", "olive", opacity=0.22))
    b.append(poly(inter, t, "clay", "clay", opacity=0.55, sw=2))

    # B's point-only estimate -> would be a second dot
    b.append(L(N[0], N[1], F[0], F[1], t, color="clay", sw=1.2, dash="3 4"))
    b.append(C(N[0], N[1], 7, t, fill="surface", stroke="clay", sw=2))
    b.append(T(N[0] - 16, N[1] + 4, "B's foot point alone", t, size=11.5, color="clay",
               anchor="end", weight=600))
    b.append(T(N[0] - 16, N[1] + 20, "≈ 1 m too far", t, size=11.5, color="muted",
               anchor="end"))
    b.append(C(F[0], F[1], 6.5, t, fill="clay_deep", stroke="bg", sw=2))
    b.append(L(F[0] - 8, F[1] - 8, F[0] - 40, F[1] - 60, t, color="text", sw=1))
    b.append(T(F[0] - 44, F[1] - 66, "fused position", t, size=12, color="text",
               weight=600, anchor="end"))

    def camera(c, target, label, sub, dx_, dy_):
        ang = math.degrees(math.atan2(target[1] - c[1], target[0] - c[0]))
        g = (f'<g transform="translate({c[0]},{c[1]}) rotate({ang:.1f})">'
             f'<rect x="-14" y="-10" width="20" height="20" rx="4" fill="{t["text"]}"/>'
             f'<path d="M6,-6 L16,-11 L16,11 L6,6 Z" fill="{t["text"]}"/></g>')
        return (g + T(c[0] + dx_, c[1] + dy_, label, t, size=12.5, weight=600)
                + T(c[0] + dx_, c[1] + dy_ + 16, sub, t, size=11.5, color="muted"))

    b.append(camera(camA, F, "Camera A", "sees the feet", 24, 18))
    b.append(camera(camB, F, "Camera B", "feet hidden by desk", -150, -30))

    # scale bar
    b.append(L(fx1 - 84, fy1 - 22, fx1 - 24, fy1 - 22, t, color="muted", sw=2))
    b.append(T(fx1 - 54, fy1 - 30, "1 m", t, size=10.5, color="muted", anchor="middle", mono=True))

    # legend
    lx, ly = 772, 70
    b.append(eyebrow(lx, ly - 12, "How to read it", t))
    items = [
        ("olive", 0.22, "Camera A trapezoid", ["Feet visible → short and tight."]),
        ("slate", 0.14, "Camera B trapezoid", ["Feet hidden → long, reaching",
                                               "toward the camera."]),
        ("clay", 0.55, "Overlap", ["Where the person is placed."]),
    ]
    y = ly + 16
    for col, op, head, lines in items:
        b.append(f'<rect x="{lx}" y="{y}" width="22" height="16" rx="3" fill="{t[col]}" '
                 f'fill-opacity="{op}" stroke="{t[col]}" stroke-width="1.5"/>')
        b.append(T(lx + 34, y + 13, head, t, size=13, weight=600))
        for i, ln in enumerate(lines):
            b.append(T(lx + 34, y + 32 + 16 * i, ln, t, size=12, color="muted"))
        y += 42 + 16 * len(lines)
    b.append(C(lx + 11, y + 8, 7, t, fill="surface", stroke="clay", sw=2))
    b.append(T(lx + 34, y + 13, "Point estimate alone", t, size=13, weight=600))
    b.append(T(lx + 34, y + 32, "Would become a second dot.", t, size=12, color="muted"))
    y += 76
    b.append(L(lx, y - 18, W - 32, y - 18, t, color="grid", sw=1))
    for i, ln in enumerate(["Occlusion only ever makes the feet",
                            "look farther away, so each trapezoid",
                            "reaches mostly toward its camera.",
                            "",
                            "No overlap (e.g. calibration error)?",
                            "Fall back to the point mean."]):
        b.append(T(lx, y + 18 * i, ln, t, size=12, color="muted"))
    return svg(W, H, b, t)


def fusion_steps(t):
    W, H = 1160, 230
    b = []
    cards = [
        ("1", "Cluster", ["Same person if ≤ 1.6 m apart", "or their trapezoids overlap."], "slate"),
        ("2", "Resolve", ["Inverse-variance mean (1/σ²);", "blob overlap when σ is high."], "clay"),
        ("3", "Match", ["Nearest track within 1.8 m,", "speed-capped at 2.5 m/s."], "clay"),
        ("4", "Lifecycle", ["Confirm after 3 frames,", "coast up to 1.5 s unseen."], "olive"),
    ]
    x0, cw, step = 40, 244, 272
    for i, (n, title, lines, acc) in enumerate(cards):
        x = x0 + i * step
        b.append(step_card(x, 36, cw, 132, t, f"STEP {n}", title, lines, None, accent=acc))
        if i < len(cards) - 1:
            b.append(L(x + cw + 4, 102, x + step - 5, 102, t, arrow=True))
    b.append(T(W / 2, 204, "Runs every loop step on the detections from all cameras. "
               "Defaults shown; all tunable in the fusion: block of config.yaml.",
               t, size=12, color="muted", anchor="middle"))
    return svg(W, H, b, t)


def lifecycle(t):
    W, H = 1160, 380
    b = []
    nodes = {
        "Tentative": (300, 190, "hidden", "stroke", "surface", "4 4"),
        "Confirmed": (640, 190, "shown", "clay", "clay_fill", None),
        "Coasting": (960, 190, "shown at last position", "slate", "slate_fill", None),
        "Removed": (640, 326, "dropped", "stroke", "surface", None),
    }
    nw, nh = 176, 58
    b.append(R(530, 58, 600, 184, t, fill="none", stroke="clay", r=14, dash="5 5"))
    b.append(eyebrow(548, 80, "Published to map + MQTT", t, color="clay"))
    for name, (cx, cy, sub, st, fl, dash) in nodes.items():
        b.append(R(cx - nw / 2, cy - nh / 2, nw, nh, t, fill=fl, stroke=st, r=10,
                   sw=1.5, dash=dash))
        b.append(T(cx, cy - 3, name, t, size=14.5, weight=600, anchor="middle",
                   color="muted" if name == "Removed" else "text"))
        b.append(T(cx, cy + 15, sub, t, size=11.5, color="muted", anchor="middle"))

    def label(x, y, s, anchor="middle", color="muted"):
        return T(x, y, s, t, size=12, color=color, anchor=anchor)

    # start
    b.append(C(84, 190, 8, t, fill="text"))
    b.append(L(94, 190, 300 - nw / 2 - 6, 190, t, arrow=True))
    b.append(label(152, 170, "new cluster"))
    b.append(label(152, 212, "no track ≤ 1.0 m", color="faint"))
    # tentative -> confirmed
    b.append(L(300 + nw / 2 + 4, 190, 640 - nw / 2 - 6, 190, t, arrow=True, color="clay"))
    b.append(label(470, 180, "matched 3 frames", color="clay"))
    # self loop
    b.append(P("M612,161 C600,112 680,112 668,161", t, arrow=True))
    b.append(label(640, 110, "matched · smoothed, ≤ 2.5 m/s"))
    # confirmed <-> coasting
    b.append(P(f"M{640 + nw / 2 + 4},178 L{960 - nw / 2 - 6},178", t, arrow=True))
    b.append(label(800, 168, "missed a frame"))
    b.append(P(f"M{960 - nw / 2 - 4},204 L{640 + nw / 2 + 6},204", t, arrow=True))
    b.append(label(800, 224, "matched again"))
    # to removed
    b.append(P(f"M300,{190 + nh / 2 + 4} C300,326 400,326 {640 - nw / 2 - 6},326", t,
               arrow=True, color="faint"))
    b.append(label(322, 266, "unseen 0.4 s", anchor="start", color="faint"))
    b.append(P(f"M960,{190 + nh / 2 + 4} C960,326 860,326 {640 + nw / 2 + 6},326", t,
               arrow=True, color="faint"))
    b.append(label(936, 266, "unseen 1.5 s", anchor="end", color="faint"))
    return svg(W, H, b, t)


def world(t):
    W, H = 960, 380
    b = []
    s, ox, oy = 50, 96, 318
    X = lambda m: ox + m * s          # noqa: E731
    Y = lambda m: oy - m * s          # noqa: E731
    b.append(R(ox - 30, Y(5) - 30, 9 * s + 60, 5 * s + 60, t, fill="surface", stroke="stroke",
               r=8, dash="4 4"))
    b.append(T(ox + 12, Y(5) - 12, "map canvas = scan footprint + margin_m", t, size=10.5,
               color="faint", mono=True))
    room = [(0, 0), (9, 0), (9, 3.5), (5.5, 3.5), (5.5, 5), (0, 5)]
    b.append(poly([(X(x), Y(y)) for x, y in room], t, "slate", "slate", opacity=0.10, sw=1.5))
    for m in range(0, 10):
        b.append(L(X(m), oy, X(m), oy + 5, t, color="muted", sw=1))
        b.append(T(X(m), oy + 20, str(m), t, size=10.5, color="faint", anchor="middle", mono=True))
    for m in range(0, 6):
        b.append(L(ox - 5, Y(m), ox, Y(m), t, color="muted", sw=1))
        if m:
            b.append(T(ox - 10, Y(m) + 4, str(m), t, size=10.5, color="faint", anchor="end",
                       mono=True))
    b.append(L(ox, oy, X(9.6), oy, t, color="text", sw=1.5, arrow=False))
    b.append(L(ox, oy, ox, Y(5.6), t, color="text", sw=1.5))
    b.append(T(X(9.6) + 6, oy + 4, "X (m)", t, size=12, weight=600))
    b.append(T(ox, Y(5.6) - 8, "Y (m)", t, size=12, weight=600, anchor="middle"))
    b.append(C(ox, oy, 4.5, t, fill="text"))
    b.append(T(ox + 8, oy - 8, "origin", t, size=11, color="muted"))

    for (x, y, lab) in ((3.2, 2.1, "P1 (3.2, 2.1)"), (6.8, 1.4, "P2 (6.8, 1.4)")):
        b.append(C(X(x), Y(y), 7, t, fill="clay", stroke="surface", sw=2))
        b.append(T(X(x) + 12, Y(y) + 4, lab, t, size=12, weight=600, mono=True))
    gx, gy = X(7.4), Y(4.3)
    b.append(C(gx, gy, 7, t, fill="surface", stroke="faint", sw=1.5))
    b.append(L(gx - 5, gy - 5, gx + 5, gy + 5, t, color="faint", sw=1.5))
    b.append(L(gx - 5, gy + 5, gx + 5, gy - 5, t, color="faint", sw=1.5))
    b.append(T(gx + 12, gy + 4, "outside room → ignored", t, size=11.5, color="muted"))

    lx = 640
    b.append(eyebrow(lx, 64, "World frame", t))
    rows = [("Units", "meters"), ("Origin", "min corner of the scan"),
            ("Source", "data/smart_lab.las"), ("Valid area", "auto-masked from scan"),
            ("Shared by", "every camera")]
    for i, (k, v) in enumerate(rows):
        y = 96 + 30 * i
        b.append(T(lx, y, k, t, size=12, color="muted"))
        b.append(T(lx + 88, y, v, t, size=12.5, mono=k == "Source"))
    b.append(L(lx, 252, W - 36, 252, t, color="grid", sw=1))
    b.append(R(lx, 268, 20, 14, t, fill="slate", stroke="slate", r=3, opacity=0.14))
    b.append(T(lx + 30, 280, "valid area (room)", t, size=12, color="muted"))
    b.append(C(lx + 10, 306, 6, t, fill="clay", stroke="surface", sw=2))
    b.append(T(lx + 30, 310, "tracked person", t, size=12, color="muted"))
    b.append(T(lx, 346, "Changing flip_x / flip_y / up_axis /", t, size=11.5, color="clay"))
    b.append(T(lx, 362, "clip_percentile moves the frame → recalibrate.", t, size=11.5,
               color="clay"))
    return svg(W, H, b, t)


def performance(t):
    W, H = 1160, 310
    b = []
    panels = [
        ("Loop rate", "frames per second · higher is better", 30, "FPS",
         [("Before", 3, 5), ("Now", 20, 30)], "~5×"),
        ("Detector", "milliseconds per step · lower is better", 50, "ms",
         [("Before", 46, 46), ("Now", 15, 15)], "~3×"),
        ("Cross-camera skew", "milliseconds · lower is better", 500, "ms",
         [("Before", 330, 450), ("Now", 40, 150)], "~4×"),
    ]
    x0, pw, gap = 32, 352, 18
    for i, (title, sub, vmax, unit, rows, gain) in enumerate(panels):
        px = x0 + i * (pw + gap)
        b.append(R(px, 32, pw, 220, t, r=12))
        b.append(T(px + 22, 62, title, t, size=15, weight=600))
        b.append(T(px + 22, 81, sub, t, size=11.5, color="muted"))
        b.append(T(px + pw - 22, 66, gain, t, size=24, weight=600, color="clay", anchor="end"))
        bx0, bw = px + 82, pw - 82 - 90
        for j, (lab, lo, hi) in enumerate(rows):
            y = 124 + 48 * j
            now = lab == "Now"
            col = "clay" if now else "faint"
            b.append(T(px + 22, y + 5, lab, t, size=12.5, color="text" if now else "muted",
                       weight=600 if now else 400))
            b.append(R(bx0, y - 9, bw, 18, t, fill="grid", stroke="grid", r=4, sw=0))
            b.append(R(bx0, y - 9, bw * lo / vmax, 18, t, fill=col, stroke=col, r=4, sw=0))
            if hi > lo:
                b.append(R(bx0 + bw * lo / vmax - 4, y - 9, bw * (hi - lo) / vmax + 4, 18, t,
                           fill=col, stroke=col, r=4, sw=0, opacity=0.4))
            val = f"{lo}–{hi}" if hi > lo else f"{lo}"
            b.append(T(bx0 + bw + 10, y + 5, f"{val} {unit}", t, size=12.5, mono=True,
                       color="text" if now else "muted", weight=600 if now else 400))
        b.append(T(bx0, 232, "0", t, size=10.5, color="faint", mono=True))
        b.append(T(bx0 + bw, 232, str(vmax), t, size=10.5, color="faint", mono=True,
                   anchor="end"))
    b.append(T(W / 2, 284, "Jetson AGX Thor, 3 RTSP cameras, headless. Before: PyTorch FP32 per "
               "camera + CPU face blur. Now: one batched TensorRT FP16 pass. Lighter bar = "
               "observed range.", t, size=12, color="muted", anchor="middle"))
    return svg(W, H, b, t)


DIAGRAMS = {
    "pipeline": pipeline, "calibration": calibration, "foot-point": foot_ladder,
    "sync": sync, "blob": blob, "fusion": fusion_steps, "lifecycle": lifecycle,
    "world-frame": world, "performance": performance,
}

if __name__ == "__main__":
    for name, fn in DIAGRAMS.items():
        for theme, t in THEMES.items():
            path = os.path.join(OUT, f"{name}-{theme}.svg")
            with open(path, "w") as f:
                f.write(fn(t))
    print(f"wrote {len(DIAGRAMS) * len(THEMES)} SVGs to {OUT}")
