#!/usr/bin/env python3
"""
patent_app.py — Native desktop app that builds 2005-era USPTO patent
drawings: aged color-grade, reference-number callouts, AND the diagram
itself (labeled block boxes + connector lines).

Run:    python patent_app.py
Needs:  pip install Pillow numpy
        (tkinter ships with Python on Win/macOS; Linux: sudo apt install python3-tk)

ELEMENTS
  Reference callout : ● anchor  +  ■ ref box  +  ◆ bends   (points at the image)
  Block box         : labeled rectangle (part of the diagram)
  Connector         : line between things, optional arrowheads + bends

EDIT CANVAS
  • Drag any handle to move it.            • Double-click a line  -> add a bend.
  • Drag a box corner ◣ to resize it.      • Right-click a bend   -> delete it.
  • Double-click a block box -> edit text. • Delete key           -> delete selected.
  • SNAP 45° makes segments lock to 0/45/90°.
"""

import math
import os
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageTk


# ════════════════════════════════════════════════════════════════════════════
#  IMAGE PROCESSING
# ════════════════════════════════════════════════════════════════════════════

_FONTS = [
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    "C:/Windows/Fonts/cour.ttf", "/Library/Fonts/Courier New.ttf",
    "/System/Library/Fonts/Courier New.ttf",
]
_BOLD = [
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "C:/Windows/Fonts/courbd.ttf", "/Library/Fonts/Courier New Bold.ttf",
]


def get_font(size, bold=False):
    size = max(6, int(size))
    for p in (_BOLD if bold else _FONTS):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return ImageFont.load_default()


def patent_grade(img, grayscale=False, strength=0.55):
    orig = np.array(img.convert("RGB"), dtype=np.float32)
    arr = orig.copy()
    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    keep = 0.0 if grayscale else 0.15
    arr = lum[..., None] * (1 - keep) + arr * keep
    a = arr / 255.0
    s = np.where(a < 0.5, 2 * a ** 2, 1 - 2 * (1 - a) ** 2)
    arr = (a * 0.45 + s * 0.55) * 255.0
    sharp = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    blur = np.array(sharp.filter(ImageFilter.GaussianBlur(0.8)), dtype=np.float32)
    arr = np.clip(arr + (arr - blur) * 0.7, 0, 255)
    rng = np.random.default_rng(42)
    arr = np.clip(arr + rng.normal(0, 2.8, arr.shape), 0, 255)
    pr, pg, pb = 248, 245, 232
    t = (255 - arr) / 255.0 * 0.11
    arr[..., 0] += (pr - arr[..., 0]) * t[..., 0]
    arr[..., 1] += (pg - arr[..., 1]) * t[..., 1]
    arr[..., 2] += (pb - arr[..., 2]) * t[..., 2]
    arr[..., 0] = np.clip(arr[..., 0] * 1.02, 0, 255)
    arr[..., 2] = np.clip(arr[..., 2] * 0.96, 0, 255)
    out = orig * (1 - strength) + arr * strength
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


INK = (14, 14, 14)
PALE = (240, 237, 218)
PAD = 60
TITLE_H = 68
PRESET_COLORS = ["#0e0e0e", "#1f3a93", "#9e1b1b", "#1f6b2e", "#5b4636", "#ffffff"]
PRESET_NAMES = ["Ink", "Blue", "Red", "Green", "Sepia", "White"]


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _dashed_seg(draw, x0, y0, x1, y1, fill=INK, dash=(7, 4)):
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    if L == 0:
        return
    ux, uy = dx / L, dy / L
    pos, on = 0.0, True
    while pos < L:
        seg = dash[0] if on else dash[1]
        end = min(pos + seg, L)
        if on:
            draw.line([(round(x0 + ux * pos), round(y0 + uy * pos)),
                       (round(x0 + ux * end), round(y0 + uy * end))], fill=fill, width=1)
        pos, on = end, not on


def _poly(draw, pts, fill=INK, dashed=False, width=1):
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        if dashed:
            _dashed_seg(draw, x0, y0, x1, y1, fill=fill)
        else:
            draw.line([(int(x0), int(y0)), (int(x1), int(y1))], fill=fill, width=width)


def _arrowhead(draw, x, y, fromx, fromy, size=8, fill=INK):
    dx, dy = x - fromx, y - fromy
    L = math.hypot(dx, dy)
    if L == 0:
        return
    ux, uy = dx / L, dy / L
    px, py = -uy, ux
    draw.polygon([(x, y),
                  (x - ux * size + px * size * 0.45, y - uy * size + py * size * 0.45),
                  (x - ux * size - px * size * 0.45, y - uy * size - py * size * 0.45)], fill=fill)


def _tsize(draw, text, fnt):
    bb = draw.textbbox((0, 0), text, font=fnt)
    return bb[2] - bb[0], bb[3] - bb[1]


def _nearest_edge(cx, cy, hw, hh, target):
    edges = [(cx + hw, cy), (cx - hw, cy), (cx, cy + hh), (cx, cy - hh)]
    return min(edges, key=lambda p: math.hypot(p[0] - target[0], p[1] - target[1]))


def render_patent(img, labels, boxes=None, connectors=None, fig_num=1, title="",
                  grayscale=False, grade_strength=0.55, font_scale=1.0,
                  dashed=True, arrow=False, border_style="double"):
    boxes = boxes or []
    connectors = connectors or []
    graded = patent_grade(img, grayscale=grayscale, strength=grade_strength)
    sw, sh = graded.size
    cw, ch = sw + PAD * 2, sh + PAD * 2 + TITLE_H
    canvas = Image.new("RGB", (cw, ch), (250, 247, 234))
    canvas.paste(graded, (PAD, PAD))
    draw = ImageDraw.Draw(canvas)

    f_ref = get_font(17 * font_scale, True)
    f_lbl = get_font(14 * font_scale)
    f_box = get_font(15 * font_scale, True)
    f_fig = get_font(24, True)
    f_ttl = get_font(19, True)

    # ── diagram block boxes ───────────────────────────────────────────────
    for b in boxes:
        x0 = PAD + b["x"] * sw
        y0 = PAD + b["y"] * sh
        w = max(8, b["w"] * sw)
        h = max(8, b["h"] * sh)
        draw.rectangle((x0, y0, x0 + w, y0 + h), fill=PALE, outline=INK, width=2)
        lines = b.get("text", "").split("\n")
        lh = (_tsize(draw, "Mg", f_box)[1]) + 4
        total = lh * len(lines)
        ty0 = y0 + (h - total) / 2
        for k, ln in enumerate(lines):
            tw, _ = _tsize(draw, ln, f_box)
            draw.text((x0 + (w - tw) / 2, ty0 + k * lh), ln, font=f_box, fill=INK)

    # ── connectors ────────────────────────────────────────────────────────
    for cn in connectors:
        col = hex_to_rgb(cn.get("color", "#0e0e0e"))
        pts = [(PAD + p["x"] * sw, PAD + p["y"] * sh) for p in cn["pts"]]
        if len(pts) < 2:
            continue
        _poly(draw, pts, fill=col, dashed=cn.get("dashed", False))
        if cn.get("arrow_end", True):
            _arrowhead(draw, int(pts[-1][0]), int(pts[-1][1]),
                       int(pts[-2][0]), int(pts[-2][1]), fill=col)
        if cn.get("arrow_start", False):
            _arrowhead(draw, int(pts[0][0]), int(pts[0][1]),
                       int(pts[1][0]), int(pts[1][1]), fill=col)

    # ── reference-number callouts ─────────────────────────────────────────
    for lbl in labels:
        ax = PAD + lbl["ax"] * sw
        ay = PAD + lbl["ay"] * sh
        lx = PAD + lbl["lx"] * sw
        ly = PAD + lbl["ly"] * sh
        ref, text = lbl["ref"], lbl["text"]
        wps = [(PAD + w["x"] * sw, PAD + w["y"] * sh) for w in lbl.get("waypoints", [])]
        col = hex_to_rgb(lbl.get("color", "#0e0e0e"))

        rw, rh = _tsize(draw, ref, f_ref)
        tw, th = (_tsize(draw, text, f_lbl) if text else (0, rh))
        gap, dw, cp = 6, 18, 7
        bw = cp * 2 + rw + (gap + dw + gap + tw if text else 0)
        bh = cp * 2 + max(rh, th) + 6
        bx = max(4, min(lx - bw // 2, cw - bw - 4))
        by = max(4, min(ly - bh // 2, ch - bh - 4))
        bcx, bcy = bx + bw / 2, by + bh / 2
        draw.rectangle((bx, by, bx + bw, by + bh), fill=PALE, outline=INK, width=1)
        draw.text((bx + cp, by + cp + 2), ref, font=f_ref, fill=INK)
        if text:
            my = by + bh // 2
            draw.line([(bx + cp + rw + gap, my), (bx + cp + rw + gap + dw, my)], fill=INK, width=1)
            draw.text((bx + cp + rw + gap + dw + gap, by + cp + 2), text, font=f_lbl, fill=INK)
        first_target = wps[0] if wps else (ax, ay)
        edge = _nearest_edge(bcx, bcy, bw / 2, bh / 2, first_target)
        path = [edge] + wps + [(ax, ay)]
        _poly(draw, path, fill=col, dashed=dashed)
        if arrow:
            _arrowhead(draw, int(ax), int(ay), int(path[-2][0]), int(path[-2][1]), fill=col)
        draw.ellipse((ax - 4, ay - 4, ax + 4, ay + 4), fill=col)

    # ── frame + title ─────────────────────────────────────────────────────
    m = 9
    if border_style in ("double", "single"):
        draw.rectangle((m, m, cw - m, ch - m), outline=INK, width=2)
    if border_style == "double":
        draw.rectangle((m + 5, m + 5, cw - m - 5, ch - m - 5), outline=INK, width=1)
    ty = ch - TITLE_H
    if border_style != "none":
        draw.line([(m, ty), (cw - m, ty)], fill=INK, width=1)
    draw.text((m + 20, ty + TITLE_H // 2 - 12), f"FIG. {fig_num}", font=f_fig, fill=INK)
    if title:
        tw2, _ = _tsize(draw, title, f_ttl)
        draw.text(((cw - tw2) // 2, ty + TITLE_H // 2 - 10), title, font=f_ttl, fill=INK)
    if border_style == "double":
        for cx, cy in [(m + 3, m + 3), (cw - m - 3, m + 3),
                       (m + 3, ch - m - 3), (cw - m - 3, ch - m - 3)]:
            draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=INK)
    return canvas


# ════════════════════════════════════════════════════════════════════════════
#  APP
# ════════════════════════════════════════════════════════════════════════════

BG = "#1e1e1e"; PANEL = "#262626"; ACCENT = "#f0c040"; BLUE = "#7aacee"
GREEN = "#7ec87e"; FG = "#e8e4d8"; SUBTLE = "#888888"; MONO = ("Courier New", 10)
HANDLE_R = 7; WP_R = 6; CORNER = 8


def snap45(px, py, x, y):
    dx, dy = x - px, y - py
    if dx == 0 and dy == 0:
        return x, y
    step = math.pi / 4
    ang = round(math.atan2(dy, dx) / step) * step
    dist = math.hypot(dx, dy)
    return px + math.cos(ang) * dist, py + math.sin(ang) * dist


class PatentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Patent Diagram Tool")
        self.geometry("1280x880")
        self.configure(bg=BG)
        self.minsize(1040, 680)

        self.source_img = None
        self.labels = []
        self.boxes = []
        self.connectors = []
        self.sel = None               # (kind, idx)  kind in label|box|conn
        self.placing = False
        self.drag = None
        self.current_color = PRESET_COLORS[0]
        self._swatches = []

        self.dx = self.dy = 0
        self.dw = self.dh = 1
        self._src_photo = None
        self._out_photo = None
        self._rendered = None
        self._dirty_job = None

        self._build_ui()
        self.bind("<Delete>", lambda e: self.delete_selected())
        self.bind("<BackSpace>", lambda e: self.delete_selected())

    # ── UI ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        side = tk.Frame(self, bg=PANEL, width=300)
        side.pack(side="left", fill="y"); side.pack_propagate(False)
        outer = tk.Frame(side, bg=PANEL); outer.pack(fill="both", expand=True)
        canv = tk.Canvas(outer, bg=PANEL, highlightthickness=0, width=284)
        sb = ttk.Scrollbar(outer, orient="vertical", command=canv.yview)
        s = tk.Frame(canv, bg=PANEL)
        s.bind("<Configure>", lambda e: canv.configure(scrollregion=canv.bbox("all")))
        canv.create_window((0, 0), window=s, anchor="nw", width=284)
        canv.configure(yscrollcommand=sb.set)
        canv.pack(side="left", fill="both", expand=True); sb.pack(side="right", fill="y")
        canv.bind_all("<MouseWheel>", lambda e: canv.yview_scroll(int(-e.delta / 120), "units"))
        canv.bind_all("<Button-4>", lambda e: canv.yview_scroll(-1, "units"))
        canv.bind_all("<Button-5>", lambda e: canv.yview_scroll(1, "units"))

        tk.Label(s, text="PATENT DIAGRAM TOOL", bg=PANEL, fg=FG,
                 font=("Courier New", 12, "bold")).pack(anchor="w", padx=14, pady=(14, 0))
        tk.Label(s, text="USPTO-STYLE BUILDER", bg=PANEL, fg="#555",
                 font=("Courier New", 7)).pack(anchor="w", padx=14)
        self._btn(s, "📂  OPEN IMAGE", self.open_image, BLUE, "#16243f"
                  ).pack(fill="x", padx=14, pady=(14, 2))
        self._btn(s, "▱  BLANK CANVAS", self.blank_canvas, "#bbb", "#2a2a2a"
                  ).pack(fill="x", padx=14, pady=(0, 4))

        self._head(s, "FIGURE")
        self._flabel(s, "NUMBER")
        self.fig_var = tk.StringVar(value="1")
        self.fig_var.trace_add("write", lambda *_: self.schedule_render()); self._entry(s, self.fig_var)
        self._flabel(s, "TITLE")
        self.title_var = tk.StringVar(value="")
        self.title_var.trace_add("write", lambda *_: self.schedule_render()); self._entry(s, self.title_var)

        self._head(s, "DIAGRAM ELEMENTS")
        self._btn(s, "▭  ADD BLOCK BOX", self.add_box, GREEN, "#16301a"
                  ).pack(fill="x", padx=14, pady=(0, 3))
        self._btn(s, "→  ADD CONNECTOR", self.add_connector, GREEN, "#16301a"
                  ).pack(fill="x", padx=14, pady=(0, 3))
        self._btn(s, "①  ADD REF CALLOUT", self.start_placing, GREEN, "#16301a"
                  ).pack(fill="x", padx=14, pady=(0, 3))
        self.status = tk.Label(s, text="Drag handles to move. Double-click a\n"
                                       "line to bend it, a box to rename it.",
                               bg=PANEL, fg=SUBTLE, font=("Courier New", 8),
                               wraplength=258, justify="left")
        self.status.pack(anchor="w", padx=14, pady=(2, 4))

        self._head(s, "SELECTED ELEMENT")
        self.sel_label = tk.Label(s, text="(none)", bg=PANEL, fg=ACCENT,
                                  font=("Courier New", 9), wraplength=258, justify="left")
        self.sel_label.pack(anchor="w", padx=14, pady=(0, 4))
        gr = tk.Frame(s, bg=PANEL); gr.pack(fill="x", padx=14)
        self._btn(gr, "RENAME", self.rename_selected, "#bbb", "#2a2a2a"
                  ).pack(side="left", fill="x", expand=True, padx=(0, 3))
        self._btn(gr, "DELETE", self.delete_selected, "#c77", "#3a1a1a"
                  ).pack(side="left", fill="x", expand=True, padx=(3, 0))
        self._btn(s, "+ ADD BEND TO SELECTED LINE", self.add_bend, ACCENT, "#332b14"
                  ).pack(fill="x", padx=14, pady=(4, 0))
        cr = tk.Frame(s, bg=PANEL); cr.pack(fill="x", padx=14, pady=(3, 0))
        self._btn(cr, "↤ START ARROW", lambda: self.toggle_arrow("start"), "#bbb", "#2a2a2a"
                  ).pack(side="left", fill="x", expand=True, padx=(0, 3))
        self._btn(cr, "END ARROW ↦", lambda: self.toggle_arrow("end"), "#bbb", "#2a2a2a"
                  ).pack(side="left", fill="x", expand=True, padx=(3, 0))
        self._btn(s, "TOGGLE DASHED (line)", self.toggle_dashed_conn, "#bbb", "#2a2a2a"
                  ).pack(fill="x", padx=14, pady=(3, 0))

        self._flabel(s, "PATH COLOR (selected line)")
        sw_row = tk.Frame(s, bg=PANEL); sw_row.pack(fill="x", padx=14, pady=(0, 2))
        self._swatches = []
        for hexc in PRESET_COLORS:
            holder = tk.Frame(sw_row, bg=PANEL, highlightthickness=2,
                              highlightbackground=PANEL, cursor="hand2")
            holder.pack(side="left", padx=2)
            chip = tk.Frame(holder, bg=hexc, width=30, height=22,
                            highlightthickness=1, highlightbackground="#555", cursor="hand2")
            chip.pack_propagate(False); chip.pack()
            for w in (holder, chip):
                w.bind("<Button-1>", lambda e, h=hexc: self.set_path_color(h))
            self._swatches.append(holder)
        self._refresh_swatches()

        self._head(s, "APPEARANCE")
        self._flabel(s, "GRADE STRENGTH")
        self.grade_var = tk.DoubleVar(value=0.55); self._scale(s, self.grade_var, 0, 1)
        self._flabel(s, "TEXT / LABEL SIZE")
        self.fsize_var = tk.DoubleVar(value=1.0); self._scale(s, self.fsize_var, 0.6, 1.8)
        self.gray_var = tk.BooleanVar(value=False); self._check(s, "GREYSCALE", self.gray_var)
        self.dash_var = tk.BooleanVar(value=True); self._check(s, "DASHED CALLOUT LINES", self.dash_var)
        self.arrow_var = tk.BooleanVar(value=False); self._check(s, "CALLOUT ARROWHEADS", self.arrow_var)
        self.snap_var = tk.BooleanVar(value=True); self._check(s, "SNAP LINES TO 45°", self.snap_var)
        self._flabel(s, "BORDER STYLE")
        self.border_var = tk.StringVar(value="double")
        self.border_var.trace_add("write", lambda *_: self.schedule_render())
        bf = tk.Frame(s, bg=PANEL); bf.pack(fill="x", padx=14)
        for txt in ("double", "single", "none"):
            tk.Radiobutton(bf, text=txt, value=txt, variable=self.border_var, bg=PANEL,
                           fg=SUBTLE, selectcolor="#141414", activebackground=PANEL,
                           activeforeground=FG, font=("Courier New", 8),
                           highlightthickness=0).pack(side="left")

        self._head(s, "OUTPUT")
        self._btn(s, "↓  SAVE PNG", self.save_png, GREEN, "#16301a"
                  ).pack(fill="x", padx=14, pady=(0, 16))

        right = tk.Frame(self, bg=BG); right.pack(side="left", fill="both", expand=True)
        tk.Label(right, text="EDIT  —  drag handles · dbl-click line=bend · dbl-click box=rename · right-click bend=delete · Del=remove",
                 bg=BG, fg="#666", font=("Courier New", 8)).pack(anchor="w", padx=16, pady=(12, 4))
        self.src_canvas = tk.Canvas(right, bg="#111", highlightthickness=1,
                                    highlightbackground="#2a2a2a", height=380)
        self.src_canvas.pack(fill="x", padx=16)
        for ev, fn in [("<Button-1>", self.on_press), ("<B1-Motion>", self.on_drag),
                       ("<ButtonRelease-1>", self.on_release), ("<Double-Button-1>", self.on_double),
                       ("<Button-3>", self.on_right), ("<Configure>", lambda e: self.redraw_source())]:
            self.src_canvas.bind(ev, fn)
        tk.Label(right, text="OUTPUT PREVIEW", bg=BG, fg="#666",
                 font=("Courier New", 8)).pack(anchor="w", padx=16, pady=(14, 4))
        ow = tk.Frame(right, bg=BG); ow.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.out_canvas = tk.Canvas(ow, bg="#0d0d0d", highlightthickness=1,
                                    highlightbackground="#2a2a2a")
        self.out_canvas.pack(fill="both", expand=True)
        self.out_canvas.bind("<Configure>", lambda e: self.redraw_output())
        self._show_empty()

    # widget factories
    def _btn(self, p, text, cmd, fg=FG, bg="#333"):
        return tk.Button(p, text=text, command=cmd, fg=fg, bg=bg, activeforeground=fg,
                         activebackground=bg, font=("Courier New", 9, "bold"), relief="flat",
                         bd=0, padx=6, pady=6, cursor="hand2", highlightthickness=0)

    def _head(self, p, txt):
        tk.Label(p, text=txt, bg=PANEL, fg=SUBTLE,
                 font=("Courier New", 8, "bold")).pack(anchor="w", padx=14, pady=(14, 6))

    def _flabel(self, p, txt):
        tk.Label(p, text=txt, bg=PANEL, fg=SUBTLE,
                 font=("Courier New", 8)).pack(anchor="w", padx=14, pady=(8, 2))

    def _entry(self, p, var):
        tk.Entry(p, textvariable=var, bg="#141414", fg=FG, insertbackground=FG, font=MONO,
                 relief="flat", highlightthickness=1, highlightbackground="#3a3a3a"
                 ).pack(fill="x", padx=14, ipady=3)

    def _scale(self, p, var, lo, hi):
        tk.Scale(p, variable=var, from_=lo, to=hi, resolution=0.01, orient="horizontal",
                 bg=PANEL, fg=SUBTLE, troughcolor="#141414", highlightthickness=0,
                 font=("Courier New", 7), sliderrelief="flat", activebackground=ACCENT,
                 command=lambda *_: self.schedule_render()).pack(fill="x", padx=12)

    def _check(self, p, txt, var):
        var.trace_add("write", lambda *_: self.schedule_render())
        tk.Checkbutton(p, text=txt, variable=var, bg=PANEL, fg=SUBTLE, selectcolor="#141414",
                       activebackground=PANEL, activeforeground=FG, font=("Courier New", 9),
                       relief="flat", highlightthickness=0).pack(anchor="w", padx=10, pady=(6, 0))

    def _show_empty(self):
        self.src_canvas.delete("all")
        self.src_canvas.create_text(420, 180, text="OPEN AN IMAGE  OR  USE A BLANK CANVAS",
                                    fill="#444", font=("Courier New", 11))

    def _opts(self):
        return dict(fig_num=self.fig_var.get() or "1", title=self.title_var.get().strip(),
                    grayscale=self.gray_var.get(), grade_strength=float(self.grade_var.get()),
                    font_scale=float(self.fsize_var.get()), dashed=self.dash_var.get(),
                    arrow=self.arrow_var.get(), border_style=self.border_var.get())

    # ── files / canvas init ────────────────────────────────────────────────
    def open_image(self):
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.gif *.bmp *.webp *.tif *.tiff"),
                       ("All files", "*.*")])
        if not path:
            return
        try:
            self.source_img = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open image:\n{e}"); return
        self._reset_elements(); self.redraw_source(); self.render_now()

    def blank_canvas(self):
        # plain off-white sheet to draw a pure block diagram on
        self.source_img = Image.new("RGB", (900, 620), (250, 248, 238))
        self._reset_elements(); self.redraw_source(); self.render_now()

    def _reset_elements(self):
        self.labels.clear(); self.boxes.clear(); self.connectors.clear()
        self.sel = None; self.cancel_placing(); self._update_sel_label(); self._refresh_swatches()

    def save_png(self):
        if self.source_img is None:
            messagebox.showinfo("No image", "Open an image or blank canvas first."); return
        path = filedialog.asksaveasfilename(title="Save patent PNG", defaultextension=".png",
                                            initialfile="patent.png",
                                            filetypes=[("PNG image", "*.png")])
        if not path:
            return
        render_patent(self.source_img, self.labels, self.boxes, self.connectors,
                      **self._opts()).save(path)
        messagebox.showinfo("Saved", f"Saved:\n{path}")

    # ── coords ──────────────────────────────────────────────────────────────
    def c2n(self, x, y): return (x - self.dx) / self.dw, (y - self.dy) / self.dh
    def n2c(self, nx, ny): return self.dx + nx * self.dw, self.dy + ny * self.dh

    def label_path_canvas(self, l):
        ax, ay = self.n2c(l["ax"], l["ay"])
        lx, ly = self.n2c(l["lx"], l["ly"])
        wps = [self.n2c(w["x"], w["y"]) for w in l.get("waypoints", [])]
        first = wps[0] if wps else (ax, ay)
        edge = _nearest_edge(lx, ly, 24, 12, first)
        return [edge] + wps + [(ax, ay)]

    # ── element creation ──────────────────────────────────────────────────
    def _require_canvas(self):
        if self.source_img is None:
            messagebox.showinfo("No canvas", "Open an image or blank canvas first.")
            return False
        return True

    def add_box(self):
        if not self._require_canvas():
            return
        text = simpledialog.askstring("Block box", "Box label:", parent=self, initialvalue="BLOCK")
        if text is None:
            return
        self.boxes.append({"x": 0.40, "y": 0.42, "w": 0.20, "h": 0.14, "text": text.strip()})
        self.sel = ("box", len(self.boxes) - 1)
        self._after_change()

    def add_connector(self):
        if not self._require_canvas():
            return
        self.connectors.append({"pts": [{"x": 0.35, "y": 0.50}, {"x": 0.62, "y": 0.50}],
                                "color": self.current_color, "arrow_start": False,
                                "arrow_end": True, "dashed": False})
        self.sel = ("conn", len(self.connectors) - 1)
        self._after_change()

    def start_placing(self):
        if not self._require_canvas():
            return
        self.placing = True
        self.src_canvas.config(cursor="crosshair")
        self.status.config(text="↗ Click on the canvas to drop the callout anchor.", fg=ACCENT)

    def cancel_placing(self):
        self.placing = False
        self.src_canvas.config(cursor="")
        self.status.config(text="Drag handles to move. Double-click a\n"
                                "line to bend it, a box to rename it.", fg=SUBTLE)

    def _next_ref(self):
        nums = [int(l["ref"]) for l in self.labels if l["ref"].isdigit()]
        return str(max(nums) + 2) if nums else "100"

    def _after_change(self):
        self._update_sel_label(); self._refresh_swatches()
        self.update_overlays(); self.render_now()

    # ── mouse: press ──────────────────────────────────────────────────────
    def on_press(self, event):
        if self.source_img is None:
            return
        if self.placing:
            nx, ny = self.c2n(event.x, event.y)
            if not (0 <= nx <= 1 and 0 <= ny <= 1):
                return
            ref = simpledialog.askstring("Reference numeral", "Reference numeral (e.g. 102):",
                                         parent=self, initialvalue=self._next_ref())
            if ref is None:
                return
            text = simpledialog.askstring("Description", "Label description:",
                                          parent=self, initialvalue="")
            if text is None:
                text = ""
            lx = max(-0.18, min(1.18, nx - 0.32 if nx > 0.5 else nx + 0.32))
            ly = max(-0.05, min(1.05, ny - 0.18 if ny > 0.5 else ny + 0.18))
            self.labels.append({"ref": ref.strip(), "text": text.strip(), "ax": nx, "ay": ny,
                                "lx": lx, "ly": ly, "waypoints": [], "color": self.current_color})
            self.sel = ("label", len(self.labels) - 1)
            self.cancel_placing(); self._after_change()
            return
        self.drag = self._hit(event.x, event.y)
        self.sel = (self.drag["kind"], self.drag["index"]) if self.drag else None
        self._update_sel_label(); self._refresh_swatches(); self.update_overlays()

    def _hit(self, x, y):
        # callout handles
        for i, l in enumerate(self.labels):
            ax, ay = self.n2c(l["ax"], l["ay"])
            if (x - ax) ** 2 + (y - ay) ** 2 <= (HANDLE_R + 4) ** 2:
                return {"kind": "label", "index": i, "part": "anchor"}
            for j, w in enumerate(l.get("waypoints", [])):
                wx, wy = self.n2c(w["x"], w["y"])
                if (x - wx) ** 2 + (y - wy) ** 2 <= (WP_R + 4) ** 2:
                    return {"kind": "label", "index": i, "part": "wp", "wp": j}
            lx, ly = self.n2c(l["lx"], l["ly"])
            if abs(x - lx) <= 26 and abs(y - ly) <= 14:
                return {"kind": "label", "index": i, "part": "box"}
        # connector points
        for i, cn in enumerate(self.connectors):
            for j, p in enumerate(cn["pts"]):
                px, py = self.n2c(p["x"], p["y"])
                if (x - px) ** 2 + (y - py) ** 2 <= (WP_R + 5) ** 2:
                    return {"kind": "conn", "index": i, "part": "pt", "pt": j}
        # box resize corner, then body
        for i, b in enumerate(self.boxes):
            bx, by = self.n2c(b["x"], b["y"])
            bw, bh = b["w"] * self.dw, b["h"] * self.dh
            if abs(x - (bx + bw)) <= CORNER and abs(y - (by + bh)) <= CORNER:
                return {"kind": "box", "index": i, "part": "resize"}
            if bx <= x <= bx + bw and by <= y <= by + bh:
                ox, oy = x - bx, y - by
                return {"kind": "box", "index": i, "part": "move", "ox": ox, "oy": oy}
        return None

    # ── mouse: drag ─────────────────────────────────────────────────────────
    def _conn_ref(self, cn, j):
        pts = cn["pts"]
        if j == 0 and len(pts) > 1:
            return self.n2c(pts[1]["x"], pts[1]["y"])
        if j > 0:
            return self.n2c(pts[j - 1]["x"], pts[j - 1]["y"])
        return None

    def _label_ref(self, l, part, wp=None):
        wps = l.get("waypoints", [])
        if part == "anchor":
            return self.n2c(wps[-1]["x"], wps[-1]["y"]) if wps else self.n2c(l["lx"], l["ly"])
        if part == "box":
            return self.n2c(wps[0]["x"], wps[0]["y"]) if wps else self.n2c(l["ax"], l["ay"])
        if wp > 0:
            return self.n2c(wps[wp - 1]["x"], wps[wp - 1]["y"])
        return self.n2c(l["lx"], l["ly"])

    def on_drag(self, event):
        if not self.drag:
            return
        d = self.drag; i = d["index"]; x, y = event.x, event.y
        if d["kind"] == "label":
            l = self.labels[i]
            if self.snap_var.get() and d["part"] != "box":
                rp = self._label_ref(l, d["part"], d.get("wp"))
                x, y = snap45(rp[0], rp[1], x, y)
            nx, ny = self.c2n(x, y)
            if d["part"] == "anchor":
                l["ax"] = min(1, max(0, nx)); l["ay"] = min(1, max(0, ny))
            elif d["part"] == "box":
                l["lx"] = min(1.25, max(-0.25, nx)); l["ly"] = min(1.1, max(-0.1, ny))
            else:
                w = l["waypoints"][d["wp"]]
                w["x"] = min(1.25, max(-0.25, nx)); w["y"] = min(1.1, max(-0.1, ny))
        elif d["kind"] == "conn":
            cn = self.connectors[i]; j = d["pt"]
            if self.snap_var.get():
                rp = self._conn_ref(cn, j)
                if rp:
                    x, y = snap45(rp[0], rp[1], x, y)
            nx, ny = self.c2n(x, y)
            cn["pts"][j]["x"] = min(1.2, max(-0.2, nx))
            cn["pts"][j]["y"] = min(1.2, max(-0.2, ny))
        elif d["kind"] == "box":
            b = self.boxes[i]
            nx, ny = self.c2n(x, y)
            if d["part"] == "resize":
                b["w"] = max(0.04, nx - b["x"]); b["h"] = max(0.03, ny - b["y"])
            else:
                offx, offy = self.c2n(d["ox"] + self.dx, d["oy"] + self.dy)
                b["x"] = nx - offx; b["y"] = ny - offy
        self.update_overlays(); self.schedule_render()

    def on_release(self, event):
        if self.drag:
            self.drag = None; self.render_now()

    # ── mouse: double-click & right-click ────────────────────────────────────
    def on_double(self, event):
        if self.source_img is None or self.placing:
            return
        # box rename?
        for i, b in enumerate(self.boxes):
            bx, by = self.n2c(b["x"], b["y"])
            bw, bh = b["w"] * self.dw, b["h"] * self.dh
            if bx <= event.x <= bx + bw and by <= event.y <= by + bh:
                self.sel = ("box", i); self.rename_selected(); return
        # add bend to nearest connector or callout segment
        best = None; bestd = 16 ** 2
        for i, cn in enumerate(self.connectors):
            pts = [self.n2c(p["x"], p["y"]) for p in cn["pts"]]
            for k in range(len(pts) - 1):
                d2 = self._seg_dist2(event.x, event.y, pts[k], pts[k + 1])
                if d2 < bestd:
                    bestd = d2; best = ("conn", i, k)
        for i, l in enumerate(self.labels):
            path = self.label_path_canvas(l)
            for k in range(len(path) - 1):
                d2 = self._seg_dist2(event.x, event.y, path[k], path[k + 1])
                if d2 < bestd:
                    bestd = d2; best = ("label", i, k)
        if not best:
            return
        kind, i, k = best
        nx, ny = self.c2n(event.x, event.y)
        if kind == "conn":
            self.connectors[i]["pts"].insert(k + 1, {"x": nx, "y": ny})
            self.sel = ("conn", i)
        else:
            self.labels[i].setdefault("waypoints", []).insert(k, {"x": nx, "y": ny})
            self.sel = ("label", i)
        self._after_change()

    def on_right(self, event):
        # delete a connector interior point or a callout waypoint
        for i, cn in enumerate(self.connectors):
            for j in range(1, len(cn["pts"]) - 1):
                px, py = self.n2c(cn["pts"][j]["x"], cn["pts"][j]["y"])
                if (event.x - px) ** 2 + (event.y - py) ** 2 <= (WP_R + 5) ** 2:
                    cn["pts"].pop(j); self._after_change(); return
        for i, l in enumerate(self.labels):
            for j, w in enumerate(l.get("waypoints", [])):
                wx, wy = self.n2c(w["x"], w["y"])
                if (event.x - wx) ** 2 + (event.y - wy) ** 2 <= (WP_R + 5) ** 2:
                    l["waypoints"].pop(j); self._after_change(); return

    @staticmethod
    def _seg_dist2(px, py, a, b):
        ax, ay = a; bx, by = b
        dx, dy = bx - ax, by - ay
        if dx == 0 and dy == 0:
            return (px - ax) ** 2 + (py - ay) ** 2
        t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
        cx, cy = ax + t * dx, ay + t * dy
        return (px - cx) ** 2 + (py - cy) ** 2

    # ── selected-element actions ──────────────────────────────────────────
    def _update_sel_label(self):
        if not self.sel:
            self.sel_label.config(text="(none)"); return
        kind, i = self.sel
        if kind == "label" and i < len(self.labels):
            l = self.labels[i]
            self.sel_label.config(text=f"Callout {l['ref']}  {l['text']}")
        elif kind == "box" and i < len(self.boxes):
            self.sel_label.config(text=f"Block box: {self.boxes[i]['text']}")
        elif kind == "conn" and i < len(self.connectors):
            cn = self.connectors[i]
            self.sel_label.config(text=f"Connector ({len(cn['pts'])} pts)")
        else:
            self.sel = None; self.sel_label.config(text="(none)")

    def rename_selected(self):
        if not self.sel:
            return
        kind, i = self.sel
        if kind == "box" and i < len(self.boxes):
            t = simpledialog.askstring("Box label", "Box text (use \\n for new line):",
                                       parent=self, initialvalue=self.boxes[i]["text"])
            if t is not None:
                self.boxes[i]["text"] = t.replace("\\n", "\n").strip()
        elif kind == "label" and i < len(self.labels):
            l = self.labels[i]
            ref = simpledialog.askstring("Reference", "Reference numeral:",
                                         parent=self, initialvalue=l["ref"])
            if ref is None:
                return
            text = simpledialog.askstring("Description", "Label description:",
                                          parent=self, initialvalue=l["text"])
            l["ref"] = ref.strip()
            if text is not None:
                l["text"] = text.strip()
        else:
            messagebox.showinfo("Connector", "Connectors have no text to rename."); return
        self._after_change()

    def delete_selected(self):
        if not self.sel:
            return
        kind, i = self.sel
        arr = {"label": self.labels, "box": self.boxes, "conn": self.connectors}[kind]
        if 0 <= i < len(arr):
            arr.pop(i)
        self.sel = None; self._after_change()

    def add_bend(self):
        if not self.sel:
            messagebox.showinfo("Select a line", "Select a connector or callout first."); return
        kind, i = self.sel
        if kind == "conn":
            pts = [self.n2c(p["x"], p["y"]) for p in self.connectors[i]["pts"]]
            k = max(range(len(pts) - 1),
                    key=lambda j: (pts[j + 1][0] - pts[j][0]) ** 2 + (pts[j + 1][1] - pts[j][1]) ** 2)
            mx, my = (pts[k][0] + pts[k + 1][0]) / 2, (pts[k][1] + pts[k + 1][1]) / 2
            nx, ny = self.c2n(mx, my)
            self.connectors[i]["pts"].insert(k + 1, {"x": nx, "y": ny})
        elif kind == "label":
            path = self.label_path_canvas(self.labels[i])
            k = max(range(len(path) - 1),
                    key=lambda j: (path[j + 1][0] - path[j][0]) ** 2 + (path[j + 1][1] - path[j][1]) ** 2)
            mx, my = (path[k][0] + path[k + 1][0]) / 2, (path[k][1] + path[k + 1][1]) / 2
            nx, ny = self.c2n(mx, my)
            self.labels[i].setdefault("waypoints", []).insert(k, {"x": nx, "y": ny})
        else:
            messagebox.showinfo("Box", "Boxes don't have bends."); return
        self._after_change()

    def toggle_arrow(self, end):
        if not self.sel or self.sel[0] != "conn":
            messagebox.showinfo("Connector", "Select a connector first."); return
        cn = self.connectors[self.sel[1]]
        key = "arrow_start" if end == "start" else "arrow_end"
        cn[key] = not cn.get(key, end == "end")
        self._after_change()

    def toggle_dashed_conn(self):
        if not self.sel or self.sel[0] != "conn":
            messagebox.showinfo("Connector", "Select a connector first."); return
        cn = self.connectors[self.sel[1]]
        cn["dashed"] = not cn.get("dashed", False)
        self._after_change()

    # ── color ─────────────────────────────────────────────────────────────
    def set_path_color(self, hexc):
        self.current_color = hexc
        if self.sel and self.sel[0] in ("label", "conn"):
            arr = self.labels if self.sel[0] == "label" else self.connectors
            if self.sel[1] < len(arr):
                arr[self.sel[1]]["color"] = hexc
                self.update_overlays(); self.render_now()
        self._refresh_swatches()

    def _refresh_swatches(self):
        active = self.current_color
        if self.sel and self.sel[0] in ("label", "conn"):
            arr = self.labels if self.sel[0] == "label" else self.connectors
            if self.sel[1] < len(arr):
                active = arr[self.sel[1]].get("color", self.current_color)
        for b, hexc in zip(self._swatches, PRESET_COLORS):
            b.config(highlightbackground=(ACCENT if hexc == active else PANEL))

    # ── edit canvas drawing ─────────────────────────────────────────────────
    def redraw_source(self):
        self.src_canvas.delete("all")
        if self.source_img is None:
            self._show_empty(); return
        cw = self.src_canvas.winfo_width(); ch = self.src_canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        iw, ih = self.source_img.size
        scale = min(cw / iw, ch / ih, 1.0)
        self.dw, self.dh = max(1, int(iw * scale)), max(1, int(ih * scale))
        self.dx = (cw - self.dw) // 2; self.dy = (ch - self.dh) // 2
        disp = self.source_img.resize((self.dw, self.dh), Image.LANCZOS)
        self._src_photo = ImageTk.PhotoImage(disp)
        self.src_canvas.create_image(self.dx, self.dy, anchor="nw", image=self._src_photo, tags="img")
        self.update_overlays()

    def update_overlays(self):
        c = self.src_canvas; c.delete("ov")
        # block boxes
        for i, b in enumerate(self.boxes):
            on = (self.sel == ("box", i))
            bx, by = self.n2c(b["x"], b["y"])
            bw, bh = b["w"] * self.dw, b["h"] * self.dh
            c.create_rectangle(bx, by, bx + bw, by + bh, outline=ACCENT if on else "#e0ddd0",
                               width=2 if on else 1, fill="#1c1c1c", tags="ov")
            c.create_text(bx + bw / 2, by + bh / 2, text=b["text"].replace("\n", " "),
                          fill="#e8e4d8", font=("Courier New", 9, "bold"), tags="ov")
            if on:
                c.create_rectangle(bx + bw - CORNER, by + bh - CORNER, bx + bw, by + bh,
                                   outline=ACCENT, fill=ACCENT, tags="ov")
        # connectors
        for i, cn in enumerate(self.connectors):
            on = (self.sel == ("conn", i))
            base = cn.get("color", "#0e0e0e")
            col = base if base != "#0e0e0e" else "#cfcfcf"
            pts = [self.n2c(p["x"], p["y"]) for p in cn["pts"]]
            flat = [v for p in pts for v in p]
            c.create_line(*flat, fill=col, width=3 if on else 2,
                          dash=(5, 3) if cn.get("dashed") else None, tags="ov")
            for j, (px, py) in enumerate(pts):
                ends = (j == 0 or j == len(pts) - 1)
                r = HANDLE_R if ends else WP_R
                if ends:
                    c.create_oval(px - r, py - r, px + r, py + r, fill="#0e0e0e",
                                  outline=col, width=2 if on else 1, tags="ov")
                else:
                    c.create_polygon(px, py - r, px + r, py, px, py + r, px - r, py,
                                     fill="#1a1a1a", outline=col, width=2 if on else 1, tags="ov")
        # callouts
        for i, l in enumerate(self.labels):
            on = (self.sel == ("label", i))
            base = l.get("color", "#0e0e0e")
            col = base if base != "#0e0e0e" else "#cfcfcf"
            path = self.label_path_canvas(l)
            flat = [v for p in path for v in p]
            c.create_line(*flat, fill=col, width=3 if on else 2,
                          dash=(5, 3) if self.dash_var.get() else None, tags="ov")
            for w in l.get("waypoints", []):
                wx, wy = self.n2c(w["x"], w["y"])
                c.create_polygon(wx, wy - WP_R, wx + WP_R, wy, wx, wy + WP_R, wx - WP_R, wy,
                                 fill="#1a1a1a", outline=col, width=2 if on else 1, tags="ov")
            lx, ly = self.n2c(l["lx"], l["ly"])
            c.create_rectangle(lx - 24, ly - 12, lx + 24, ly + 12,
                               fill="#3a3320" if on else "#222018", outline=col,
                               width=2 if on else 1, tags="ov")
            c.create_text(lx, ly, text=l["ref"], fill=ACCENT,
                          font=("Courier New", 9, "bold"), tags="ov")
            ax, ay = self.n2c(l["ax"], l["ay"])
            c.create_oval(ax - HANDLE_R, ay - HANDLE_R, ax + HANDLE_R, ay + HANDLE_R,
                          fill="#0e0e0e", outline=col, width=2 if on else 1, tags="ov")

    # ── output ────────────────────────────────────────────────────────────
    def schedule_render(self):
        if self._dirty_job:
            self.after_cancel(self._dirty_job)
        self._dirty_job = self.after(120, self.render_now)

    def render_now(self):
        self._dirty_job = None
        if self.source_img is None:
            return
        self._rendered = render_patent(self.source_img, self.labels, self.boxes,
                                       self.connectors, **self._opts())
        self.redraw_output()

    def redraw_output(self):
        self.out_canvas.delete("all")
        if self._rendered is None:
            return
        cw = self.out_canvas.winfo_width(); ch = self.out_canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        iw, ih = self._rendered.size
        scale = min(cw / iw, ch / ih, 1.0)
        dw, dh = max(1, int(iw * scale)), max(1, int(ih * scale))
        disp = self._rendered.resize((dw, dh), Image.LANCZOS)
        self._out_photo = ImageTk.PhotoImage(disp)
        self.out_canvas.create_image(cw // 2, ch // 2, image=self._out_photo)


if __name__ == "__main__":
    PatentApp().mainloop()