#!/usr/bin/env python3
"""
patent_app.py — Native desktop app that turns any image into a
2005-era USPTO patent drawing, with interactive, snappy leader lines.

Run:
    python patent_app.py

Requires:
    pip install Pillow numpy
    (tkinter ships with Python on Windows/macOS;
     on Linux:  sudo apt install python3-tk)

EDITING (top "Edit" canvas):
  • Drag ● anchor handles   — where a line points to.
  • Drag ■ label boxes      — the callout (can go into the margins).
  • Drag ◆ bend points      — kinks in the leader line.
  • Double-click a line     — add a bend point there.
  • Right-click a bend      — delete it.
  • "ADD BEND" button       — add a bend to the selected line.
  • SNAP 45° toggle         — segments snap to 0/45/90° while dragging,
                              giving the clean "diagonal-then-straight" look.
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
    "C:/Windows/Fonts/cour.ttf",
    "/Library/Fonts/Courier New.ttf",
    "/System/Library/Fonts/Courier New.ttf",
]
_BOLD = [
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "C:/Windows/Fonts/courbd.ttf",
    "/Library/Fonts/Courier New Bold.ttf",
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
    a = a * 0.45 + s * 0.55
    arr = a * 255.0

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

# muted, technical-looking leader-line colors (+ white for dark image areas)
PRESET_COLORS = ["#0e0e0e", "#1f3a93", "#9e1b1b", "#1f6b2e", "#5b4636", "#ffffff"]
PRESET_NAMES  = ["Ink", "Blue", "Red", "Green", "Sepia", "White"]


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
                       (round(x0 + ux * end), round(y0 + uy * end))],
                      fill=fill, width=1)
        pos, on = end, not on


def _arrowhead(draw, x, y, fromx, fromy, size=8, fill=INK):
    dx, dy = x - fromx, y - fromy
    L = math.hypot(dx, dy)
    if L == 0:
        return
    ux, uy = dx / L, dy / L
    px, py = -uy, ux
    draw.polygon([(x, y),
                  (x - ux * size + px * size * 0.45, y - uy * size + py * size * 0.45),
                  (x - ux * size - px * size * 0.45, y - uy * size - py * size * 0.45)],
                 fill=fill)


def _tsize(draw, text, fnt):
    bb = draw.textbbox((0, 0), text, font=fnt)
    return bb[2] - bb[0], bb[3] - bb[1]


def _nearest_edge(cx, cy, hw, hh, target):
    """Midpoint of the box edge nearest to target point."""
    edges = [(cx + hw, cy), (cx - hw, cy), (cx, cy + hh), (cx, cy - hh)]
    return min(edges, key=lambda p: math.hypot(p[0] - target[0], p[1] - target[1]))


def render_patent(img, labels, fig_num=1, title="", grayscale=False,
                  grade_strength=0.55, font_scale=1.0,
                  dashed=True, arrow=False, border_style="double"):
    graded = patent_grade(img, grayscale=grayscale, strength=grade_strength)
    sw, sh = graded.size
    cw, ch = sw + PAD * 2, sh + PAD * 2 + TITLE_H

    canvas = Image.new("RGB", (cw, ch), (250, 247, 234))
    canvas.paste(graded, (PAD, PAD))
    draw = ImageDraw.Draw(canvas)

    f_ref = get_font(17 * font_scale, True)
    f_lbl = get_font(14 * font_scale)
    f_fig = get_font(24, True)
    f_ttl = get_font(19, True)

    for lbl in labels:
        ax = PAD + lbl["ax"] * sw
        ay = PAD + lbl["ay"] * sh
        lx = PAD + lbl["lx"] * sw
        ly = PAD + lbl["ly"] * sh
        ref, text = lbl["ref"], lbl["text"]
        wps = [(PAD + w["x"] * sw, PAD + w["y"] * sh) for w in lbl.get("waypoints", [])]

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
            draw.line([(bx + cp + rw + gap, my), (bx + cp + rw + gap + dw, my)],
                      fill=INK, width=1)
            draw.text((bx + cp + rw + gap + dw + gap, by + cp + 2), text,
                      font=f_lbl, fill=INK)

        # build leader path: box edge -> waypoints -> anchor
        col = hex_to_rgb(lbl.get("color", "#0e0e0e"))
        first_target = wps[0] if wps else (ax, ay)
        edge = _nearest_edge(bcx, bcy, bw / 2, bh / 2, first_target)
        path = [edge] + wps + [(ax, ay)]
        for i in range(len(path) - 1):
            x0, y0 = path[i]
            x1, y1 = path[i + 1]
            if dashed:
                _dashed_seg(draw, x0, y0, x1, y1, fill=col)
            else:
                draw.line([(int(x0), int(y0)), (int(x1), int(y1))], fill=col, width=1)
        if arrow:
            px, py = path[-2]
            _arrowhead(draw, int(ax), int(ay), int(px), int(py), fill=col)
        draw.ellipse((ax - 4, ay - 4, ax + 4, ay + 4), fill=col)

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
HANDLE_R = 7
WP_R = 6


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
        self.geometry("1240x840")
        self.configure(bg=BG)
        self.minsize(1000, 660)

        self.source_img = None
        self.labels = []
        self.selected = None
        self.placing = False
        self.drag = None              # {"type":..,"index":i,"wp":j}
        self.current_color = PRESET_COLORS[0]
        self._swatches = []

        self.dx = self.dy = 0
        self.dw = self.dh = 1
        self._src_photo = None
        self._out_photo = None
        self._rendered = None
        self._dirty_job = None

        self._build_ui()

    # ── UI ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        side = tk.Frame(self, bg=PANEL, width=290)
        side.pack(side="left", fill="y"); side.pack_propagate(False)
        outer = tk.Frame(side, bg=PANEL); outer.pack(fill="both", expand=True)
        canv = tk.Canvas(outer, bg=PANEL, highlightthickness=0, width=274)
        sb = ttk.Scrollbar(outer, orient="vertical", command=canv.yview)
        s = tk.Frame(canv, bg=PANEL)
        s.bind("<Configure>", lambda e: canv.configure(scrollregion=canv.bbox("all")))
        canv.create_window((0, 0), window=s, anchor="nw", width=274)
        canv.configure(yscrollcommand=sb.set)
        canv.pack(side="left", fill="both", expand=True); sb.pack(side="right", fill="y")
        # mouse wheel scroll for the panel
        canv.bind_all("<MouseWheel>", lambda e: canv.yview_scroll(int(-e.delta/120), "units"))
        canv.bind_all("<Button-4>", lambda e: canv.yview_scroll(-1, "units"))
        canv.bind_all("<Button-5>", lambda e: canv.yview_scroll(1, "units"))

        tk.Label(s, text="PATENT DIAGRAM TOOL", bg=PANEL, fg=FG,
                 font=("Courier New", 12, "bold")).pack(anchor="w", padx=14, pady=(16, 0))
        tk.Label(s, text="USPTO-STYLE LABELING", bg=PANEL, fg="#555",
                 font=("Courier New", 7)).pack(anchor="w", padx=14)

        self._btn(s, "📂  OPEN IMAGE", self.open_image, BLUE, "#16243f"
                  ).pack(fill="x", padx=14, pady=(16, 4))

        self._head(s, "FIGURE")
        self._flabel(s, "NUMBER")
        self.fig_var = tk.StringVar(value="1")
        self.fig_var.trace_add("write", lambda *_: self.schedule_render())
        self._entry(s, self.fig_var)
        self._flabel(s, "TITLE")
        self.title_var = tk.StringVar(value="")
        self.title_var.trace_add("write", lambda *_: self.schedule_render())
        self._entry(s, self.title_var)

        self._head(s, "APPEARANCE")
        self._flabel(s, "GRADE STRENGTH")
        self.grade_var = tk.DoubleVar(value=0.55); self._scale(s, self.grade_var, 0, 1)
        self._flabel(s, "LABEL SIZE")
        self.fsize_var = tk.DoubleVar(value=1.0); self._scale(s, self.fsize_var, 0.6, 1.8)
        self.gray_var = tk.BooleanVar(value=False); self._check(s, "GREYSCALE", self.gray_var)
        self.dash_var = tk.BooleanVar(value=True); self._check(s, "DASHED LEADER LINES", self.dash_var)
        self.arrow_var = tk.BooleanVar(value=False); self._check(s, "ARROWHEADS", self.arrow_var)
        self.snap_var = tk.BooleanVar(value=True); self._check(s, "SNAP LINES TO 45°", self.snap_var)

        self._flabel(s, "BORDER STYLE")
        self.border_var = tk.StringVar(value="double")
        self.border_var.trace_add("write", lambda *_: self.schedule_render())
        bf = tk.Frame(s, bg=PANEL); bf.pack(fill="x", padx=14)
        for txt in ("double", "single", "none"):
            tk.Radiobutton(bf, text=txt, value=txt, variable=self.border_var,
                           bg=PANEL, fg=SUBTLE, selectcolor="#141414",
                           activebackground=PANEL, activeforeground=FG,
                           font=("Courier New", 8), highlightthickness=0).pack(side="left")

        self._head(s, "LABELS")
        self._btn(s, "+  ADD LABEL", self.start_placing, GREEN, "#16301a"
                  ).pack(fill="x", padx=14, pady=(0, 4))
        self.status = tk.Label(s, text="Double-click a line to add a bend.\n"
                                       "Drag ● ■ ◆ handles to shape it.",
                               bg=PANEL, fg=SUBTLE, font=("Courier New", 8),
                               wraplength=250, justify="left")
        self.status.pack(anchor="w", padx=14, pady=(2, 4))

        self.listbox = tk.Listbox(s, bg="#141414", fg=FG, font=MONO, height=6,
                                  relief="flat", highlightthickness=1,
                                  highlightbackground="#3a3a3a",
                                  selectbackground="#2a3a5a", activestyle="none")
        self.listbox.pack(fill="x", padx=14, pady=(0, 4))
        self.listbox.bind("<<ListboxSelect>>", self.on_list_select)
        self.listbox.bind("<Double-Button-1>", lambda e: self.edit_label())

        br = tk.Frame(s, bg=PANEL); br.pack(fill="x", padx=14)
        self._btn(br, "EDIT", self.edit_label, "#bbb", "#2a2a2a"
                  ).pack(side="left", fill="x", expand=True, padx=(0, 3))
        self._btn(br, "DELETE", self.delete_label, "#c77", "#3a1a1a"
                  ).pack(side="left", fill="x", expand=True, padx=(3, 0))
        self._btn(s, "+ ADD BEND TO SELECTED", self.add_bend, ACCENT, "#332b14"
                  ).pack(fill="x", padx=14, pady=(4, 0))

        self._flabel(s, "PATH COLOR (selected line)")
        sw_row = tk.Frame(s, bg=PANEL)
        sw_row.pack(fill="x", padx=14, pady=(0, 2))
        self._swatches = []
        for idx, hexc in enumerate(PRESET_COLORS):
            holder = tk.Frame(sw_row, bg=PANEL, highlightthickness=2,
                              highlightbackground=PANEL, cursor="hand2")
            holder.pack(side="left", padx=2)
            chip = tk.Frame(holder, bg=hexc, width=30, height=24,
                            highlightthickness=1, highlightbackground="#555",
                            cursor="hand2")
            chip.pack_propagate(False)
            chip.pack()
            for w in (holder, chip):
                w.bind("<Button-1>", lambda e, h=hexc: self.set_path_color(h))
            self._swatches.append(holder)
        self._refresh_swatches()

        self._head(s, "OUTPUT")
        self._btn(s, "↓  SAVE PNG", self.save_png, GREEN, "#16301a"
                  ).pack(fill="x", padx=14, pady=(0, 16))

        right = tk.Frame(self, bg=BG); right.pack(side="left", fill="both", expand=True)
        tk.Label(right, text="EDIT  —  drag ● anchor · ■ box · ◆ bend  ·  dbl-click line = add bend  ·  right-click bend = delete",
                 bg=BG, fg="#666", font=("Courier New", 8)).pack(anchor="w", padx=16, pady=(12, 4))
        self.src_canvas = tk.Canvas(right, bg="#111", highlightthickness=1,
                                    highlightbackground="#2a2a2a", height=360)
        self.src_canvas.pack(fill="x", padx=16)
        self.src_canvas.bind("<Button-1>", self.on_press)
        self.src_canvas.bind("<B1-Motion>", self.on_drag)
        self.src_canvas.bind("<ButtonRelease-1>", self.on_release)
        self.src_canvas.bind("<Double-Button-1>", self.on_double)
        self.src_canvas.bind("<Button-3>", self.on_right)
        self.src_canvas.bind("<Configure>", lambda e: self.redraw_source())

        tk.Label(right, text="OUTPUT PREVIEW", bg=BG, fg="#666",
                 font=("Courier New", 8)).pack(anchor="w", padx=16, pady=(14, 4))
        ow = tk.Frame(right, bg=BG); ow.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.out_canvas = tk.Canvas(ow, bg="#0d0d0d", highlightthickness=1,
                                    highlightbackground="#2a2a2a")
        self.out_canvas.pack(fill="both", expand=True)
        self.out_canvas.bind("<Configure>", lambda e: self.redraw_output())
        self._show_empty()

    def _btn(self, p, text, cmd, fg=FG, bg="#333"):
        return tk.Button(p, text=text, command=cmd, fg=fg, bg=bg,
                         activeforeground=fg, activebackground=bg,
                         font=("Courier New", 9, "bold"), relief="flat", bd=0,
                         padx=8, pady=7, cursor="hand2", highlightthickness=0)

    def _head(self, p, txt):
        tk.Label(p, text=txt, bg=PANEL, fg=SUBTLE,
                 font=("Courier New", 8, "bold")).pack(anchor="w", padx=14, pady=(16, 6))

    def _flabel(self, p, txt):
        tk.Label(p, text=txt, bg=PANEL, fg=SUBTLE,
                 font=("Courier New", 8)).pack(anchor="w", padx=14, pady=(8, 2))

    def _entry(self, p, var):
        tk.Entry(p, textvariable=var, bg="#141414", fg=FG, insertbackground=FG,
                 font=MONO, relief="flat", highlightthickness=1,
                 highlightbackground="#3a3a3a").pack(fill="x", padx=14, ipady=3)

    def _scale(self, p, var, lo, hi):
        tk.Scale(p, variable=var, from_=lo, to=hi, resolution=0.01, orient="horizontal",
                 bg=PANEL, fg=SUBTLE, troughcolor="#141414", highlightthickness=0,
                 font=("Courier New", 7), sliderrelief="flat", activebackground=ACCENT,
                 command=lambda *_: self.schedule_render()).pack(fill="x", padx=12)

    def _check(self, p, txt, var):
        var.trace_add("write", lambda *_: self.schedule_render())
        tk.Checkbutton(p, text=txt, variable=var, bg=PANEL, fg=SUBTLE,
                       selectcolor="#141414", activebackground=PANEL, activeforeground=FG,
                       font=("Courier New", 9), relief="flat",
                       highlightthickness=0).pack(anchor="w", padx=10, pady=(6, 0))

    def _show_empty(self):
        self.src_canvas.delete("all")
        self.src_canvas.create_text(400, 170, text="OPEN AN IMAGE TO BEGIN",
                                    fill="#444", font=("Courier New", 11))

    def _opts(self):
        return dict(fig_num=self.fig_var.get() or "1", title=self.title_var.get().strip(),
                    grayscale=self.gray_var.get(), grade_strength=float(self.grade_var.get()),
                    font_scale=float(self.fsize_var.get()), dashed=self.dash_var.get(),
                    arrow=self.arrow_var.get(), border_style=self.border_var.get())

    # ── files ─────────────────────────────────────────────────────────────
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
            messagebox.showerror("Error", f"Could not open image:\n{e}")
            return
        self.labels.clear(); self.selected = None
        self.refresh_list(); self.cancel_placing()
        self.redraw_source(); self.render_now()

    def save_png(self):
        if self.source_img is None:
            messagebox.showinfo("No image", "Open an image first."); return
        path = filedialog.asksaveasfilename(title="Save patent PNG", defaultextension=".png",
                                            initialfile="patent.png",
                                            filetypes=[("PNG image", "*.png")])
        if not path:
            return
        render_patent(self.source_img, self.labels, **self._opts()).save(path)
        messagebox.showinfo("Saved", f"Saved:\n{path}")

    # ── coord mapping ─────────────────────────────────────────────────────
    def c2n(self, x, y): return (x - self.dx) / self.dw, (y - self.dy) / self.dh
    def n2c(self, nx, ny): return self.dx + nx * self.dw, self.dy + ny * self.dh

    # ── path helper (canvas space) ────────────────────────────────────────
    def label_path_canvas(self, l):
        ax, ay = self.n2c(l["ax"], l["ay"])
        lx, ly = self.n2c(l["lx"], l["ly"])
        wps = [self.n2c(w["x"], w["y"]) for w in l.get("waypoints", [])]
        first = wps[0] if wps else (ax, ay)
        edge = _nearest_edge(lx, ly, 24, 12, first)   # 24x12 = half handle box
        return [edge] + wps + [(ax, ay)]

    # ── placement + drag ──────────────────────────────────────────────────
    def start_placing(self):
        if self.source_img is None:
            messagebox.showinfo("No image", "Open an image first."); return
        self.placing = True
        self.src_canvas.config(cursor="crosshair")
        self.status.config(text="↗ Click on the image to drop the anchor.", fg=ACCENT)

    def cancel_placing(self):
        self.placing = False
        self.src_canvas.config(cursor="")
        self.status.config(text="Double-click a line to add a bend.\n"
                                "Drag ● ■ ◆ handles to shape it.", fg=SUBTLE)

    def _next_ref(self):
        nums = [int(l["ref"]) for l in self.labels if l["ref"].isdigit()]
        return str(max(nums) + 2) if nums else "100"

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
            self.labels.append({"ref": ref.strip(), "text": text.strip(),
                                "ax": nx, "ay": ny, "lx": lx, "ly": ly,
                                "waypoints": [], "color": self.current_color})
            self.selected = len(self.labels) - 1
            self.cancel_placing(); self.refresh_list()
            self.redraw_source(); self.render_now()
            return
        hit = self._hit(event.x, event.y)
        self.drag = hit
        self.selected = hit["index"] if hit else None
        self.refresh_list_selection(); self._refresh_swatches(); self.update_overlays()

    def _hit(self, x, y):
        for i, l in enumerate(self.labels):
            ax, ay = self.n2c(l["ax"], l["ay"])
            if (x - ax) ** 2 + (y - ay) ** 2 <= (HANDLE_R + 4) ** 2:
                return {"type": "anchor", "index": i}
        for i, l in enumerate(self.labels):
            for j, w in enumerate(l.get("waypoints", [])):
                wx, wy = self.n2c(w["x"], w["y"])
                if (x - wx) ** 2 + (y - wy) ** 2 <= (WP_R + 4) ** 2:
                    return {"type": "waypoint", "index": i, "wp": j}
        for i, l in enumerate(self.labels):
            lx, ly = self.n2c(l["lx"], l["ly"])
            if abs(x - lx) <= 26 and abs(y - ly) <= 14:
                return {"type": "label", "index": i}
        return None

    def _ref_point(self, l, drag):
        """The fixed neighbour used for 45° snapping."""
        wps = l.get("waypoints", [])
        if drag["type"] == "anchor":
            return self.n2c(wps[-1]["x"], wps[-1]["y"]) if wps else self.n2c(l["lx"], l["ly"])
        if drag["type"] == "label":
            return self.n2c(wps[0]["x"], wps[0]["y"]) if wps else self.n2c(l["ax"], l["ay"])
        j = drag["wp"]
        if j > 0:
            return self.n2c(wps[j - 1]["x"], wps[j - 1]["y"])
        return self.n2c(l["lx"], l["ly"])

    def on_drag(self, event):
        if not self.drag:
            return
        i = self.drag["index"]; l = self.labels[i]
        x, y = event.x, event.y
        if self.snap_var.get():
            px, py = self._ref_point(l, self.drag)
            x, y = snap45(px, py, x, y)
        nx, ny = self.c2n(x, y)
        if self.drag["type"] == "anchor":
            l["ax"] = max(0.0, min(1.0, nx)); l["ay"] = max(0.0, min(1.0, ny))
        elif self.drag["type"] == "label":
            l["lx"] = max(-0.25, min(1.25, nx)); l["ly"] = max(-0.1, min(1.1, ny))
        else:
            w = l["waypoints"][self.drag["wp"]]
            w["x"] = max(-0.25, min(1.25, nx)); w["y"] = max(-0.1, min(1.1, ny))
        self.update_overlays(); self.schedule_render()

    def on_release(self, event):
        if self.drag:
            self.drag = None; self.render_now()

    def on_double(self, event):
        """Add a bend point on the nearest leader segment."""
        if self.source_img is None or self.placing:
            return
        best = None; bestd = 14 ** 2
        for i, l in enumerate(self.labels):
            path = self.label_path_canvas(l)
            for k in range(len(path) - 1):
                d2 = self._seg_dist2(event.x, event.y, path[k], path[k + 1])
                if d2 < bestd:
                    bestd = d2; best = (i, k)
        if best is None:
            return
        i, k = best
        nx, ny = self.c2n(event.x, event.y)
        self.labels[i].setdefault("waypoints", []).insert(k, {"x": nx, "y": ny})
        self.selected = i
        self.refresh_list_selection(); self.update_overlays(); self.render_now()

    def on_right(self, event):
        """Delete a bend point."""
        for i, l in enumerate(self.labels):
            for j, w in enumerate(l.get("waypoints", [])):
                wx, wy = self.n2c(w["x"], w["y"])
                if (event.x - wx) ** 2 + (event.y - wy) ** 2 <= (WP_R + 5) ** 2:
                    l["waypoints"].pop(j)
                    self.update_overlays(); self.render_now()
                    return

    @staticmethod
    def _seg_dist2(px, py, a, b):
        ax, ay = a; bx, by = b
        dx, dy = bx - ax, by - ay
        if dx == 0 and dy == 0:
            return (px - ax) ** 2 + (py - ay) ** 2
        t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
        cx, cy = ax + t * dx, ay + t * dy
        return (px - cx) ** 2 + (py - cy) ** 2

    def add_bend(self):
        if self.selected is None:
            messagebox.showinfo("No label", "Select a label first."); return
        l = self.labels[self.selected]
        path = self.label_path_canvas(l)
        # longest segment midpoint
        k = max(range(len(path) - 1),
                key=lambda i: (path[i + 1][0] - path[i][0]) ** 2 + (path[i + 1][1] - path[i][1]) ** 2)
        mx = (path[k][0] + path[k + 1][0]) / 2
        my = (path[k][1] + path[k + 1][1]) / 2
        nx, ny = self.c2n(mx, my)
        l.setdefault("waypoints", []).insert(k, {"x": nx, "y": ny})
        self.update_overlays(); self.render_now()

    # ── color ─────────────────────────────────────────────────────────────
    def set_path_color(self, hexc):
        self.current_color = hexc
        if self.selected is not None and 0 <= self.selected < len(self.labels):
            self.labels[self.selected]["color"] = hexc
            self.update_overlays()
            self.render_now()
        self._refresh_swatches()

    def _refresh_swatches(self):
        # the "active" color = selected label's color, else current default
        if self.selected is not None and 0 <= self.selected < len(self.labels):
            active = self.labels[self.selected].get("color", self.current_color)
        else:
            active = self.current_color
        for b, hexc in zip(self._swatches, PRESET_COLORS):
            b.config(highlightbackground=(ACCENT if hexc == active else PANEL),
                     highlightcolor=(ACCENT if hexc == active else PANEL))

    # ── list ──────────────────────────────────────────────────────────────
    def on_list_select(self, event):
        sel = self.listbox.curselection()
        self.selected = sel[0] if sel else None
        self._refresh_swatches()
        self.update_overlays()

    def refresh_list(self):
        self.listbox.delete(0, "end")
        for l in self.labels:
            nb = len(l.get("waypoints", []))
            tag = f" ({nb} bend{'s' if nb != 1 else ''})" if nb else ""
            self.listbox.insert("end", f" {l['ref']:<5} {l['text']}{tag}")
        self.refresh_list_selection()

    def refresh_list_selection(self):
        self.listbox.selection_clear(0, "end")
        if self.selected is not None and 0 <= self.selected < len(self.labels):
            self.listbox.selection_set(self.selected)

    def edit_label(self):
        if self.selected is None:
            return
        l = self.labels[self.selected]
        ref = simpledialog.askstring("Edit reference", "Reference numeral:",
                                     parent=self, initialvalue=l["ref"])
        if ref is None:
            return
        text = simpledialog.askstring("Edit description", "Label description:",
                                      parent=self, initialvalue=l["text"])
        if text is None:
            text = l["text"]
        l["ref"], l["text"] = ref.strip(), text.strip()
        self.refresh_list(); self.redraw_source(); self.render_now()

    def delete_label(self):
        if self.selected is None:
            return
        self.labels.pop(self.selected); self.selected = None
        self.refresh_list(); self.redraw_source(); self.render_now()

    # ── edit canvas ─────────────────────────────────────────────────────────
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
        self.src_canvas.create_image(self.dx, self.dy, anchor="nw",
                                     image=self._src_photo, tags="img")
        self.update_overlays()

    def update_overlays(self):
        c = self.src_canvas; c.delete("ov")
        for i, l in enumerate(self.labels):
            on = (i == self.selected)
            base = l.get("color", "#0e0e0e")
            # use the label's own color; if it's near-black, lift to grey so it's visible on dark canvas
            col = base if base != "#0e0e0e" else "#cfcfcf"
            path = self.label_path_canvas(l)
            flat = [v for p in path for v in p]
            c.create_line(*flat, fill=col, width=3 if on else 2,
                          dash=(5, 3) if self.dash_var.get() else None, tags="ov")
            # waypoints
            for w in l.get("waypoints", []):
                wx, wy = self.n2c(w["x"], w["y"])
                c.create_polygon(wx, wy - WP_R, wx + WP_R, wy, wx, wy + WP_R, wx - WP_R, wy,
                                 fill="#1a1a1a", outline=col, width=2 if on else 1, tags="ov")
            # label box
            lx, ly = self.n2c(l["lx"], l["ly"])
            c.create_rectangle(lx - 24, ly - 12, lx + 24, ly + 12,
                               fill="#3a3320" if on else "#222018", outline=col,
                               width=2 if on else 1, tags="ov")
            c.create_text(lx, ly, text=l["ref"], fill=ACCENT,
                          font=("Courier New", 9, "bold"), tags="ov")
            # anchor
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
        self._rendered = render_patent(self.source_img, self.labels, **self._opts())
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