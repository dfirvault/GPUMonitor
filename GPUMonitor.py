"""
gpu_monitor.py — GPU System Monitor  (multi-layout edition)
Requires: pip install nvidia-ml-py psutil
"""

import tkinter as tk
from tkinter import ttk
try:
    import pynvml
except ImportError:
    raise SystemExit("Run:  pip install nvidia-ml-py")
try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False

import time, math

# ── Palette ────────────────────────────────────────────────────────────────────
BG           = "#0a0c10"
PANEL_BG     = "#0f1218"
BORDER       = "#1c2333"
BORDER2      = "#252d3d"
ACCENT_CYAN  = "#00e5ff"
ACCENT_GREEN = "#00ff9d"
ACCENT_WARN  = "#ffb300"
ACCENT_CRIT  = "#ff3d5a"
ACCENT_PURP  = "#bd93f9"
ACCENT_ORG   = "#ff9500"
ACCENT_TEAL  = "#00b4d8"
TEXT_DIM     = "#4a5568"
TEXT_MID     = "#8896a8"
WHITE        = "#e8edf2"

UPDATE_MS    = 1000
HISTORY_SEC  = 1800   # 30-min ring buffer

HISTORY_SPANS = [
    (60,   "1 min"),
    (120,  "2 min"),
    (300,  "5 min"),
    (600,  "10 min"),
    (1800, "30 min"),
]
DEFAULT_SPAN = 300

LAYOUTS = [
    (1, "Bars",     "Classic metric bars"),
    (2, "History",  "Bars + history charts"),
    (3, "Dials",    "Combined arc gauges"),
    (4, "Heatmap",  "Colour-coded cells"),
    (5, "Terminal", "Large numeric readout"),
]

# Minimum usable content width
MIN_CONTENT_W = 400

# ── Helpers ────────────────────────────────────────────────────────────────────
def bar_colour(pct, warn=70, crit=90):
    if pct >= crit:  return ACCENT_CRIT
    if pct >= warn:  return ACCENT_WARN
    return ACCENT_CYAN

def temp_colour(c, warn=75, crit=85):
    if c >= crit:  return ACCENT_CRIT
    if c >= warn:  return ACCENT_WARN
    return ACCENT_GREEN

def lerp_hex(c1, c2, t):
    r1,g1,b1 = int(c1[1:3],16),int(c1[3:5],16),int(c1[5:7],16)
    r2,g2,b2 = int(c2[1:3],16),int(c2[3:5],16),int(c2[5:7],16)
    return "#{:02x}{:02x}{:02x}".format(
        int(r1+(r2-r1)*t), int(g1+(g2-g1)*t), int(b1+(b2-b1)*t))

def heatmap_colour(pct):
    if pct < 50:   return lerp_hex("#0f3a2a","#1a5c3a", pct/50)
    elif pct < 75: return lerp_hex("#3a2e00","#7a5800", (pct-50)/25)
    else:          return lerp_hex("#4a0f14","#8a1a22", (pct-75)/25)


# ══════════════════════════════════════════════════════════════════════════════
#  Scrollable content container
# ══════════════════════════════════════════════════════════════════════════════

class ScrollableFrame(tk.Frame):
    """A frame whose contents scroll vertically; fills all available height."""
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, **kw)

        self._canvas = tk.Canvas(self, bg=BG, highlightthickness=0,
                                  bd=0, yscrollincrement=1)
        self._scrollbar = tk.Scrollbar(self, orient="vertical",
                                        command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._scrollbar_set)

        self._scrollbar.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        self.inner = tk.Frame(self._canvas, bg=BG)
        self._win_id = self._canvas.create_window((0, 0), window=self.inner,
                                                   anchor="nw")

        self.inner.bind("<Configure>", self._on_inner_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)

        # Mouse-wheel scrolling
        self._canvas.bind("<Enter>", self._bind_wheel)
        self._canvas.bind("<Leave>", self._unbind_wheel)

        self._sb_needed = False

    def _scrollbar_set(self, lo, hi):
        needed = not (float(lo) == 0.0 and float(hi) == 1.0)
        if needed != self._sb_needed:
            self._sb_needed = needed
            if needed:
                self._scrollbar.pack(side="right", fill="y")
            else:
                self._scrollbar.pack_forget()
        self._scrollbar.set(lo, hi)

    def _on_inner_configure(self, event):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self._canvas.itemconfig(self._win_id, width=event.width)

    def _bind_wheel(self, event):
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_wheel(self, event):
        self._canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        self._canvas.yview_scroll(int(-1*(event.delta/120)), "units")


# ══════════════════════════════════════════════════════════════════════════════
#  Data stores
# ══════════════════════════════════════════════════════════════════════════════

class GPUData:
    def __init__(self, index, name, vram_total_mb):
        self.index      = index
        self.name       = name
        self.vram_total = vram_total_mb
        self.load = self.vram = self.temp = 0
        self.vram_pct = 0.0
        self.ok   = True
        self.h_load = []; self.h_vram_pct = []; self.h_temp = []

    def ingest(self, handle):
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem  = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            self.load     = util.gpu
            self.vram     = mem.used // (1024*1024)
            self.vram_pct = mem.used / mem.total * 100
            self.temp     = temp
            self.ok       = True
            self.h_load.append(float(self.load))
            self.h_vram_pct.append(self.vram_pct)
            self.h_temp.append(float(self.temp))
            for lst in (self.h_load, self.h_vram_pct, self.h_temp):
                if len(lst) > HISTORY_SEC: lst.pop(0)
        except pynvml.NVMLError:
            self.ok = False


class SystemData:
    """CPU + RAM sampled via psutil."""
    def __init__(self):
        self.cpu_pct  = 0.0
        self.cpu_temp = None
        self.ram_pct  = 0.0
        self.ram_used_mb = 0
        self.ram_total_mb = 0
        self.h_cpu  = []
        self.h_ram  = []
        self.h_ctemp = []
        if PSUTIL_OK:
            vm = psutil.virtual_memory()
            self.ram_total_mb = vm.total // (1024*1024)
            psutil.cpu_percent(interval=None)

    def ingest(self):
        if not PSUTIL_OK:
            return
        self.cpu_pct = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        self.ram_pct  = vm.percent
        self.ram_used_mb = vm.used // (1024*1024)
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for key in ("coretemp","cpu_thermal","k10temp","Package id 0","CPU"):
                    if key in temps:
                        self.cpu_temp = temps[key][0].current
                        break
                else:
                    first = next(iter(temps.values()))
                    self.cpu_temp = first[0].current
        except (AttributeError, Exception):
            self.cpu_temp = None

        self.h_cpu.append(self.cpu_pct)
        self.h_ram.append(self.ram_pct)
        if self.cpu_temp is not None:
            self.h_ctemp.append(float(self.cpu_temp))
        for lst in (self.h_cpu, self.h_ram, self.h_ctemp):
            if len(lst) > HISTORY_SEC: lst.pop(0)


class DiskData:
    """Per-physical-disk I/O metrics via psutil."""
    def __init__(self):
        self.disks   = {}
        self.h_read  = {}
        self.h_write = {}
        self.h_active= {}
        self._prev   = {}
        if PSUTIL_OK:
            self._init_disks()

    def _init_disks(self):
        try:
            counters = psutil.disk_io_counters(perdisk=True)
            if not counters:
                return
            now = time.monotonic()
            for name, c in counters.items():
                busy = getattr(c, 'busy_time', 0)
                self._prev[name] = (c.read_bytes, c.write_bytes, busy, now)
                self.disks[name]  = {"read_mbs": 0.0, "write_mbs": 0.0, "active_pct": 0.0}
                self.h_read[name]  = []
                self.h_write[name] = []
                self.h_active[name]= []
        except Exception:
            pass

    def ingest(self):
        if not PSUTIL_OK or not self._prev:
            return
        try:
            counters = psutil.disk_io_counters(perdisk=True)
            if not counters:
                return
            now = time.monotonic()
            for name, c in counters.items():
                if name not in self._prev:
                    busy = getattr(c, 'busy_time', 0)
                    self._prev[name] = (c.read_bytes, c.write_bytes, busy, now)
                    self.disks[name]  = {"read_mbs": 0.0, "write_mbs": 0.0, "active_pct": 0.0}
                    self.h_read[name]  = []
                    self.h_write[name] = []
                    self.h_active[name]= []
                    continue
                pr, pw, pb, pt = self._prev[name]
                dt = now - pt
                if dt <= 0:
                    continue
                read_mbs  = (c.read_bytes  - pr) / dt / (1024*1024)
                write_mbs = (c.write_bytes - pw) / dt / (1024*1024)
                busy = getattr(c, 'busy_time', 0)
                busy_delta = max(busy - pb, 0)
                active_pct = min(busy_delta / (dt * 1000) * 100, 100)

                self.disks[name] = {
                    "read_mbs":   max(read_mbs, 0.0),
                    "write_mbs":  max(write_mbs, 0.0),
                    "active_pct": active_pct,
                }
                self.h_read[name].append(max(read_mbs, 0.0))
                self.h_write[name].append(max(write_mbs, 0.0))
                self.h_active[name].append(active_pct)
                for lst in (self.h_read[name], self.h_write[name], self.h_active[name]):
                    if len(lst) > HISTORY_SEC: lst.pop(0)
                self._prev[name] = (c.read_bytes, c.write_bytes, busy, now)
        except Exception:
            pass

    @property
    def disk_names(self):
        return list(self.disks.keys())


# ── Resizable BarWidget ────────────────────────────────────────────────────────
class BarWidget:
    H = 14; R = 7
    def __init__(self, parent, bg=PANEL_BG):
        self.w = 100   # will be updated on first resize
        self.c = tk.Canvas(parent, height=self.H,
                           bg=bg, highlightthickness=0)
        self.c.pack(anchor="w", pady=(2,0), fill="x")
        self.c.bind("<Configure>", self._on_resize)
        self._colour = ACCENT_CYAN
        self._pct    = 0
        self._draw_track()

    def _on_resize(self, event):
        new_w = max(event.width, 20)
        if new_w != self.w:
            self.w = new_w
            self._draw_track()
            self._draw_fill()

    def _draw_track(self):
        self.c.delete("track")
        r, w, h = self.R, self.w, self.H
        self.c.create_arc(0, 0, r*2, h, start=90, extent=180,
                          fill=BORDER, outline="", tags="track")
        self.c.create_arc(w-r*2, 0, w, h, start=-90, extent=180,
                          fill=BORDER, outline="", tags="track")
        self.c.create_rectangle(r, 0, w-r, h, fill=BORDER,
                                outline="", tags="track")

    def _draw_fill(self):
        self.c.delete("fill")
        fw = max(int(self.w * self._pct / 100), 1)
        r, h, col = self.R, self.H, self._colour
        if fw >= r*2:
            self.c.create_arc(0, 0, r*2, h, start=90, extent=180,
                              fill=col, outline="", tags="fill")
            self.c.create_rectangle(r, 0, fw, h, fill=col,
                                    outline="", tags="fill")
        else:
            self.c.create_rectangle(0, 0, fw, h, fill=col,
                                    outline="", tags="fill")
        if fw >= self.w - r:
            self.c.create_arc(self.w-r*2, 0, self.w, h, start=-90, extent=180,
                              fill=col, outline="", tags="fill")

    def set(self, pct, colour):
        self._pct    = pct
        self._colour = colour
        self._draw_fill()


# ── Resizable SparkChart ───────────────────────────────────────────────────────
class SparkChart:
    """Line chart that resizes with its parent container."""
    def __init__(self, parent, label, colour, height=60, bg=PANEL_BG):
        self.colour = colour
        self.h      = height
        self._data  = []
        self._window_sec = DEFAULT_SPAN
        self._max_val    = 100
        frame = tk.Frame(parent, bg=bg)
        frame.pack(fill="x", pady=(0, 2))
        tk.Label(frame, text=label, bg=bg, fg=TEXT_DIM,
                 font=("Consolas", 7)).pack(anchor="w")
        self.canvas = tk.Canvas(frame, height=height,
                                bg="#090b0e", highlightthickness=1,
                                highlightbackground=BORDER)
        self.canvas.pack(fill="x")
        self.canvas.bind("<Configure>", self._on_resize)
        self._draw_grid()

    def _on_resize(self, event):
        self._draw_grid()
        self._redraw()

    def _draw_grid(self):
        self.canvas.delete("grid")
        w = self.canvas.winfo_width() or 100
        h = self.h
        for pct in [25, 50, 75]:
            y = h - int(h * pct / 100)
            self.canvas.create_line(0, y, w, y, fill=BORDER,
                                    dash=(2, 4), tags="grid")
        for pct in [0, 50, 100]:
            y = h - int(h * pct / 100) - 2
            self.canvas.create_text(4, max(y, 8), text=f"{pct}", fill=TEXT_DIM,
                                    font=("Consolas", 6), anchor="w", tags="grid")

    def draw(self, data, window_sec, max_val=100):
        self._data       = data
        self._window_sec = window_sec
        self._max_val    = max_val
        self._redraw()

    def _redraw(self):
        self.canvas.delete("spark")
        d = list(self._data[-self._window_sec:]) if self._data else []
        if len(d) < 2:
            return
        w = self.canvas.winfo_width() or 100
        h = self.h
        step = w / (len(d) - 1)
        max_val = self._max_val or 1

        def pt(i):
            x = i * step
            y = h - max(int(h * min(d[i], max_val) / max_val), 1)
            return x, y

        pts  = [pt(i) for i in range(len(d))]
        poly = [(pts[0][0], h)] + pts + [(pts[-1][0], h)]
        flat = [c for p in poly for c in p]
        self.canvas.create_polygon(flat,
                                   fill=lerp_hex(self.colour, "#0a0c10", 0.82),
                                   outline="", tags="spark")
        self.canvas.create_line([c for p in pts for c in p],
                                fill=self.colour, width=1.5,
                                smooth=True, tags="spark")
        lx, ly = pts[-1]
        self.canvas.create_oval(lx-3, ly-3, lx+3, ly+3,
                                fill=self.colour, outline=BG,
                                width=1, tags="spark")


class ArcGauge:
    def __init__(self, parent, label, unit, colour, size=160):
        self.size=size; self.colour=colour
        self.canvas = tk.Canvas(parent, width=size, height=size+20,
                                bg=PANEL_BG, highlightthickness=0)
        self.canvas.pack()
        m=14
        self.canvas.create_arc(m, m, size-m, size-m, start=220, extent=-260,
                               style="arc", outline=BORDER, width=10)
        self.val_var = tk.StringVar(value="—")
        self.val_lbl = tk.Label(parent, textvariable=self.val_var,
                                bg=PANEL_BG, fg=WHITE,
                                font=("Consolas", 16, "bold"))
        self.val_lbl.pack()
        tk.Label(parent, text=f"{label}  {unit}", bg=PANEL_BG,
                 fg=TEXT_DIM, font=("Consolas", 8)).pack()

    def set(self, pct, display_val):
        self.canvas.delete("arc_fill")
        s, m = self.size, 14
        extent = -260*(pct/100)
        col = ACCENT_CRIT if pct>=90 else ACCENT_WARN if pct>=70 else self.colour
        if extent != 0:
            self.canvas.create_arc(m, m, s-m, s-m, start=220, extent=extent,
                                   style="arc", outline=col, width=10,
                                   tags="arc_fill")
        ang = math.radians(220+extent)
        cx=cy=s/2; r=(s/2)-m
        ex=cx+r*math.cos(-ang); ey=cy+r*math.sin(-ang)
        self.canvas.create_oval(ex-6, ey-6, ex+6, ey+6,
                                fill=col, outline=BG, width=2,
                                tags="arc_fill")
        self.val_var.set(str(display_val))
        self.val_lbl.config(fg=col)


# ══════════════════════════════════════════════════════════════════════════════
#  Toggle helper — remembers its widget so pack/forget works properly
# ══════════════════════════════════════════════════════════════════════════════

class TogglePanel:
    """Wraps a panel widget and handles show/hide without destroy/recreate."""
    def __init__(self, panel_widget):
        self._panel = panel_widget
        self._visible = False

    def show(self):
        if not self._visible:
            self._panel.pack(fill="x", padx=8, pady=4)
            self._visible = True

    def hide(self):
        if self._visible:
            self._panel.pack_forget()
            self._visible = False

    @property
    def panel(self):
        return self._panel

    @property
    def visible(self):
        return self._visible


# ══════════════════════════════════════════════════════════════════════════════
#  System resource panel
# ══════════════════════════════════════════════════════════════════════════════

class SystemPanel(tk.Frame):
    """Collapsible CPU / RAM panel."""
    def __init__(self, parent, sys_data, window_sec_ref, show_spark=False):
        super().__init__(parent, bg=PANEL_BG, highlightthickness=1,
                         highlightbackground=BORDER2)
        self.sd  = sys_data
        self.win = window_sec_ref
        self.show_spark = show_spark

        hdr = tk.Frame(self, bg=PANEL_BG)
        hdr.pack(fill="x", padx=16, pady=(10, 8))
        tk.Label(hdr, text="SYSTEM", bg=PANEL_BG, fg=ACCENT_ORG,
                 font=("Consolas", 9, "bold")).pack(side="left")
        tk.Label(hdr, text="CPU · RAM", bg=PANEL_BG, fg=TEXT_DIM,
                 font=("Consolas", 8)).pack(side="left", padx=(8, 0))
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=12, pady=(0, 10))

        mf = tk.Frame(self, bg=PANEL_BG)
        mf.pack(fill="x", padx=16, pady=(0, 4))

        self.rows = {}
        specs = [
            ("cpu",   "CPU LOAD",  "%",  ACCENT_ORG,   True),
            ("ctemp", "CPU TEMP",  "°C", ACCENT_GREEN,  False),
            ("ram",   "RAM USED",  "%",  ACCENT_PURP,   True),
        ]
        for key, label, unit, col, has_bar in specs:
            r = tk.Frame(mf, bg=PANEL_BG)
            r.pack(fill="x", pady=(0, 6))
            tk.Label(r, text=label, bg=PANEL_BG, fg=TEXT_DIM,
                     font=("Consolas", 8), width=10, anchor="w").pack(side="left")
            vv = tk.StringVar(value="—")
            vl = tk.Label(r, textvariable=vv, bg=PANEL_BG, fg=col,
                          font=("Consolas", 12, "bold"), width=6, anchor="e")
            vl.pack(side="left", padx=(4, 2))
            tk.Label(r, text=unit, bg=PANEL_BG, fg=TEXT_DIM,
                     font=("Consolas", 8)).pack(side="left")
            bar = None
            if has_bar:
                bf = tk.Frame(mf, bg=PANEL_BG)
                bf.pack(fill="x", pady=(0, 2))
                bar = BarWidget(bf, bg=PANEL_BG)
            self.rows[key] = {"var": vv, "lbl": vl, "bar": bar, "col": col}

        self.sparks = {}
        if show_spark:
            tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=12, pady=(4, 8))
            sf = tk.Frame(self, bg=PANEL_BG)
            sf.pack(fill="x", padx=16, pady=(0, 10))
            self.sparks["cpu"]   = SparkChart(sf, "CPU LOAD %",  ACCENT_ORG,   height=45)
            self.sparks["ram"]   = SparkChart(sf, "RAM USED %",  ACCENT_PURP,  height=45)
            self.sparks["ctemp"] = SparkChart(sf, "CPU TEMP °C", ACCENT_GREEN, height=45)

    def refresh(self):
        sd = self.sd
        cl = bar_colour(sd.cpu_pct)
        self.rows["cpu"]["var"].set(f"{sd.cpu_pct:.0f}")
        self.rows["cpu"]["lbl"].config(fg=cl)
        if self.rows["cpu"]["bar"]: self.rows["cpu"]["bar"].set(sd.cpu_pct, cl)

        if sd.cpu_temp is not None:
            ct = temp_colour(sd.cpu_temp)
            self.rows["ctemp"]["var"].set(f"{sd.cpu_temp:.0f}")
            self.rows["ctemp"]["lbl"].config(fg=ct)
        else:
            self.rows["ctemp"]["var"].set("N/A")

        rv = bar_colour(sd.ram_pct)
        self.rows["ram"]["var"].set(f"{sd.ram_pct:.0f}")
        self.rows["ram"]["lbl"].config(fg=rv)
        if self.rows["ram"]["bar"]: self.rows["ram"]["bar"].set(sd.ram_pct, rv)

        if self.sparks:
            w = self.win()
            self.sparks["cpu"].draw(sd.h_cpu, w, 100)
            self.sparks["ram"].draw(sd.h_ram, w, 100)
            self.sparks["ctemp"].draw(sd.h_ctemp, w, 120)


# ══════════════════════════════════════════════════════════════════════════════
#  Disk I/O panel
# ══════════════════════════════════════════════════════════════════════════════

class DiskPanel(tk.Frame):
    _rate_max = 100.0

    def __init__(self, parent, disk_data, window_sec_ref, show_spark=False):
        super().__init__(parent, bg=PANEL_BG, highlightthickness=1,
                         highlightbackground=BORDER2)
        self.dd  = disk_data
        self.win = window_sec_ref
        self.show_spark = show_spark
        self._disk_widgets = {}
        self._disk_sparks  = {}

        hdr = tk.Frame(self, bg=PANEL_BG)
        hdr.pack(fill="x", padx=16, pady=(10, 8))
        tk.Label(hdr, text="DISK I/O", bg=PANEL_BG, fg=ACCENT_TEAL,
                 font=("Consolas", 9, "bold")).pack(side="left")
        tk.Label(hdr, text="transfer rate · active time", bg=PANEL_BG, fg=TEXT_DIM,
                 font=("Consolas", 8)).pack(side="left", padx=(8, 0))
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=12, pady=(0, 10))

        self.body = tk.Frame(self, bg=PANEL_BG)
        self.body.pack(fill="x", padx=16, pady=(0, 10))

        self._build_disk_rows()

    def _build_disk_rows(self):
        for w in self.body.winfo_children():
            w.destroy()
        self._disk_widgets.clear()
        self._disk_sparks.clear()

        names = self.dd.disk_names
        if not names:
            tk.Label(self.body, text="No disk I/O data available",
                     bg=PANEL_BG, fg=TEXT_DIM,
                     font=("Consolas", 8)).pack()
            return

        for disk in names:
            dh = tk.Frame(self.body, bg=PANEL_BG)
            dh.pack(fill="x", pady=(6, 2))
            tk.Label(dh, text=f"▸ {disk}", bg=PANEL_BG, fg=ACCENT_TEAL,
                     font=("Consolas", 8, "bold")).pack(side="left")

            mf = tk.Frame(self.body, bg=PANEL_BG)
            mf.pack(fill="x", pady=(0, 2))

            widgets = {}
            for key, label, unit, col in [
                ("read",   "READ",   "MB/s", ACCENT_CYAN),
                ("write",  "WRITE",  "MB/s", ACCENT_PURP),
                ("active", "ACTIVE", "%",    ACCENT_TEAL),
            ]:
                r = tk.Frame(mf, bg=PANEL_BG)
                r.pack(fill="x", pady=(0, 4))
                tk.Label(r, text=label, bg=PANEL_BG, fg=TEXT_DIM,
                         font=("Consolas", 8), width=8, anchor="w").pack(side="left")
                vv = tk.StringVar(value="—")
                vl = tk.Label(r, textvariable=vv, bg=PANEL_BG, fg=col,
                              font=("Consolas", 12, "bold"), width=8, anchor="e")
                vl.pack(side="left", padx=(4, 2))
                tk.Label(r, text=unit, bg=PANEL_BG, fg=TEXT_DIM,
                         font=("Consolas", 8)).pack(side="left")
                bf = tk.Frame(mf, bg=PANEL_BG)
                bf.pack(fill="x", pady=(0, 2))
                bar = BarWidget(bf, bg=PANEL_BG)
                widgets[key] = {"var": vv, "lbl": vl, "bar": bar, "col": col}

            self._disk_widgets[disk] = widgets

            if self.show_spark:
                tk.Frame(self.body, bg=BORDER, height=1).pack(fill="x", pady=(4, 4))
                sf = tk.Frame(self.body, bg=PANEL_BG)
                sf.pack(fill="x")
                self._disk_sparks[disk] = {
                    "read":   SparkChart(sf, f"{disk} READ MB/s",  ACCENT_CYAN,  height=40),
                    "write":  SparkChart(sf, f"{disk} WRITE MB/s", ACCENT_PURP,  height=40),
                    "active": SparkChart(sf, f"{disk} ACTIVE %",   ACCENT_TEAL,  height=40),
                }

            tk.Frame(self.body, bg=BORDER, height=1).pack(fill="x", pady=(4, 0))

    def refresh(self):
        if set(self.dd.disk_names) != set(self._disk_widgets.keys()):
            self._build_disk_rows()

        all_rates = []
        for d in self.dd.disk_names:
            dd = self.dd.disks.get(d, {})
            all_rates.append(dd.get("read_mbs", 0))
            all_rates.append(dd.get("write_mbs", 0))
        if all_rates:
            peak = max(all_rates)
            if peak > DiskPanel._rate_max:
                DiskPanel._rate_max = peak * 1.2
            else:
                DiskPanel._rate_max = max(DiskPanel._rate_max * 0.995,
                                          max(peak * 1.2, 10.0))
        rate_max = max(DiskPanel._rate_max, 10.0)

        for disk, widgets in self._disk_widgets.items():
            dd = self.dd.disks.get(disk, {})
            read_mbs  = dd.get("read_mbs",  0.0)
            write_mbs = dd.get("write_mbs", 0.0)
            active    = dd.get("active_pct", 0.0)

            read_pct  = min(read_mbs  / rate_max * 100, 100)
            write_pct = min(write_mbs / rate_max * 100, 100)

            rc = bar_colour(read_pct,  warn=60, crit=85)
            wc = bar_colour(write_pct, warn=60, crit=85)
            ac = bar_colour(active,    warn=70, crit=90)

            widgets["read"]["var"].set(f"{read_mbs:.1f}")
            widgets["read"]["lbl"].config(fg=rc)
            widgets["read"]["bar"].set(read_pct, rc)

            widgets["write"]["var"].set(f"{write_mbs:.1f}")
            widgets["write"]["lbl"].config(fg=wc)
            widgets["write"]["bar"].set(write_pct, wc)

            widgets["active"]["var"].set(f"{active:.0f}")
            widgets["active"]["lbl"].config(fg=ac)
            widgets["active"]["bar"].set(active, ac)

            if disk in self._disk_sparks:
                w = self.win()
                self._disk_sparks[disk]["read"].draw(
                    self.dd.h_read.get(disk, []), w, rate_max)
                self._disk_sparks[disk]["write"].draw(
                    self.dd.h_write.get(disk, []), w, rate_max)
                self._disk_sparks[disk]["active"].draw(
                    self.dd.h_active.get(disk, []), w, 100)


def make_toggle(parent, text_off, text_on, callback, bg=BG):
    state = {"on": False}
    lbl = tk.Label(parent, text=text_off, bg=BORDER, fg=TEXT_DIM,
                   font=("Consolas", 8, "bold"), padx=10, pady=4, cursor="hand2")
    def click(e):
        state["on"] = not state["on"]
        if state["on"]:
            lbl.config(bg=PANEL_BG, fg=ACCENT_ORG, text=text_on)
        else:
            lbl.config(bg=BORDER, fg=TEXT_DIM, text=text_off)
        callback(state["on"])
    lbl.bind("<Button-1>", click)
    return lbl


# ══════════════════════════════════════════════════════════════════════════════
#  Shared mixin for toggle panel logic
# ══════════════════════════════════════════════════════════════════════════════

class PanelToggleMixin:
    """Creates sys_panel and disk_panel once; shows/hides with pack/pack_forget."""
    def _init_toggleable_panels(self, sys_holder, disk_holder,
                                 window_sec_fn, show_spark=False):
        self._sys_holder  = sys_holder
        self._disk_holder = disk_holder

        self._sys_panel  = SystemPanel(sys_holder,  self.sd, window_sec_fn,
                                        show_spark=show_spark)
        self._disk_panel = DiskPanel(disk_holder, self.dd, window_sec_fn,
                                      show_spark=show_spark)

        # Initially hidden
        self._sys_visible  = False
        self._disk_visible = False

    def _toggle_sys(self, on):
        if on:
            self._sys_panel.pack(fill="x", padx=8, pady=4)
            self._sys_visible = True
        else:
            self._sys_panel.pack_forget()
            self._sys_visible = False
        self._sys_panel.refresh()

    def _toggle_disk(self, on):
        if on:
            self._disk_panel.pack(fill="x", padx=8, pady=4)
            self._disk_visible = True
        else:
            self._disk_panel.pack_forget()
            self._disk_visible = False
        if self._disk_visible:
            self._disk_panel.refresh()

    def _refresh_toggleable(self):
        if self._sys_visible:  self._sys_panel.refresh()
        if self._disk_visible: self._disk_panel.refresh()


# ══════════════════════════════════════════════════════════════════════════════
#  LAYOUT 1 — Classic bars
# ══════════════════════════════════════════════════════════════════════════════

class Layout1(PanelToggleMixin, tk.Frame):
    def __init__(self, parent, gpu_data_list, sys_data, disk_data):
        tk.Frame.__init__(self, parent, bg=BG)
        self.gdl = gpu_data_list
        self.sd  = sys_data
        self.dd  = disk_data
        self.panels = []

        tb = tk.Frame(self, bg=BG)
        tb.pack(fill="x", padx=20, pady=(8, 4))
        tog_disk = make_toggle(tb, "⊞  DISK I/O", "⊟  DISK I/O", self._toggle_disk)
        tog_disk.pack(side="right", padx=(4, 0))
        tog_sys  = make_toggle(tb, "⊞  SYSTEM RESOURCES", "⊟  SYSTEM RESOURCES",
                               self._toggle_sys)
        tog_sys.pack(side="right")

        row = tk.Frame(self, bg=BG)
        row.pack(padx=16, pady=(4, 8), fill="x")
        # Allow GPU columns to share width evenly
        for i, gd in enumerate(gpu_data_list):
            row.columnconfigure(i, weight=1)
        for gd in gpu_data_list:
            p = self._make_panel(row, gd)
            p.frame.grid(row=0, column=gd.index, padx=8, pady=4, sticky="nsew")
            self.panels.append(p)

        self._sys_holder  = tk.Frame(self, bg=BG)
        self._sys_holder.pack(fill="x", padx=16, pady=(0, 4))
        self._disk_holder = tk.Frame(self, bg=BG)
        self._disk_holder.pack(fill="x", padx=16, pady=(0, 8))

        self._init_toggleable_panels(self._sys_holder, self._disk_holder,
                                     lambda: DEFAULT_SPAN, show_spark=False)

    def _make_panel(self, parent, gd):
        f = tk.Frame(parent, bg=PANEL_BG, highlightthickness=1,
                     highlightbackground=BORDER)
        hdr = tk.Frame(f, bg=PANEL_BG)
        hdr.pack(fill="x", padx=20, pady=(16, 12))
        tk.Label(hdr, text=f"GPU {gd.index}", bg=PANEL_BG, fg=ACCENT_CYAN,
                 font=("Consolas", 9, "bold")).pack(side="left")
        tk.Label(hdr, text=gd.name, bg=PANEL_BG, fg=TEXT_MID,
                 font=("Consolas", 8)).pack(side="left", padx=(8, 0))
        dot = tk.Label(hdr, text="●", bg=PANEL_BG, fg=ACCENT_GREEN,
                       font=("Consolas", 10))
        dot.pack(side="right")
        tk.Frame(f, bg=BORDER, height=1).pack(fill="x", padx=16, pady=(0, 14))
        mf = tk.Frame(f, bg=PANEL_BG)
        mf.pack(fill="x", padx=20, pady=(0, 16))
        rows = {}
        for key, label, unit, has_bar in [
            ("load","LOAD","%",True), ("vram","VRAM","MiB",True), ("temp","TEMP","°C",False)
        ]:
            r = tk.Frame(mf, bg=PANEL_BG)
            r.pack(fill="x", pady=(0, 10))
            tk.Label(r, text=label, bg=PANEL_BG, fg=TEXT_DIM,
                     font=("Consolas", 8), width=5, anchor="w").pack(side="left")
            vv = tk.StringVar(value="—")
            vl = tk.Label(r, textvariable=vv, bg=PANEL_BG, fg=WHITE,
                          font=("Consolas", 12, "bold"), width=7, anchor="e")
            vl.pack(side="left", padx=(4, 2))
            tk.Label(r, text=unit, bg=PANEL_BG, fg=TEXT_DIM,
                     font=("Consolas", 8), anchor="w").pack(side="left")
            bar = None
            if has_bar:
                bf = tk.Frame(mf, bg=PANEL_BG)
                bf.pack(fill="x", pady=(0, 2))
                bar = BarWidget(bf)
            rows[key] = {"var": vv, "lbl": vl, "bar": bar}
        class P: pass
        p = P(); p.frame = f; p.dot = dot; p.rows = rows
        return p

    def refresh(self):
        for p, gd in zip(self.panels, self.gdl):
            p.dot.config(fg=ACCENT_GREEN if gd.ok else ACCENT_CRIT)
            cl = bar_colour(gd.load)
            p.rows["load"]["var"].set(str(gd.load))
            p.rows["load"]["lbl"].config(fg=cl)
            p.rows["load"]["bar"].set(gd.load, cl)
            cv = bar_colour(gd.vram_pct)
            p.rows["vram"]["var"].set(str(gd.vram))
            p.rows["vram"]["lbl"].config(fg=cv)
            p.rows["vram"]["bar"].set(gd.vram_pct, cv)
            ct = temp_colour(gd.temp)
            p.rows["temp"]["var"].set(str(gd.temp))
            p.rows["temp"]["lbl"].config(fg=ct)
        self._refresh_toggleable()


# ══════════════════════════════════════════════════════════════════════════════
#  LAYOUT 2 — Bars + history charts
# ══════════════════════════════════════════════════════════════════════════════

class Layout2(PanelToggleMixin, tk.Frame):
    def __init__(self, parent, gpu_data_list, sys_data, disk_data):
        tk.Frame.__init__(self, parent, bg=BG)
        self.gdl = gpu_data_list
        self.sd  = sys_data
        self.dd  = disk_data
        self.panels = []
        self.window_sec = DEFAULT_SPAN

        ctrl = tk.Frame(self, bg=BG)
        ctrl.pack(fill="x", padx=20, pady=(8, 4))

        tk.Label(ctrl, text="WINDOW:", bg=BG, fg=TEXT_DIM,
                 font=("Consolas", 8)).pack(side="left", padx=(4, 8))

        self.span_btns = {}
        for secs, label in HISTORY_SPANS:
            btn = tk.Label(ctrl, text=label, bg=BORDER, fg=TEXT_DIM,
                           font=("Consolas", 8, "bold"), padx=10, pady=4, cursor="hand2")
            btn.pack(side="left", padx=(0, 2))
            btn.bind("<Button-1>", lambda e, s=secs: self._set_span(s))
            self.span_btns[secs] = btn
        self._highlight_span(self.window_sec)

        self.buf_var = tk.StringVar(value="")
        tk.Label(ctrl, textvariable=self.buf_var, bg=BG, fg=TEXT_DIM,
                 font=("Consolas", 7)).pack(side="left", padx=(12, 0))

        tog_disk = make_toggle(ctrl, "⊞  DISK I/O", "⊟  DISK I/O", self._toggle_disk)
        tog_disk.pack(side="right", padx=(4, 0))
        tog_sys  = make_toggle(ctrl, "⊞  SYSTEM", "⊟  SYSTEM", self._toggle_sys)
        tog_sys.pack(side="right")

        row = tk.Frame(self, bg=BG)
        row.pack(padx=16, pady=(4, 4), fill="x")
        for i, gd in enumerate(gpu_data_list):
            row.columnconfigure(i, weight=1)
        for gd in gpu_data_list:
            p = self._make_panel(row, gd)
            p["frame"].grid(row=0, column=gd.index, padx=8, pady=4, sticky="nsew")
            self.panels.append(p)

        self._sys_holder  = tk.Frame(self, bg=BG)
        self._sys_holder.pack(fill="x", padx=16, pady=(0, 4))
        self._disk_holder = tk.Frame(self, bg=BG)
        self._disk_holder.pack(fill="x", padx=16, pady=(0, 8))

        self._init_toggleable_panels(self._sys_holder, self._disk_holder,
                                     lambda: self.window_sec, show_spark=True)

    def _set_span(self, secs):
        self.window_sec = secs
        self._highlight_span(secs)

    def _highlight_span(self, active):
        for secs, btn in self.span_btns.items():
            btn.config(bg=PANEL_BG if secs==active else BORDER,
                       fg=ACCENT_CYAN if secs==active else TEXT_DIM)

    def _make_panel(self, parent, gd):
        f = tk.Frame(parent, bg=PANEL_BG, highlightthickness=1,
                     highlightbackground=BORDER)
        hdr = tk.Frame(f, bg=PANEL_BG)
        hdr.pack(fill="x", padx=20, pady=(14, 10))
        tk.Label(hdr, text=f"GPU {gd.index}", bg=PANEL_BG, fg=ACCENT_CYAN,
                 font=("Consolas", 9, "bold")).pack(side="left")
        tk.Label(hdr, text=gd.name, bg=PANEL_BG, fg=TEXT_MID,
                 font=("Consolas", 8)).pack(side="left", padx=(8, 0))
        dot = tk.Label(hdr, text="●", bg=PANEL_BG, fg=ACCENT_GREEN,
                       font=("Consolas", 10))
        dot.pack(side="right")
        tk.Frame(f, bg=BORDER, height=1).pack(fill="x", padx=16, pady=(0, 10))

        sf = tk.Frame(f, bg=PANEL_BG)
        sf.pack(fill="x", padx=20, pady=(0, 8))
        sparks = {
            "load": SparkChart(sf, "LOAD %",  ACCENT_CYAN,  height=50),
            "vram": SparkChart(sf, "VRAM %",  ACCENT_PURP,  height=50),
            "temp": SparkChart(sf, "TEMP °C", ACCENT_GREEN, height=50),
        }
        tk.Frame(f, bg=BORDER, height=1).pack(fill="x", padx=16, pady=(4, 10))

        mf = tk.Frame(f, bg=PANEL_BG)
        mf.pack(fill="x", padx=20, pady=(0, 14))
        rows = {}
        for key, label, unit, has_bar in [
            ("load","LOAD","%",True), ("vram","VRAM","MiB",True), ("temp","TEMP","°C",False)
        ]:
            r = tk.Frame(mf, bg=PANEL_BG)
            r.pack(fill="x", pady=(0, 8))
            tk.Label(r, text=label, bg=PANEL_BG, fg=TEXT_DIM,
                     font=("Consolas", 8), width=5, anchor="w").pack(side="left")
            vv = tk.StringVar(value="—")
            vl = tk.Label(r, textvariable=vv, bg=PANEL_BG, fg=WHITE,
                          font=("Consolas", 12, "bold"), width=7, anchor="e")
            vl.pack(side="left", padx=(4, 2))
            tk.Label(r, text=unit, bg=PANEL_BG, fg=TEXT_DIM,
                     font=("Consolas", 8), anchor="w").pack(side="left")
            bar = None
            if has_bar:
                bf = tk.Frame(mf, bg=PANEL_BG)
                bf.pack(fill="x", pady=(0, 2))
                bar = BarWidget(bf)
            rows[key] = {"var": vv, "lbl": vl, "bar": bar}
        return {"frame": f, "dot": dot, "rows": rows, "sparks": sparks, "gd": gd}

    def refresh(self):
        w = self.window_sec
        if self.gdl:
            buf = len(self.gdl[0].h_load)
            self.buf_var.set(f"  buffer {buf}/{HISTORY_SEC}s")
        for p in self.panels:
            gd = p["gd"]
            p["dot"].config(fg=ACCENT_GREEN if gd.ok else ACCENT_CRIT)
            cl = bar_colour(gd.load)
            p["rows"]["load"]["var"].set(str(gd.load))
            p["rows"]["load"]["lbl"].config(fg=cl)
            p["rows"]["load"]["bar"].set(gd.load, cl)
            cv = bar_colour(gd.vram_pct)
            p["rows"]["vram"]["var"].set(str(gd.vram))
            p["rows"]["vram"]["lbl"].config(fg=cv)
            p["rows"]["vram"]["bar"].set(gd.vram_pct, cv)
            ct = temp_colour(gd.temp)
            p["rows"]["temp"]["var"].set(str(gd.temp))
            p["rows"]["temp"]["lbl"].config(fg=ct)
            p["sparks"]["load"].draw(gd.h_load,     w, 100)
            p["sparks"]["vram"].draw(gd.h_vram_pct, w, 100)
            p["sparks"]["temp"].draw(gd.h_temp,     w, 120)
        self._refresh_toggleable()


# ══════════════════════════════════════════════════════════════════════════════
#  LAYOUT 3 — Combined arc dials
# ══════════════════════════════════════════════════════════════════════════════

class Layout3(PanelToggleMixin, tk.Frame):
    def __init__(self, parent, gpu_data_list, sys_data, disk_data):
        tk.Frame.__init__(self, parent, bg=BG)
        self.gdl = gpu_data_list
        self.sd  = sys_data
        self.dd  = disk_data

        tb = tk.Frame(self, bg=BG)
        tb.pack(fill="x", padx=20, pady=(8, 4))
        gpu_count = len(gpu_data_list)
        tk.Label(tb, text=f"COMBINED  ·  {gpu_count} GPU(s) AVERAGED",
                 bg=BG, fg=TEXT_DIM, font=("Consolas", 8)).pack(side="left")
        tog_disk = make_toggle(tb, "⊞  DISK I/O", "⊟  DISK I/O", self._toggle_disk)
        tog_disk.pack(side="right", padx=(4, 0))
        tog = make_toggle(tb, "⊞  SYSTEM RESOURCES", "⊟  SYSTEM RESOURCES",
                          self._toggle_sys)
        tog.pack(side="right")

        gauge_row = tk.Frame(self, bg=BG)
        gauge_row.pack(padx=32, pady=8)
        configs = [
            ("LOAD", "%",  ACCENT_CYAN,  "load"),
            ("VRAM", "%",  ACCENT_PURP,  "vram"),
            ("TEMP", "°C", ACCENT_GREEN, "temp"),
        ]
        self.gauges = {}
        for ci, (label, unit, col, key) in enumerate(configs):
            box = tk.Frame(gauge_row, bg=PANEL_BG, highlightthickness=1,
                           highlightbackground=BORDER)
            box.grid(row=0, column=ci, padx=12, pady=8, ipadx=12, ipady=12)
            self.gauges[key] = ArcGauge(box, label, unit, col, size=180)

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=24, pady=(8, 10))
        status_row = tk.Frame(self, bg=BG)
        status_row.pack()
        self.gpu_labels = []
        for gd in gpu_data_list:
            f = tk.Frame(status_row, bg=PANEL_BG, highlightthickness=1,
                         highlightbackground=BORDER)
            f.pack(side="left", padx=10, pady=(0, 10), ipadx=16, ipady=8)
            tk.Label(f, text=f"GPU {gd.index}", bg=PANEL_BG, fg=ACCENT_CYAN,
                     font=("Consolas", 8, "bold")).pack()
            tk.Label(f, text=gd.name, bg=PANEL_BG, fg=TEXT_DIM,
                     font=("Consolas", 7)).pack()
            items = {}
            for key, col in [("LOAD",ACCENT_CYAN), ("VRAM",ACCENT_PURP), ("TEMP",ACCENT_GREEN)]:
                r = tk.Frame(f, bg=PANEL_BG); r.pack(fill="x", pady=1)
                tk.Label(r, text=key, bg=PANEL_BG, fg=TEXT_DIM,
                         font=("Consolas", 7), width=5, anchor="w").pack(side="left")
                vv = tk.StringVar(value="—")
                tk.Label(r, textvariable=vv, bg=PANEL_BG, fg=col,
                         font=("Consolas", 9, "bold"), width=6, anchor="e").pack(side="left")
                items[key] = vv
            dot = tk.Label(f, text="●", bg=PANEL_BG, fg=ACCENT_GREEN,
                           font=("Consolas", 9))
            dot.pack()
            self.gpu_labels.append({"items": items, "dot": dot, "gd": gd})

        self._sys_holder  = tk.Frame(self, bg=BG)
        self._sys_holder.pack(fill="x", padx=16, pady=(0, 4))
        self._disk_holder = tk.Frame(self, bg=BG)
        self._disk_holder.pack(fill="x", padx=16, pady=(0, 8))
        self._init_toggleable_panels(self._sys_holder, self._disk_holder,
                                     lambda: DEFAULT_SPAN, show_spark=False)

    def refresh(self):
        gdl = self.gdl
        avg_load = sum(g.load     for g in gdl) / len(gdl)
        avg_vram = sum(g.vram_pct for g in gdl) / len(gdl)
        avg_temp = sum(g.temp     for g in gdl) / len(gdl)
        self.gauges["load"].set(avg_load, f"{avg_load:.0f}")
        self.gauges["vram"].set(avg_vram, f"{avg_vram:.0f}")
        self.gauges["temp"].set(min(avg_temp/120*100, 100), f"{avg_temp:.0f}")
        for entry in self.gpu_labels:
            gd = entry["gd"]
            entry["dot"].config(fg=ACCENT_GREEN if gd.ok else ACCENT_CRIT)
            entry["items"]["LOAD"].set(f"{gd.load}%")
            entry["items"]["VRAM"].set(f"{gd.vram_pct:.0f}%")
            entry["items"]["TEMP"].set(f"{gd.temp}°C")
        self._refresh_toggleable()


# ══════════════════════════════════════════════════════════════════════════════
#  LAYOUT 4 — Heatmap
# ══════════════════════════════════════════════════════════════════════════════

class Layout4(PanelToggleMixin, tk.Frame):
    METRICS = [
        ("LOAD", "%",  "load",     lambda g: g.load,       lambda v: v),
        ("VRAM", "%",  "vram_pct", lambda g: g.vram_pct,   lambda v: v),
        ("TEMP", "°C", "temp",     lambda g: float(g.temp), lambda v: v/120*100),
    ]

    def __init__(self, parent, gpu_data_list, sys_data, disk_data):
        tk.Frame.__init__(self, parent, bg=BG)
        self.gdl = gpu_data_list
        self.sd  = sys_data
        self.dd  = disk_data

        tb = tk.Frame(self, bg=BG)
        tb.pack(fill="x", padx=20, pady=(8, 4))
        tk.Label(tb, text="HEATMAP  ·  COLOUR = SEVERITY",
                 bg=BG, fg=TEXT_DIM, font=("Consolas", 8)).pack(side="left")
        tog_disk = make_toggle(tb, "⊞  DISK I/O", "⊟  DISK I/O", self._toggle_disk)
        tog_disk.pack(side="right", padx=(4, 0))
        tog = make_toggle(tb, "⊞  SYSTEM RESOURCES", "⊟  SYSTEM RESOURCES",
                          self._toggle_sys)
        tog.pack(side="right")

        grid = tk.Frame(self, bg=BG)
        grid.pack(padx=24, pady=(0, 8))
        CELL_W, CELL_H = 220, 110
        tk.Label(grid, text="", bg=BG, width=8).grid(row=0, column=0)
        for gd in gpu_data_list:
            tk.Label(grid, text=f"GPU {gd.index}\n{gd.name}", bg=BG,
                     fg=ACCENT_CYAN, font=("Consolas", 8, "bold"),
                     width=CELL_W//8).grid(row=0, column=gd.index+1, padx=4)

        self.cells = {}
        for ri, (label, unit, key, get_val, to_pct) in enumerate(self.METRICS):
            tk.Label(grid, text=label, bg=BG, fg=TEXT_MID,
                     font=("Consolas", 10, "bold"), width=6,
                     anchor="e").grid(row=ri+1, column=0, padx=(8, 4), pady=4)
            for gd in gpu_data_list:
                cell_bg = tk.Frame(grid, width=CELL_W, height=CELL_H, bg=PANEL_BG,
                                   highlightthickness=1, highlightbackground=BORDER)
                cell_bg.grid(row=ri+1, column=gd.index+1, padx=4, pady=4)
                cell_bg.pack_propagate(False)
                inner = tk.Frame(cell_bg, bg=PANEL_BG)
                inner.pack(expand=True)
                vv = tk.StringVar(value="—")
                vl = tk.Label(inner, textvariable=vv, bg=PANEL_BG, fg=WHITE,
                              font=("Consolas", 28, "bold"))
                vl.pack()
                ul = tk.Label(inner, text=unit, bg=PANEL_BG, fg=TEXT_DIM,
                              font=("Consolas", 10))
                ul.pack()
                spark_c = tk.Canvas(cell_bg, width=CELL_W-8, height=16,
                                    bg=PANEL_BG, highlightthickness=0)
                spark_c.pack(side="bottom", pady=(0, 4))
                self.cells[(key, gd.index)] = {
                    "frame": cell_bg, "inner": inner, "var": vv,
                    "val_lbl": vl, "unit_lbl": ul, "spark_canvas": spark_c,
                    "get_val": get_val, "to_pct": to_pct,
                    "hist_key": key, "gd": gd,
                }

        self._sys_holder  = tk.Frame(self, bg=BG)
        self._sys_holder.pack(fill="x", padx=16, pady=(0, 4))
        self._disk_holder = tk.Frame(self, bg=BG)
        self._disk_holder.pack(fill="x", padx=16, pady=(0, 8))
        self._init_toggleable_panels(self._sys_holder, self._disk_holder,
                                     lambda: DEFAULT_SPAN, show_spark=False)

    def refresh(self):
        for (key, idx), cell in self.cells.items():
            gd  = cell["gd"]
            raw = cell["get_val"](gd)
            pct = min(max(cell["to_pct"](raw), 0), 100)
            bg  = heatmap_colour(pct)
            cell["frame"].config(highlightbackground=bg)
            cell["inner"].config(bg=bg)
            cell["val_lbl"].config(bg=bg, fg=WHITE)
            cell["unit_lbl"].config(bg=bg)
            cell["var"].set(f"{raw:.0f}")
            hk = cell["hist_key"]
            hist_map = {"load": gd.h_load, "vram_pct": gd.h_vram_pct, "temp": gd.h_temp}
            sd2 = hist_map.get(hk, [])[-60:]
            sc  = cell["spark_canvas"]
            sc.config(bg=bg); sc.delete("all")
            if len(sd2) >= 2:
                W, H = int(sc["width"]), int(sc["height"])
                step = W / max(len(sd2)-1, 1)
                pts  = []
                for i, v in enumerate(sd2):
                    norm = cell["to_pct"](v) if hk == "temp" else v
                    pts.extend([i*step, H - max(int(H*norm/100), 1)])
                sc.create_line(pts, fill="#2a2f3a", width=1, smooth=True)
        self._refresh_toggleable()


# ══════════════════════════════════════════════════════════════════════════════
#  LAYOUT 5 — Terminal
# ══════════════════════════════════════════════════════════════════════════════

class Layout5(PanelToggleMixin, tk.Frame):
    def __init__(self, parent, gpu_data_list, sys_data, disk_data):
        tk.Frame.__init__(self, parent, bg=BG)
        self.gdl = gpu_data_list
        self.sd  = sys_data
        self.dd  = disk_data
        self.panels = []

        tb = tk.Frame(self, bg=BG)
        tb.pack(fill="x", padx=20, pady=(8, 4))
        tk.Label(tb, text="TERMINAL  ·  RAW READOUT",
                 bg=BG, fg=TEXT_DIM, font=("Consolas", 8)).pack(side="left")
        tog_disk = make_toggle(tb, "⊞  DISK I/O", "⊟  DISK I/O", self._toggle_disk)
        tog_disk.pack(side="right", padx=(4, 0))
        tog = make_toggle(tb, "⊞  SYSTEM RESOURCES", "⊟  SYSTEM RESOURCES",
                          self._toggle_sys)
        tog.pack(side="right")

        for gd in gpu_data_list:
            p = self._make_panel(gd)
            p["frame"].pack(fill="x", padx=24, pady=6)
            self.panels.append(p)

        self._sys_holder  = tk.Frame(self, bg=BG)
        self._sys_holder.pack(fill="x", padx=16, pady=(0, 4))
        self._disk_holder = tk.Frame(self, bg=BG)
        self._disk_holder.pack(fill="x", padx=16, pady=(0, 8))
        self._init_toggleable_panels(self._sys_holder, self._disk_holder,
                                     lambda: DEFAULT_SPAN, show_spark=False)

    def _make_panel(self, gd):
        f = tk.Frame(self, bg=PANEL_BG, highlightthickness=1,
                     highlightbackground=BORDER)
        hdr = tk.Frame(f, bg=PANEL_BG)
        hdr.pack(fill="x", padx=20, pady=(12, 8))
        tk.Label(hdr, text=f"▌GPU {gd.index}", bg=PANEL_BG, fg=ACCENT_CYAN,
                 font=("Consolas", 10, "bold")).pack(side="left")
        tk.Label(hdr, text=gd.name, bg=PANEL_BG, fg=TEXT_DIM,
                 font=("Consolas", 8)).pack(side="left", padx=(8, 0))
        dot = tk.Label(hdr, text="●", bg=PANEL_BG, fg=ACCENT_GREEN,
                       font=("Consolas", 10))
        dot.pack(side="right")
        tk.Frame(f, bg=BORDER, height=1).pack(fill="x", padx=16, pady=(0, 8))
        metric_row = tk.Frame(f, bg=PANEL_BG)
        metric_row.pack(fill="x", padx=20, pady=(0, 14))
        entries = {}
        for ci, (label, unit, col, key) in enumerate([
            ("LOAD","%",ACCENT_CYAN,"load"),
            ("VRAM","MiB",ACCENT_PURP,"vram"),
            ("TEMP","°C",ACCENT_GREEN,"temp"),
        ]):
            box = tk.Frame(metric_row, bg=PANEL_BG)
            box.grid(row=0, column=ci, padx=20)
            tk.Label(box, text=label, bg=PANEL_BG, fg=TEXT_DIM,
                     font=("Consolas", 8)).pack(anchor="w")
            vv = tk.StringVar(value="—")
            vl = tk.Label(box, textvariable=vv, bg=PANEL_BG, fg=col,
                          font=("Consolas", 34, "bold"), anchor="w")
            vl.pack(anchor="w")
            tk.Label(box, text=unit, bg=PANEL_BG, fg=TEXT_DIM,
                     font=("Consolas", 9)).pack(anchor="w")
            tv = tk.StringVar(value="  —")
            tl = tk.Label(box, textvariable=tv, bg=PANEL_BG, fg=TEXT_DIM,
                          font=("Consolas", 9))
            tl.pack(anchor="w")
            entries[key] = {"var": vv, "lbl": vl, "trend": tv, "tlbl": tl, "prev": None}
        return {"frame": f, "dot": dot, "entries": entries, "gd": gd}

    def refresh(self):
        for p in self.panels:
            gd = p["gd"]
            p["dot"].config(fg=ACCENT_GREEN if gd.ok else ACCENT_CRIT)
            vals = {"load": gd.load, "vram": gd.vram, "temp": gd.temp}
            cols = {"load": bar_colour(gd.load), "vram": bar_colour(gd.vram_pct),
                    "temp": temp_colour(gd.temp)}
            for key, val in vals.items():
                e = p["entries"][key]
                e["var"].set(str(val)); e["lbl"].config(fg=cols[key])
                if e["prev"] is not None:
                    d = val - e["prev"]
                    if d > 0:   e["trend"].set(f"▲ +{d}"); e["tlbl"].config(fg=ACCENT_WARN)
                    elif d < 0: e["trend"].set(f"▼ {d}");  e["tlbl"].config(fg=ACCENT_GREEN)
                    else:       e["trend"].set("  ━");      e["tlbl"].config(fg=TEXT_DIM)
                e["prev"] = val
        self._refresh_toggleable()


# ══════════════════════════════════════════════════════════════════════════════
#  Main window
# ══════════════════════════════════════════════════════════════════════════════

class GPUMonitor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.configure(bg=BG)
        self.minsize(MIN_CONTENT_W, 300)
        self.resizable(True, True)

        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError as e:
            self._fatal(f"NVML init failed:\n{e}"); return

        count = pynvml.nvmlDeviceGetCount()
        if count == 0:
            self._fatal("No NVIDIA GPUs detected."); return

        self.handles  = []
        self.gpu_data = []
        for i in range(count):
            h    = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes): name = name.decode()
            mem  = pynvml.nvmlDeviceGetMemoryInfo(h)
            self.handles.append(h)
            self.gpu_data.append(GPUData(i, name, mem.total//(1024*1024)))

        self.title(f"GPU Monitor  ·  {count} GPU(s) detected")

        self.sys_data  = SystemData()
        self.disk_data = DiskData()
        if not PSUTIL_OK:
            print("WARNING: psutil not installed. System metrics unavailable.\n"
                  "Run:  pip install psutil")

        self.active_view = None
        self._build_chrome()
        self._switch_layout(1)
        self._poll()

    def _build_chrome(self):
        title_frame = tk.Frame(self, bg=BG)
        title_frame.pack(fill="x", padx=24, pady=(16, 4))
        tk.Label(title_frame, text="GPU MONITOR", bg=BG, fg=ACCENT_CYAN,
                 font=("Consolas", 14, "bold")).pack(side="left")
        self.clock_var = tk.StringVar()
        tk.Label(title_frame, textvariable=self.clock_var, bg=BG,
                 fg=TEXT_DIM, font=("Consolas", 9)).pack(side="right", pady=(4, 0))

        name_frame = tk.Frame(self, bg=BG)
        name_frame.pack(fill="x", padx=24, pady=(0, 4))
        for gd in self.gpu_data:
            tk.Label(name_frame,
                     text=f"  GPU{gd.index}: {gd.name}  ({gd.vram_total} MiB)",
                     bg=BG, fg=TEXT_DIM,
                     font=("Consolas", 7)).pack(side="left", padx=(0, 16))

        tab_frame = tk.Frame(self, bg=BG)
        tab_frame.pack(fill="x", padx=20, pady=(4, 0))
        self.tab_btns = {}
        for num, short, tip in LAYOUTS:
            btn = tk.Label(tab_frame, text=f"  {num}·{short}  ",
                           bg=BORDER, fg=TEXT_DIM,
                           font=("Consolas", 8, "bold"), cursor="hand2", pady=5)
            btn.pack(side="left", padx=(0, 2))
            btn.bind("<Button-1>", lambda e, n=num: self._switch_layout(n))
            self.tab_btns[num] = btn

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # Scrollable content area fills all remaining vertical space
        self._scroll_frame = ScrollableFrame(self)
        self._scroll_frame.pack(fill="both", expand=True)
        self.content = self._scroll_frame.inner

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")
        footer = tk.Frame(self, bg=BG)
        footer.pack(fill="x", padx=24, pady=(6, 12))
        gpu_names = "  ·  ".join(gd.name for gd in self.gpu_data)
        tk.Label(footer, text=gpu_names, bg=BG, fg=TEXT_DIM,
                 font=("Consolas", 7)).pack(side="left")
        self.layout_label = tk.StringVar()
        tk.Label(footer, textvariable=self.layout_label, bg=BG,
                 fg=TEXT_DIM, font=("Consolas", 8)).pack(side="right")

    def _switch_layout(self, n):
        for num, btn in self.tab_btns.items():
            btn.config(bg=PANEL_BG if num==n else BORDER,
                       fg=ACCENT_CYAN if num==n else TEXT_DIM)
        for w in self.content.winfo_children():
            w.destroy()
        self.active_view = None
        tip = next(t for (nn, _, t) in LAYOUTS if nn == n)
        self.layout_label.set(tip + "  ·  1 s refresh")

        kwargs = dict(gpu_data_list=self.gpu_data, sys_data=self.sys_data,
                      disk_data=self.disk_data)
        views = {1: Layout1, 2: Layout2, 3: Layout3, 4: Layout4, 5: Layout5}
        if n in views:
            self.active_view = views[n](self.content, **kwargs)
            self.active_view.pack(fill="both", expand=True)
            self.active_view.refresh()

    def _poll(self):
        self.clock_var.set(time.strftime("%H:%M:%S"))
        for gd, h in zip(self.gpu_data, self.handles):
            gd.ingest(h)
        self.sys_data.ingest()
        self.disk_data.ingest()
        if self.active_view:
            self.active_view.refresh()
        self.after(UPDATE_MS, self._poll)

    def _fatal(self, msg):
        self.title("GPU Monitor  ·  Error")
        tk.Label(self, text=msg, bg=BG, fg=ACCENT_CRIT,
                 font=("Consolas", 11), padx=30, pady=30).pack()

    def on_close(self):
        try: pynvml.nvmlShutdown()
        except Exception: pass
        self.destroy()


if __name__ == "__main__":
    app = GPUMonitor()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
