"""Thesis figures (standalone SVG, no plotting dependency) from analyze_10m.py outputs.

Colours: reference categorical palette, slots 1-4 in fixed order (static, slot,
proportional, tdensity2); every series also has its own marker shape and a
direct label, so identity never depends on colour alone.
"""
import json, math, sys
from pathlib import Path

D = Path(sys.argv[1])  # derived/ directory
F = D.parent / "figures"
F.mkdir(exist_ok=True)
a10 = json.loads((D / "analysis_10m.json").read_text())
a65 = json.loads((D / "analysis_65k.json").read_text())

INK, INK2, GRID, BG = "#1f1f1e", "#5f5e5a", "#e4e2dc", "#ffffff"
STYLE = {  # policy -> (label, colour, marker)
    "static_flat": ("Static LMI", "#2a78d6", "circle"),
    "slot": ("DLI slot (default)", "#eb6834", "square"),
    "proportional": ("DLI proportional", "#1baf7a", "diamond"),
    "tdensity2": ("DLI T=2 density", "#eda100", "triangle"),
}
FONT = "font-family='Inter, Helvetica, Arial, sans-serif'"


class Plot:
    def __init__(self, x0, y0, w, h, xlim, ylim, xlog=False):
        self.x0, self.y0, self.w, self.h, self.xlim, self.ylim, self.xlog = x0, y0, w, h, xlim, ylim, xlog

    def X(self, v):
        lo, hi = self.xlim
        if self.xlog:
            return self.x0 + self.w * (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        return self.x0 + self.w * (v - lo) / (hi - lo)

    def Y(self, v):
        lo, hi = self.ylim
        return self.y0 + self.h - self.h * (v - lo) / (hi - lo)


def axes(p, xticks, yticks, xlabel, ylabel, xfmt=str, yfmt=str):
    s = []
    for t in yticks:
        y = p.Y(t)
        s.append(f"<line x1='{p.x0}' x2='{p.x0 + p.w}' y1='{y:.1f}' y2='{y:.1f}' stroke='{GRID}' stroke-width='1'/>")
        s.append(f"<text x='{p.x0 - 8}' y='{y + 4:.1f}' text-anchor='end' font-size='12' fill='{INK2}' {FONT}>{yfmt(t)}</text>")
    for t in xticks:
        x = p.X(t)
        s.append(f"<line x1='{x:.1f}' x2='{x:.1f}' y1='{p.y0 + p.h}' y2='{p.y0 + p.h + 4}' stroke='{INK2}'/>")
        s.append(f"<text x='{x:.1f}' y='{p.y0 + p.h + 18}' text-anchor='middle' font-size='12' fill='{INK2}' {FONT}>{xfmt(t)}</text>")
    s.append(f"<line x1='{p.x0}' x2='{p.x0 + p.w}' y1='{p.y0 + p.h}' y2='{p.y0 + p.h}' stroke='{INK2}'/>")
    s.append(f"<text x='{p.x0 + p.w / 2}' y='{p.y0 + p.h + 38}' text-anchor='middle' font-size='13' fill='{INK}' {FONT}>{xlabel}</text>")
    s.append(f"<text transform='translate({p.x0 - 48},{p.y0 + p.h / 2}) rotate(-90)' text-anchor='middle' font-size='13' fill='{INK}' {FONT}>{ylabel}</text>")
    return s


def marker(kind, x, y, c, title=""):
    t = f"<title>{title}</title>" if title else ""
    ring = f"stroke='{BG}' stroke-width='2'"
    if kind == "circle":
        return f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4.5' fill='{c}' {ring}>{t}</circle>"
    if kind == "square":
        return f"<rect x='{x - 4.5:.1f}' y='{y - 4.5:.1f}' width='9' height='9' rx='1.5' fill='{c}' {ring}>{t}</rect>"
    if kind == "diamond":
        return f"<polygon points='{x:.1f},{y - 6:.1f} {x + 6:.1f},{y:.1f} {x:.1f},{y + 6:.1f} {x - 6:.1f},{y:.1f}' fill='{c}' {ring}>{t}</polygon>"
    return f"<polygon points='{x:.1f},{y - 6:.1f} {x + 6:.1f},{y + 4.5:.1f} {x - 6:.1f},{y + 4.5:.1f}' fill='{c}' {ring}>{t}</polygon>"


def svg(w, h, body, title, desc):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}' role='img'>"
            f"<title>{title}</title><desc>{desc}</desc><rect width='100%' height='100%' fill='{BG}'/>"
            + "".join(body) + "</svg>")


def kfmt(v):
    return f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}K"


def legend(x, y, pols, step=150):
    s = []
    for i, pol in enumerate(pols):
        lab, c, m = STYLE[pol]
        xx = x + i * step
        s.append(f"<line x1='{xx}' x2='{xx + 22}' y1='{y}' y2='{y}' stroke='{c}' stroke-width='2'/>")
        s.append(marker(m, xx + 11, y, c))
        s.append(f"<text x='{xx + 28}' y='{y + 4}' font-size='12' fill='{INK}' {FONT}>{lab}</text>")
    return s


def frontier(p, ev, pols, key="recall30", ci_key="recall30_ci", label_last=True):
    s = []
    for pol in pols:
        lab, c, m = STYLE[pol]
        pts = ev["curves"][pol]
        path = " ".join(f"{'M' if i == 0 else 'L'}{p.X(q['work_mean']):.1f},{p.Y(q[key]):.1f}" for i, q in enumerate(pts))
        s.append(f"<path d='{path}' fill='none' stroke='{c}' stroke-width='2'/>")
        for q in pts:
            x = p.X(q["work_mean"])
            lo, hi = q[ci_key]
            s.append(f"<line x1='{x:.1f}' x2='{x:.1f}' y1='{p.Y(lo):.1f}' y2='{p.Y(hi):.1f}' stroke='{c}' stroke-width='1.5'/>")
            s.append(marker(m, x, p.Y(q[key]), c,
                            f"{lab}, budget {q['budget']}: work {q['work_mean']:.0f}, {key} {q[key]:.3f} [{lo:.3f}, {hi:.3f}]"))
    return s


# ---------------- Figure 1: 10M frontier, Recall@30 and Recall@10
for key, name in (("recall30", "Recall@30"), ("recall10", "Recall@10")):
    ev = a10["eval"]
    W, H = 760, 500
    p = Plot(80, 60, 640, 360, (1.5e4, 1e6), (0.35, 1.0), xlog=True)
    body = [f"<text x='80' y='28' font-size='16' font-weight='600' fill='{INK}' {FONT}>10.12M vectors: mean test {name} versus scored work</text>"]
    body += axes(p, [2e4, 5e4, 1e5, 2e5, 5e5, 1e6], [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                 "Mean live vectors scored per query (log scale)", f"Mean {name}", kfmt, lambda t: f"{t:.1f}")
    if key == "recall30":
        y = p.Y(0.90)
        body.append(f"<line x1='{p.x0}' x2='{p.x0 + p.w}' y1='{y:.1f}' y2='{y:.1f}' stroke='{INK2}' stroke-dasharray='4 4'/>")
        body.append(f"<text x='{p.x0 + 6}' y='{y - 6:.1f}' font-size='11' fill='{INK2}' {FONT}>H-10M target 0.90</text>")
    body += frontier(p, ev, list(STYLE), key, key + "_ci")
    body += legend(80, 48, list(STYLE))
    body.append(f"<text x='80' y='{H - 12}' font-size='11' fill='{INK2}' {FONT}>n = 256 DLI-unseen test queries (rows 9000–9255); markers = requested budgets 10K–640K; whiskers = 95% bootstrap CI over queries. One topology, one seed.</text>")
    (F / f"fig1_{key}_vs_work_10m.svg").write_text(svg(W, H, body, f"10M {name} versus scored work",
                                                         f"Mean {name} against mean live scored vectors per query for four selection policies."))

# ---------------- Figure 2: realized work decomposition (learned vs Plain) by requested budget
ev = a10["eval"]
W, H = 900, 430
body = [f"<text x='70' y='28' font-size='16' font-weight='600' fill='{INK}' {FONT}>10.12M: realized work per query, learned buckets versus the Plain buffer</text>"]
budgets = [q["budget"] for q in ev["curves"]["slot"]]
pw, gap = 190, 22
LEARN, PLAIN = "#2a78d6", "#eb6834"
for k, pol in enumerate(STYLE):
    x0 = 70 + k * (pw + gap)
    p = Plot(x0, 70, pw, 280, (0, len(budgets)), (0, 800000))
    if k == 0:
        for t in [0, 2e5, 4e5, 6e5, 8e5]:
            body.append(f"<text x='{x0 - 6}' y='{p.Y(t) + 4:.1f}' text-anchor='end' font-size='11' fill='{INK2}' {FONT}>{kfmt(t) if t else '0'}</text>")
        body.append(f"<text transform='translate(22,210) rotate(-90)' text-anchor='middle' font-size='12' fill='{INK}' {FONT}>Mean vectors scored per query</text>")
    for t in [2e5, 4e5, 6e5, 8e5]:
        body.append(f"<line x1='{x0}' x2='{x0 + pw}' y1='{p.Y(t):.1f}' y2='{p.Y(t):.1f}' stroke='{GRID}'/>")
    body.append(f"<text x='{x0 + pw / 2}' y='62' text-anchor='middle' font-size='12' font-weight='600' fill='{INK}' {FONT}>{STYLE[pol][0]}</text>")
    bw = pw / len(budgets) - 4
    for i, q in enumerate(ev["curves"][pol]):
        x = x0 + i * (pw / len(budgets)) + 2
        yl = p.Y(q["learned_work"])
        body.append(f"<rect x='{x:.1f}' y='{yl:.1f}' width='{bw:.1f}' height='{p.Y(0) - yl:.1f}' fill='{LEARN}'><title>{STYLE[pol][0]} budget {q['budget']}: learned {q['learned_work']:.0f}</title></rect>")
        if q["plain_work"] > 0:
            yp = p.Y(q["learned_work"] + q["plain_work"])
            body.append(f"<rect x='{x:.1f}' y='{yp:.1f}' width='{bw:.1f}' height='{yl - yp - 2:.1f}' fill='{PLAIN}'><title>{STYLE[pol][0]} budget {q['budget']}: Plain {q['plain_work']:.0f} (selected in {100 * q['plain_selected_frac']:.0f}% of queries)</title></rect>")
        # requested budget tick
        yb = p.Y(q["budget"])
        body.append(f"<line x1='{x - 1:.1f}' x2='{x + bw + 1:.1f}' y1='{yb:.1f}' y2='{yb:.1f}' stroke='{INK}' stroke-width='1.5'/>")
        body.append(f"<text x='{x + bw / 2:.1f}' y='{p.Y(0) + 14:.1f}' text-anchor='middle' font-size='9' fill='{INK2}' {FONT}>{kfmt(q['budget'])}</text>")
    body.append(f"<line x1='{x0}' x2='{x0 + pw}' y1='{p.Y(0)}' y2='{p.Y(0)}' stroke='{INK2}'/>")
body.append(f"<text x='{70 + 2 * (pw + gap) - gap / 2}' y='382' text-anchor='middle' font-size='12' fill='{INK}' {FONT}>Requested candidate budget</text>")
body.append(f"<rect x='70' y='398' width='12' height='12' fill='{LEARN}'/><text x='88' y='409' font-size='12' fill='{INK}' {FONT}>Learned buckets</text>")
body.append(f"<rect x='200' y='398' width='12' height='12' fill='{PLAIN}'/><text x='218' y='409' font-size='12' fill='{INK}' {FONT}>Plain buffer (120,191 vectors, all-or-nothing)</text>")
body.append(f"<line x1='520' x2='540' y1='404' y2='404' stroke='{INK}' stroke-width='1.5'/><text x='546' y='409' font-size='12' fill='{INK}' {FONT}>Requested budget</text>")
body.append(f"<text x='70' y='426' font-size='11' fill='{INK2}' {FONT}>Means over n = 256 test queries. Budget is a minimum target; the last whole bucket or the whole Plain buffer may overshoot it.</text>")
(F / "fig2_work_decomposition_10m.svg").write_text(svg(W, H, body, "10M realized work decomposition", "Stacked learned and Plain work per policy and budget."))

# ---------------- Figure 3: matched-work differences static - DLI
W, H = 760, 470
p = Plot(80, 60, 640, 330, (1.5e4, 1e6), (-0.06, 0.26), xlog=True)
body = [f"<text x='80' y='28' font-size='16' font-weight='600' fill='{INK}' {FONT}>10.12M: Recall@30 advantage of static LMI at matched scored work</text>"]
body += axes(p, [2e4, 5e4, 1e5, 2e5, 5e5, 1e6], [-0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25],
             "Mean live vectors scored per query (log scale)", "Static − DLI mean Recall@30", kfmt, lambda t: f"{t:+.2f}")
body.append(f"<line x1='{p.x0}' x2='{p.x0 + p.w}' y1='{p.Y(0):.1f}' y2='{p.Y(0):.1f}' stroke='{INK}' stroke-width='1'/>")
for pol in ["slot", "proportional", "tdensity2"]:
    lab, c, m = STYLE[pol]
    L = ev["matched_work"][f"static_flat - {pol}"]
    up = " ".join(f"{p.X(x['work']):.1f},{p.Y(x['ci95'][1]):.1f}" for x in L)
    dn = " ".join(f"{p.X(x['work']):.1f},{p.Y(x['ci95'][0]):.1f}" for x in reversed(L))
    body.append(f"<polygon points='{up} {dn}' fill='{c}' fill-opacity='0.15' stroke='none'/>")
    body.append("<path d='" + " ".join(f"{'M' if i == 0 else 'L'}{p.X(x['work']):.1f},{p.Y(x['diff']):.1f}" for i, x in enumerate(L)) + f"' fill='none' stroke='{c}' stroke-width='2'/>")
    for x in L:
        body.append(marker(m, p.X(x["work"]), p.Y(x["diff"]), c, f"static − {pol} at {x['work']:.0f}: {x['diff']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]"))
body += legend(80, 48, ["slot", "proportional", "tdensity2"], step=170)
body.append(f"<text x='80' y='{H - 26}' font-size='11' fill='{INK2}' {FONT}>Differences of interpolated mean curves, evaluated only inside both policies' measured work range (no extrapolation);</text>")
body.append(f"<text x='80' y='{H - 12}' font-size='11' fill='{INK2}' {FONT}>bands = paired 95% percentile bootstrap over the 256 test queries (10,000 resamples). Positive = static better.</text>")
(F / "fig3_matched_work_difference_10m.svg").write_text(svg(W, H, body, "10M matched-work Recall@30 difference", "Static minus DLI Recall@30 at matched work with bootstrap bands."))

# ---------------- Figure 4: 65K vs 10M, recall vs selectivity
W, H = 900, 470
body = [f"<text x='70' y='28' font-size='16' font-weight='600' fill='{INK}' {FONT}>Scale comparison: Recall@30 versus fraction of the collection scored</text>"]
for k, (A, ttl, xl) in enumerate([(a65, "65,536 vectors (testB, n = 256)", (0.5, 50)), (a10, "10,120,191 vectors (test, n = 256)", (0.1, 10))]):
    p = Plot(80 + k * 420, 70, 360, 320, xl, (0.35, 1.0), xlog=True)
    body.append(f"<text x='{p.x0}' y='62' font-size='13' font-weight='600' fill='{INK}' {FONT}>{ttl}</text>")
    ticks = [0.5, 1, 2, 5, 10, 20, 50] if k == 0 else [0.1, 0.2, 0.5, 1, 2, 5, 10]
    body += axes(p, ticks, [0.4, 0.6, 0.8, 0.9, 1.0], "Mean scored vectors / collection size (%, log)", "Mean Recall@30" if k == 0 else "", lambda t: f"{t:g}", lambda t: f"{t:.1f}")
    y = p.Y(0.90)
    body.append(f"<line x1='{p.x0}' x2='{p.x0 + p.w}' y1='{y:.1f}' y2='{y:.1f}' stroke='{INK2}' stroke-dasharray='4 4'/>")
    for pol in STYLE:
        lab, c, m = STYLE[pol]
        pts = A["eval"]["curves"][pol]
        body.append("<path d='" + " ".join(f"{'M' if i == 0 else 'L'}{p.X(q['selectivity_pct']):.1f},{p.Y(q['recall30']):.1f}" for i, q in enumerate(pts)) + f"' fill='none' stroke='{c}' stroke-width='2'/>")
        for q in pts:
            body.append(marker(m, p.X(q["selectivity_pct"]), p.Y(q["recall30"]), c, f"{lab} budget {q['budget']}: {q['selectivity_pct']:.2f}% scored, R@30 {q['recall30']:.3f}"))
body += legend(70, 448, list(STYLE), step=180)
(F / "fig4_scale_comparison.svg").write_text(svg(W, H, body, "65K versus 10M selectivity frontier", "Recall@30 against percentage of the collection scored at two scales."))

# ---------------- Figure 5: miss location by level (slot, tdensity2)
W, H = 860, 450
body = [f"<text x='70' y='28' font-size='16' font-weight='600' fill='{INK}' {FONT}>10.12M: where the missed top-30 neighbours are, per query</text>"]
CATS = [("L5", "Level 5 (6.0M)", "#184f95"), ("L4", "Level 4 (3.0M)", "#2a78d6"), ("L2", "Level 2 (750K)", "#6da7ec"),
        ("L1", "Level 1 (250K)", "#b7d3f6"), ("plain", "Plain buffer (120K), unselected", "#eb6834"),
        ("lost", "Selected but not returned", "#9a9990")]
for k, pol in enumerate(["slot", "tdensity2", "proportional"]):
    x0 = 70 + k * 265
    p = Plot(x0, 70, 230, 290, (0, 7), (0, 20))
    if k == 0:
        for t in [0, 5, 10, 15, 20]:
            body.append(f"<text x='{x0 - 6}' y='{p.Y(t) + 4:.1f}' text-anchor='end' font-size='11' fill='{INK2}' {FONT}>{t}</text>")
        body.append(f"<text transform='translate(28,215) rotate(-90)' text-anchor='middle' font-size='12' fill='{INK}' {FONT}>Missed gold neighbours per query (of 30)</text>")
    for t in [5, 10, 15, 20]:
        body.append(f"<line x1='{x0}' x2='{x0 + 230}' y1='{p.Y(t):.1f}' y2='{p.Y(t):.1f}' stroke='{GRID}'/>")
    body.append(f"<text x='{x0 + 115}' y='62' text-anchor='middle' font-size='12' font-weight='600' fill='{INK}' {FONT}>{STYLE[pol][0]}</text>")
    for i, b in enumerate(a10["budgets"]):
        mp = a10["per_level"][f"{pol}@{b}"]["misses_per_query"]
        x = x0 + i * (230 / 7) + 3
        acc = 0.0
        for key, lab, c in CATS:
            v = mp.get(key, 0.0)
            if v <= 0:
                continue
            y1, y2 = p.Y(acc), p.Y(acc + v)
            body.append(f"<rect x='{x:.1f}' y='{y2:.1f}' width='{230 / 7 - 6:.1f}' height='{max(y1 - y2 - 1, 0.5):.1f}' fill='{c}'><title>{STYLE[pol][0]} budget {b}: {lab} {v:.2f}</title></rect>")
            acc += v
        body.append(f"<text x='{x + (230 / 7 - 6) / 2:.1f}' y='{p.Y(0) + 14:.1f}' text-anchor='middle' font-size='9' fill='{INK2}' {FONT}>{kfmt(b)}</text>")
    body.append(f"<line x1='{x0}' x2='{x0 + 230}' y1='{p.Y(0)}' y2='{p.Y(0)}' stroke='{INK2}'/>")
body.append(f"<text x='{70 + 1.5 * 265}' y='392' text-anchor='middle' font-size='12' fill='{INK}' {FONT}>Requested candidate budget</text>")
for i, (key, lab, c) in enumerate(CATS):
    xx = 70 + (i % 3) * 260
    yy = 410 + (i // 3) * 18
    body.append(f"<rect x='{xx}' y='{yy - 10}' width='12' height='12' fill='{c}'/><text x='{xx + 18}' y='{yy}' font-size='12' fill='{INK}' {FONT}>{lab}</text>")
(F / "fig5_miss_location_10m.svg").write_text(svg(W, H + 10, body, "10M miss location by level", "Stacked missed gold neighbours by DLI level and cause."))
print("figures in", F)
