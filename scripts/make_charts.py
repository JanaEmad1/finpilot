"""Draw the README charts from the generated reports (reports/*.md, reports/intent_metrics.json).
Needs matplotlib (pip install matplotlib); nothing else in the project depends on it.

    python scripts/make_charts.py
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPORTS, DOCS = ROOT / "reports", ROOT / "docs"
BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"
NAMES = {"tfidf-logreg": "TF-IDF + logistic regression (served)", "distilbert-finetuned": "DistilBERT, fine-tuned"}
COLORS = {"tfidf-logreg": BLUE, "distilbert-finetuned": ORANGE}

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": "#d6d5d0", "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#ecebe7",
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.titleweight": "bold",
    "axes.titlecolor": INK,
})
pct = plt.FuncFormatter(lambda v, _: f"{v:.0%}")

metrics = json.loads((REPORTS / "intent_metrics.json").read_text())
intent_md = (REPORTS / "intent_eval.md").read_text(encoding="utf-8")
ab_md = (REPORTS / "retrieval_ab_test.md").read_text(encoding="utf-8")


def sweep(model):
    """Coverage/accuracy rows of the threshold table for one model."""
    block = intent_md.split(f"Coverage vs. accuracy on test — {model}:")[1].split("\n\n")[1]
    rows = re.findall(r"\|\s*([\d.]+)\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)\s*\|", block)
    return [(float(t), float(c) / 100, float(a)) for t, c, a in rows]


# 1. Automation vs. accuracy: the hand-off threshold trade-off
fig, ax = plt.subplots(figsize=(8, 4.4))
for model in metrics:
    rows = sweep(model)
    ax.plot([r[1] for r in rows], [r[2] for r in rows], color=COLORS[model], lw=2, marker="o", ms=6,
            label=NAMES[model])
    m = metrics[model]
    ax.plot(m["coverage"], m["auto_accuracy"]["value"], marker="*", ms=16, color=COLORS[model],
            markeredgecolor="white", zorder=5)
m = metrics["tfidf-logreg"]
ax.annotate(f'Chosen operating point\n{m["coverage"]:.1%} answered by the bot\nat {m["auto_accuracy"]["value"]:.1%} accuracy',
            xy=(m["coverage"], m["auto_accuracy"]["value"]), xytext=(0.70, 0.925), color=INK, fontsize=9.5,
            arrowprops=dict(arrowstyle="-", color=GRAY))
ax.axhline(0.97, color=GRAY, ls="--", lw=1)
ax.text(1.0, 0.9715, "97% accuracy target", ha="right", va="bottom", color=MUTED, fontsize=9)
ax.set(xlabel="Share of messages the bot answers itself (rest go to a human)",
       ylabel="Accuracy on answered messages", title="More automation costs accuracy: pick the point on purpose")
ax.xaxis.set_major_formatter(pct)
ax.yaxis.set_major_formatter(pct)
ax.legend(frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(DOCS / "automation_tradeoff.png", dpi=150)

# 2. Model comparison with 95% bootstrap CIs
fig, ax = plt.subplots(figsize=(8, 2.8))
labels, vals, lows, highs = [], [], [], []
for metric, label in [("banking77_accuracy", "Banking77 accuracy"), ("auto_accuracy", "Accuracy on answered")]:
    for model in reversed(list(metrics)):
        d = metrics[model][metric]
        labels.append(f"{label}\n{NAMES[model].split(' (')[0]}")
        vals.append(d["value"]); lows.append(d["value"] - d["low"]); highs.append(d["high"] - d["value"])
colors = [COLORS[m] for m in reversed(list(metrics))] * 2
for i, (v, lo, hi, c) in enumerate(zip(vals, lows, highs, colors)):
    ax.errorbar(v, i, xerr=[[lo], [hi]], fmt="o", color=c, ms=8, capsize=4, lw=2)
    ax.text(v + hi + 0.002, i, f"{v:.1%}", va="center", color=INK, fontsize=9.5)
ax.set_yticks(range(len(labels)), labels, fontsize=9)
ax.xaxis.set_major_formatter(pct)
ax.set_xlim(0.88, 0.99)
ax.grid(axis="y", visible=False)
ax.set_title("The bigger model is not better (McNemar p = 0.011)")
fig.tight_layout()
fig.savefig(DOCS / "model_comparison.png", dpi=150)

# 3. Retrieval A/B test
ab = re.findall(r"\| ([AB]: [^|]+?) \| ([\d.]+) \[([\d.]+), ([\d.]+)\]", ab_md)
fig, ax = plt.subplots(figsize=(8, 2.2))
for i, (name, v, lo, hi) in enumerate(reversed(ab)):
    v, lo, hi = float(v), float(lo), float(hi)
    ax.barh(i, v, color=BLUE if name.startswith("B") else GRAY, height=0.55)
    ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], color=INK, capsize=3, lw=1)
    ax.text(hi + 0.01, i, f"{v:.1%}", va="center", color=INK)
ax.set_yticks(range(len(ab)), [a[0] for a in reversed(ab)])
ax.xaxis.set_major_formatter(pct)
ax.set_xlim(0, 1.05)
ax.grid(axis="y", visible=False)
ax.set(xlabel="Top-1 correct help article (3,080 paired questions)", title="Retrieval A/B test: ship B")
fig.tight_layout()
fig.savefig(DOCS / "retrieval_ab.png", dpi=150)
print("wrote", [p.name for p in DOCS.glob("*.png")])
