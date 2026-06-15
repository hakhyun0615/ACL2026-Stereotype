#!/usr/bin/env python3
"""Appendix figures (no GPU): binned Logit-Lens JSD layerwise (JA) and the
Korean layerwise counterparts (CKA, English-script mass, JSD). Reads local
results/<m>/metrics.json. Reuses scripts/figure_gen.py's exact styling
(rcParams, COLORS, LABELS, style_axis) so the appendix matches the main figures."""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_gen import COLORS, LABELS, style_axis  # noqa: E402  (also applies shared rcParams)

RESULTS = Path("results")
OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)


def field_curve(key, field):
    d = json.load(open(RESULTS / key / "metrics.json"))[field]
    xs = sorted(int(k) for k in d)
    return xs, [d[str(x)] for x in xs]


def script_curve(key, script):
    d = json.load(open(RESULTS / key / "metrics.json"))["script_mass_tgt"][script]
    xs = sorted(int(k) for k in d)
    return xs, [d[str(x)] for x in xs]


def line_fig(name, series, ylabel, getter, legend_loc="best", full_ylim=False):
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    for k in series:
        xs, ys = getter(k)
        ax.plot(xs, ys, color=COLORS[k], linewidth=2.0, label=LABELS[k])
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 1.0) if full_ylim else ax.set_ylim(bottom=0)
    style_axis(ax, ygrid=True)
    ax.legend(frameon=False, loc=legend_loc)
    fig.tight_layout()
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


def ko_layerwise_panel():
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.45), sharex=True)
    panels = [
        (
            "(A) Geometric alignment",
            "Linear CKA",
            lambda k: field_curve(k, "cka"),
            True,
        ),
        (
            "(B) English-hub routing",
            "Latin-script probability mass",
            lambda k: script_curve(k, "latin"),
            True,
        ),
        (
            "(C) Functional divergence",
            "Binned Logit Lens JSD",
            lambda k: field_curve(k, "logit_lens_jsd"),
            False,
        ),
    ]

    for ax, (title, ylabel, getter, full_ylim) in zip(axes, panels):
        for k in ["llama2_ko", "koen"]:
            xs, ys = getter(k)
            ax.plot(xs, ys, color=COLORS[k], linewidth=2.0, label=LABELS[k])
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 40)
        ax.set_ylim(0, 1.0) if full_ylim else ax.set_ylim(bottom=0)
        style_axis(ax, ygrid=True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        handlelength=2.2,
        columnspacing=1.6,
    )
    fig.tight_layout(rect=(0, 0.18, 1, 1), w_pad=1.0)
    fig.savefig(OUT / "fig_ko_layerwise.pdf")
    plt.close(fig)
    print("wrote", "fig_ko_layerwise.pdf")


if __name__ == "__main__":
    line_fig("fig_jsd_layerwise.pdf", ["llama2", "swallow", "llm_jp"],
             "Binned Logit Lens JSD", lambda k: field_curve(k, "logit_lens_jsd"),
             legend_loc="upper left")
    line_fig("fig_ko_cka.pdf", ["llama2_ko", "koen"], "Linear CKA",
             lambda k: field_curve(k, "cka"), legend_loc="lower right", full_ylim=True)
    line_fig("fig_ko_latin.pdf", ["llama2_ko", "koen"], "Latin-script probability mass",
             lambda k: script_curve(k, "latin"), legend_loc="upper right", full_ylim=True)
    line_fig("fig_ko_jsd.pdf", ["llama2_ko", "koen"], "Binned Logit Lens JSD",
             lambda k: field_curve(k, "logit_lens_jsd"), legend_loc="upper left")
    ko_layerwise_panel()
