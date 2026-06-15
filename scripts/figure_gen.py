#!/usr/bin/env python3
"""Generate final paper figures from local result JSON files.

Style: restrained ACL-ready plots, three model colors plus gray baselines,
Times-like serif font, no rainbow palettes, no decorative effects.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS = Path("results")
OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "llama2": "#D6604D",
    "swallow": "#4D9221",
    "llm_jp": "#2166AC",
    "llama2_ko": "#D6604D",
    "koen": "#4D9221",
}

LABELS = {
    "llama2": "Llama-2",
    "swallow": "Swallow",
    "llm_jp": "LLM-jp",
    "llama2_ko": "Llama-2",
    "koen": "koen",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def metrics(model):
    return load_json(RESULTS / model / "metrics.json")


def patch_records(model, multi=False):
    name = "patching_multi.json" if multi else "patching.json"
    return load_json(RESULTS / model / name)["records"]


def layer_values(data, key):
    xs = list(range(data["num_layers"] + 1))
    return xs, [data[key][str(layer)] for layer in xs]


def script_values(data, script):
    xs = list(range(data["num_layers"] + 1))
    return xs, [data["script_mass_tgt"][script][str(layer)] for layer in xs]


def mid_latin(model):
    latin = metrics(model)["script_mass_tgt"]["latin"]
    return float(np.mean([latin[str(layer)] for layer in range(10, 26)]))


def boot_ci(values, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def style_axis(ax, ygrid=False):
    ax.tick_params(axis="both", direction="out", length=3, width=0.7, color="#555555")
    if ygrid:
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
        ax.set_axisbelow(True)


def fig_hub_routing():
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.35), sharex=True)

    for model in ["llama2", "swallow", "llm_jp"]:
        data = metrics(model)
        xs, cka = layer_values(data, "cka")
        _, latin = script_values(data, "latin")
        axes[0].plot(xs, cka, color=COLORS[model], linewidth=2.0, label=LABELS[model])
        axes[1].plot(xs, latin, color=COLORS[model], linewidth=2.0, label=LABELS[model])

    axes[0].set_title("(A) Geometric alignment")
    axes[0].set_ylabel("Linear CKA")
    axes[0].set_ylim(0, 1.0)

    axes[1].set_title("(B) English-hub routing")
    axes[1].set_ylabel("Latin-script probability mass")
    axes[1].set_ylim(0, 1.0)

    for ax in axes:
        ax.set_xlabel("Layer")
        ax.set_xlim(0, 40)
        style_axis(ax, ygrid=True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        handlelength=2.2,
        columnspacing=1.5,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=1.2)
    fig.savefig(OUT / "fig_hub_routing.pdf")
    plt.close(fig)


def fig_adaptation():
    fig, ax = plt.subplots(figsize=(3.25, 2.18))

    items = [
        ("Japanese", "llama2"),
        ("Japanese", "swallow"),
        ("Japanese", "llm_jp"),
        ("Korean", "llama2_ko"),
        ("Korean", "koen"),
    ]
    x = np.array([0, 1, 2, 3.6, 4.6], dtype=float)
    values = [mid_latin(model) for _, model in items]

    ax.bar(
        x,
        values,
        width=0.72,
        color=[COLORS[model] for _, model in items],
        edgecolor="none",
        alpha=0.92,
    )

    ax.axvline(2.8, color="#bdbdbd", linestyle=":", linewidth=1.0)
    ax.text(1.0, 0.96, "Japanese", ha="center", va="top", fontsize=8)
    ax.text(4.1, 0.96, "Korean", ha="center", va="top", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[model] for _, model in items], rotation=25, ha="right")
    ax.set_ylabel("Mid-layer Latin mass")
    ax.set_ylim(0, 1.0)
    style_axis(ax, ygrid=True)

    for xpos, value in zip(x, values):
        ax.text(xpos, value + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "fig_adaptation.pdf")
    plt.close(fig)


def fig_patching_null():
    rows = []
    for model in ["llama2", "swallow", "llm_jp"]:
        recs = patch_records(model)
        effects = [r["gap_paired"] - r["gap_random"] for r in recs]
        rows.append((LABELS[model], "peak layer", *boot_ci(effects)))

    recs = patch_records("llama2", multi=True)
    effects = [r["gap_paired"] - r["gap_random"] for r in recs]
    rows.append(("Llama-2", "layers 10-25", *boot_ci(effects)))

    for model in ["llama2_ko", "koen"]:
        recs = patch_records(model)
        effects = [r["gap_paired"] - r["gap_random"] for r in recs]
        rows.append((LABELS[model] + " (KO)", "peak layer", *boot_ci(effects)))

    labels = [f"{model}\n{patch}" for model, patch, *_ in rows]
    means = np.array([row[2] for row in rows])
    lows = np.array([row[3] for row in rows])
    highs = np.array([row[4] for row in rows])
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(3.35, 3.3))
    ax.axvspan(-0.02, 0.02, color="#f2f2f2", zorder=0)
    ax.axvline(0, color="#5f5f5f", linewidth=1.1, zorder=1)

    ax.errorbar(
        means,
        y,
        xerr=[means - lows, highs - means],
        fmt="o",
        color="#333333",
        ecolor="#6f6f6f",
        elinewidth=1.8,
        capsize=3.0,
        markersize=4.8,
        markerfacecolor="#333333",
        markeredgewidth=0,
        zorder=2,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Transplant effect (paired - random)")
    ax.set_xlim(-0.11, 0.11)
    ax.set_xticks([-0.10, -0.05, 0, 0.05, 0.10])
    style_axis(ax, ygrid=False)

    for mean, ypos in zip(means, y):
        ax.text(
            0.105,
            ypos,
            f"{mean:+.3f}",
            ha="right",
            va="center",
            fontsize=8,
            color="#333333",
        )

    fig.tight_layout()
    fig.savefig(OUT / "fig_patching_null.pdf")
    plt.close(fig)


def print_numbers():
    print("=== Figure values ===")
    for model in ["llama2", "swallow", "llm_jp", "llama2_ko", "koen"]:
        data = metrics(model)
        peak = data["cka_peak"]
        print(f"{model:10s} CKA={peak['value']:.3f}@L{peak['layer']} mid_latin={mid_latin(model):.3f}")

    print("\n=== Bias asymmetry and patching transplant ===")
    for model in ["llama2", "swallow", "llm_jp"]:
        recs = patch_records(model)
        diff = [r["pstereo_en"] - r["pstereo_tgt"] for r in recs]
        effect = [r["gap_paired"] - r["gap_random"] for r in recs]
        dm, dlo, dhi = boot_ci(diff)
        em, elo, ehi = boot_ci(effect)
        print(
            f"{model:8s} Pdiff={dm:+.3f} [{dlo:+.3f}, {dhi:+.3f}] "
            f"transplant={em:+.3f} [{elo:+.3f}, {ehi:+.3f}]"
        )

    recs = patch_records("llama2", multi=True)
    em, elo, ehi = boot_ci([r["gap_paired"] - r["gap_random"] for r in recs])
    print(f"{'llama2':8s} multi transplant={em:+.3f} [{elo:+.3f}, {ehi:+.3f}]")


if __name__ == "__main__":
    fig_hub_routing()
    fig_adaptation()
    fig_patching_null()
    print_numbers()
    print(f"\nSaved figures to {OUT}")
