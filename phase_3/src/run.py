"""
run.py — Multi-molecule experiment grid with improved visualisation.

Molecules: aspirin (3 bonds), lidocaine (6), fluoxetine (7).
Grid: 5 strategies × 4 initialisations × 3 cutoffs = 60 experiments per molecule.

Visualisation fixes:
  1. Smoothed curves (rolling average, window=30) — kills step-to-step noise
  2. Relative energy E−E_best on symlog Y — makes all inits readable together
  3. Log colour scale on energy heatmap — 40 kcal/mol and 400 000 on same map
  4. Median (not mean) in bar charts — robust to pathological zeros init
  5. Per-step energy snapshots — captures the first-step relaxation from anti/zeros
"""

from __future__ import annotations

import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from model import MoleculeModel

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

MOLECULES = [
    ("aspirin",     "CC(=O)Oc1ccccc1C(=O)O"),
    ("lidocaine",   "CCN(CC)CC(=O)Nc1c(C)cccc1C"),
    ("fluoxetine",  "CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1"),
]

STRATEGIES = [
    "isolated",
    "local_greed",
    "consensus",
    "adaptive_density",
    "gradient_exchange",
]
INITS = ["etkdg", "random", "zeros", "anti"]
CUTOFFS = [2.5, 4.0, 7.0]

N_STEPS = 1000
TEMPERATURE = 300.0
SEED = 42

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Colour palettes
INIT_COLORS = {
    "etkdg": "#1f77b4",
    "random": "#ff7f0e",
    "zeros": "#2ca02c",
    "anti": "#d62728",
}
INIT_STYLES = {
    "etkdg": "-",
    "random": "--",
    "zeros": "-.",
    "anti": ":",
}
STRATEGY_COLORS = {
    "isolated": "#1f77b4",
    "local_greed": "#ff7f0e",
    "consensus": "#2ca02c",
    "adaptive_density": "#d62728",
    "gradient_exchange": "#9467bd",
}
STRATEGY_LABELS = {
    "isolated": "isolated",
    "local_greed": "local greed",
    "consensus": "consensus",
    "adaptive_density": "adaptive\ndensity",
    "gradient_exchange": "gradient\nexchange",
}


# ---------------------------------------------------------------------------
# Run grid
# ---------------------------------------------------------------------------


def run_grid() -> tuple[list[dict], dict]:
    """Run all experiments; return (results_list, models_dict)."""
    total = len(MOLECULES) * len(STRATEGIES) * len(INITS) * len(CUTOFFS)
    print(f"Running {total} experiments ({len(MOLECULES)} molecules, "
          f"steps={N_STEPS}) …\n")

    results: list[dict] = []
    models: dict = {}  # (mol_name, strategy, init, cutoff) → model
    done = 0

    for mol_name, smiles in MOLECULES:
        print(f"\n{'='*60}")
        print(f"  {mol_name.upper()}  ({smiles})")
        print(f"{'='*60}")
        for strategy in STRATEGIES:
            for init in INITS:
                for cutoff in CUTOFFS:
                    done += 1
                    label = (f"[{done:3d}/{total}] {mol_name:12s} "
                             f"{strategy:<20s} {init:<8s} cut={cutoff:.1f}")
                    t0 = time.time()
                    print(label, end="  ", flush=True)

                    model = MoleculeModel(
                        smiles=smiles,
                        strategy=strategy,
                        init=init,
                        comm_cutoff=cutoff,
                        n_steps=N_STEPS,
                        temperature=TEMPERATURE,
                        seed=SEED,
                    )
                    model.run()
                    s = model.summary()
                    s["molecule"] = mol_name
                    s["n_bonds"] = len(model.bonds)
                    results.append(s)
                    models[(mol_name, strategy, init, cutoff)] = model

                    dt = time.time() - t0
                    print(
                        f"E={s['final_energy']:9.1f}  "
                        f"ΔE={s['energy_drop']:9.1f}  "
                        f"acc={s['mean_acceptance']:.2f}  "
                        f"({dt:.1f}s)"
                    )

    return results, models


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "molecule", "n_bonds", "strategy", "init", "comm_cutoff",
    "final_energy", "energy_drop", "converge_step",
    "mean_coverage", "mean_acceptance", "mean_neighbors",
]


def save_csv(results: list[dict]) -> None:
    path = os.path.join(RESULTS_DIR, "all_metrics.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {path}")


# ---------------------------------------------------------------------------
# Smoothing helper
# ---------------------------------------------------------------------------

SMOOTH_W = 30  # rolling-average half-window


def _smooth(y: np.ndarray, w: int = SMOOTH_W) -> np.ndarray:
    """Causal rolling average with window 2*w+1.  No future leakage."""
    kernel = np.ones(2 * w + 1) / (2 * w + 1)
    # pad front with first value so the curve doesn't start with artefacts
    padded = np.concatenate([np.full(w, y[0]), y, np.full(w, y[-1])])
    return np.convolve(padded, kernel, mode="valid")[: len(y)]


# ---------------------------------------------------------------------------
# Plot 1: Convergence (smoothed, relative energy, symlog Y) — per molecule
# ---------------------------------------------------------------------------


def plot_convergence(models: dict, results: list[dict]) -> None:
    cutoff = 4.0

    for mol_name, _ in MOLECULES:
        e_best = min(
            r["final_energy"]
            for r in results
            if r["molecule"] == mol_name and abs(r["comm_cutoff"] - cutoff) < 0.01
        )

        fig, axes = plt.subplots(1, len(STRATEGIES), figsize=(24, 5),
                                 sharey=True)
        fig.suptitle(
            f"{mol_name.capitalize()} — energy convergence  "
            f"(cutoff = {cutoff} Å,  E_best = {e_best:.1f} kcal/mol,  "
            f"smoothed w={SMOOTH_W})",
            fontsize=12, y=1.02,
        )

        for ax, strategy in zip(axes, STRATEGIES):
            for init in INITS:
                m = models.get((mol_name, strategy, init, cutoff))
                if m is None:
                    continue
                snaps = np.array(m.energy_snapshots, dtype=float)
                e_rel = np.maximum(snaps - e_best, 0.01)  # floor at 0.01
                smoothed = _smooth(e_rel)
                steps = np.arange(len(smoothed))
                ax.plot(
                    steps, smoothed,
                    label=init,
                    color=INIT_COLORS[init],
                    linestyle=INIT_STYLES[init],
                    linewidth=1.8,
                )
            ax.set_yscale("log")
            ax.set_title(STRATEGY_LABELS[strategy], fontsize=10)
            ax.set_xlabel("Step")
            if ax is axes[0]:
                ax.set_ylabel("E − E_best  (kcal/mol, log)")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(alpha=0.25, which="both")

        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, f"convergence_{mol_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Plot 2: Cutoff effect — per molecule, log Y when range is huge
# ---------------------------------------------------------------------------


def plot_cutoff_effect(results: list[dict]) -> None:
    for mol_name, _ in MOLECULES:
        mol_res = [r for r in results if r["molecule"] == mol_name]
        fig, axes = plt.subplots(1, len(INITS), figsize=(20, 5), sharey=False)
        fig.suptitle(
            f"{mol_name.capitalize()} — final energy vs communication cutoff",
            fontsize=12, y=1.02,
        )

        for ax, init in zip(axes, INITS):
            for strategy in STRATEGIES:
                ys = [
                    next(
                        (r["final_energy"] for r in mol_res
                         if r["strategy"] == strategy
                         and r["init"] == init
                         and abs(r["comm_cutoff"] - c) < 0.01),
                        np.nan,
                    )
                    for c in CUTOFFS
                ]
                ax.plot(
                    CUTOFFS, ys, marker="o", markersize=6,
                    label=strategy.replace("_", " "),
                    color=STRATEGY_COLORS[strategy],
                    linewidth=1.8,
                )
            # Use log Y if range spans > 10×
            yvals = [r["final_energy"] for r in mol_res if r["init"] == init]
            if yvals and max(yvals) / max(min(yvals), 1) > 10:
                ax.set_yscale("log")
            ax.set_title(f"init = {init}", fontsize=10)
            ax.set_xlabel("Cutoff (Å)")
            ax.set_ylabel("Final E (kcal/mol)")
            ax.set_xticks(CUTOFFS)
            ax.legend(fontsize=7, loc="best")
            ax.grid(alpha=0.25, which="both")

        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, f"cutoff_effect_{mol_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Plot 3: Proximity dynamics — smoothed, per molecule
# ---------------------------------------------------------------------------


def plot_proximity_dynamics(models: dict) -> None:
    init = "etkdg"

    for mol_name, _ in MOLECULES:
        fig, axes = plt.subplots(1, len(CUTOFFS), figsize=(18, 5), sharey=False)
        fig.suptitle(
            f"{mol_name.capitalize()} — active proximity edges over time  "
            f"(init = {init},  smoothed w={SMOOTH_W})",
            fontsize=12, y=1.02,
        )

        for ax, cutoff in zip(axes, CUTOFFS):
            for strategy in STRATEGIES:
                m = models.get((mol_name, strategy, init, cutoff))
                if m is None:
                    continue
                edges = np.array(m.scheduler.proximity_edge_history, dtype=float)
                ax.plot(
                    _smooth(edges),
                    label=strategy.replace("_", " "),
                    color=STRATEGY_COLORS[strategy],
                    linewidth=1.5,
                )
            ax.set_title(f"cutoff = {cutoff} Å", fontsize=10)
            ax.set_xlabel("Step")
            ax.set_ylabel("Active edges")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.25)

        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, f"proximity_dynamics_{mol_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Plot 4 & 5: Heatmaps — log colour scale for energy
# ---------------------------------------------------------------------------


def _mat_for(results, mol_name, metric, cutoff=4.0):
    mat = np.full((len(STRATEGIES), len(INITS)), np.nan)
    for r in results:
        if r["molecule"] == mol_name and abs(r["comm_cutoff"] - cutoff) < 0.01:
            si = STRATEGIES.index(r["strategy"])
            ii = INITS.index(r["init"])
            mat[si, ii] = r[metric]
    return mat


def plot_heatmaps(results: list[dict]) -> None:
    cutoff = 4.0

    for metric, cbar_label, fname, use_log in [
        ("final_energy", "Final E (kcal/mol)", "heatmap_energy.png", True),
        ("mean_coverage", "Coverage (fraction)", "heatmap_coverage.png", False),
    ]:
        n_mol = len(MOLECULES)
        fig, axes = plt.subplots(1, n_mol, figsize=(6.5 * n_mol, 5.5))
        if n_mol == 1:
            axes = [axes]

        for ax, (mol_name, _) in zip(axes, MOLECULES):
            mat = _mat_for(results, mol_name, metric, cutoff)

            if use_log:
                mat_plot = np.where(mat > 0, mat, np.nan)
                vmin = max(np.nanmin(mat_plot), 1.0)
                vmax = np.nanmax(mat_plot)
                im = ax.imshow(
                    mat_plot, aspect="auto", cmap="RdYlGn_r",
                    norm=LogNorm(vmin=vmin, vmax=vmax),
                )
            else:
                im = ax.imshow(mat, aspect="auto", cmap="YlGnBu")

            ax.set_xticks(range(len(INITS)))
            ax.set_xticklabels(INITS, fontsize=9)
            ax.set_yticks(range(len(STRATEGIES)))
            ax.set_yticklabels(
                [STRATEGY_LABELS[s] for s in STRATEGIES], fontsize=8
            )
            ax.set_xlabel("Initialisation", fontsize=9)
            ax.set_title(f"{mol_name}", fontsize=11)

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=7)

            for si in range(len(STRATEGIES)):
                for ii in range(len(INITS)):
                    v = mat[si, ii]
                    if np.isnan(v):
                        continue
                    if v >= 1000:
                        txt = f"{v:.0e}"
                    elif v >= 10:
                        txt = f"{v:.0f}"
                    else:
                        txt = f"{v:.2f}"
                    # contrast: dark text on bright cells, white on dark
                    lum = (v - np.nanmin(mat)) / max(np.nanmax(mat) - np.nanmin(mat), 1)
                    color = "white" if lum > 0.55 else "black"
                    if use_log and v > 0:
                        lum_log = ((np.log10(v) - np.log10(max(np.nanmin(mat), 1)))
                                   / max(np.log10(max(np.nanmax(mat), 1))
                                         - np.log10(max(np.nanmin(mat), 1)), 1))
                        color = "white" if lum_log > 0.55 else "black"
                    ax.text(ii, si, txt, ha="center", va="center",
                            fontsize=7, fontweight="bold", color=color)

        fig.suptitle(
            f"{cbar_label} — strategy × init  (cutoff = {cutoff} Å)",
            fontsize=13,
        )
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, fname)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Plot 6: Strategy comparison — median across inits, log Y for energy
# ---------------------------------------------------------------------------


def plot_strategy_comparison(results: list[dict]) -> None:
    cutoff = 4.0
    mol_names = [m[0] for m in MOLECULES]
    n_s = len(STRATEGIES)
    x = np.arange(len(mol_names))
    width = 0.15

    fig, (ax_e, ax_c) = plt.subplots(1, 2, figsize=(15, 5.5))

    for k, strategy in enumerate(STRATEGIES):
        meds_e, meds_c = [], []
        for mol_name in mol_names:
            vals_e = [
                r["final_energy"]
                for r in results
                if r["molecule"] == mol_name
                and r["strategy"] == strategy
                and abs(r["comm_cutoff"] - cutoff) < 0.01
            ]
            vals_c = [
                r["mean_coverage"]
                for r in results
                if r["molecule"] == mol_name
                and r["strategy"] == strategy
                and abs(r["comm_cutoff"] - cutoff) < 0.01
            ]
            meds_e.append(float(np.median(vals_e)))
            meds_c.append(float(np.median(vals_c)))

        offset = (k - n_s / 2 + 0.5) * width
        ax_e.bar(x + offset, meds_e, width,
                 label=strategy.replace("_", " "),
                 color=STRATEGY_COLORS[strategy])
        ax_c.bar(x + offset, meds_c, width,
                 color=STRATEGY_COLORS[strategy])

    ax_e.set_xticks(x)
    ax_e.set_xticklabels(mol_names, fontsize=10)
    ax_e.set_ylabel("Median final E (kcal/mol)")
    ax_e.set_yscale("log")
    ax_e.set_title("Final energy  (lower = better)", fontsize=11)
    ax_e.legend(fontsize=8, ncol=2)
    ax_e.grid(axis="y", alpha=0.25, which="both")

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(mol_names, fontsize=10)
    ax_c.set_ylabel("Median angular coverage")
    ax_c.set_title("Conformational coverage  (higher = better)", fontsize=11)
    ax_c.legend(fontsize=8, ncol=2, labels=[
        s.replace("_", " ") for s in STRATEGIES])
    ax_c.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Strategy comparison across molecules  "
        "(cutoff = 4.0 Å, median over 4 inits)",
        fontsize=13,
    )
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "strategy_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Plot 7: Convergence speed
# ---------------------------------------------------------------------------


def plot_convergence_speed(results: list[dict]) -> None:
    cutoff = 4.0
    mol_names = [m[0] for m in MOLECULES]
    n_s = len(STRATEGIES)
    x = np.arange(len(mol_names))
    width = 0.15

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for k, strategy in enumerate(STRATEGIES):
        meds = []
        for mol_name in mol_names:
            vals = [
                r["converge_step"]
                for r in results
                if r["molecule"] == mol_name
                and r["strategy"] == strategy
                and abs(r["comm_cutoff"] - cutoff) < 0.01
            ]
            meds.append(float(np.median(vals)))

        offset = (k - n_s / 2 + 0.5) * width
        ax.bar(x + offset, meds, width, label=strategy.replace("_", " "),
               color=STRATEGY_COLORS[strategy])

    ax.set_xticks(x)
    ax.set_xticklabels(mol_names, fontsize=10)
    ax.set_ylabel("Convergence step (median)")
    ax.set_title(
        "Convergence speed — step at which E ≤ E_final + 0.5 kcal/mol  "
        "(cutoff = 4.0 Å, median over inits)",
        fontsize=11,
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "convergence_speed.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------


def print_summary(results: list[dict]) -> None:
    header = (
        f"{'molecule':<14} {'strategy':<22} {'init':<8} {'cut':>4}  "
        f"{'E_final':>10}  {'ΔE':>10}  {'acc':>5}  {'cov':>5}  {'conv':>5}"
    )
    sep = "-" * len(header)

    for mol_name, _ in MOLECULES:
        mol_res = [r for r in results if r["molecule"] == mol_name]
        print(f"\n{sep}")
        print(header)
        print(sep)
        for r in mol_res:
            print(
                f"{r['molecule']:<14} {r['strategy']:<22} {r['init']:<8} "
                f"{r['comm_cutoff']:>4.1f}  "
                f"{r['final_energy']:>10.1f}  {r['energy_drop']:>10.1f}  "
                f"{r['mean_acceptance']:>5.2f}  {r['mean_coverage']:>5.2f}  "
                f"{r['converge_step']:>5d}"
            )
    print(sep)


# ---------------------------------------------------------------------------
# Annealing comparison grid & plot
# ---------------------------------------------------------------------------

# Reference energies (200-conformer L-BFGS search)
REF_ENERGY = {
    "aspirin": 18.91,
    "lidocaine": 53.31,
    "fluoxetine": 73.99,
}

MODES = {
    "fixed_300K":    dict(temperature=300., annealing=False, pre_minimize=False),
    "premin_300K":   dict(temperature=300., annealing=False, pre_minimize=True),
    "premin_anneal": dict(temperature=300., annealing=True,  T_start=800., pre_minimize=True),
}

MODE_LABELS = {
    "fixed_300K":    "Fixed T=300 K",
    "premin_300K":   "Pre-min + fixed T=300 K",
    "premin_anneal": "Pre-min + anneal 800→300 K",
}
MODE_COLORS = {
    "fixed_300K":    "#aec7e8",
    "premin_300K":   "#ffbb78",
    "premin_anneal": "#98df8a",
}


def run_annealing_grid(n_seeds: int = 3) -> tuple[list[dict], dict]:
    """
    Targeted grid: 5 strategies × 3 molecules × 3 modes × n_seeds.
    Only etkdg init, cutoff=4.0 Å.  Returns results + model store.
    """
    ann_results: list[dict] = []
    ann_models: dict = {}  # (mol, strategy, mode, seed) → model
    cutoff = 4.0
    total = len(MOLECULES) * len(STRATEGIES) * len(MODES) * n_seeds
    done = 0

    print(f"\n{'='*60}")
    print(f"  ANNEALING COMPARISON  ({total} runs)")
    print(f"{'='*60}")

    for mol_name, smiles in MOLECULES:
        for strategy in STRATEGIES:
            for mode_key, mode_kw in MODES.items():
                efs, covs, accs = [], [], []
                best_model = None
                for seed in range(n_seeds):
                    done += 1
                    m = MoleculeModel(
                        smiles=smiles,
                        strategy=strategy,
                        init="etkdg",
                        comm_cutoff=cutoff,
                        n_steps=800,
                        seed=seed + 42,
                        **mode_kw,
                    )
                    m.run()
                    s = m.summary()
                    efs.append(s["final_energy"])
                    covs.append(s["mean_coverage"])
                    accs.append(s["mean_acceptance"])
                    if best_model is None or s["final_energy"] < efs[0]:
                        best_model = m
                ann_models[(mol_name, strategy, mode_key)] = best_model
                ann_results.append({
                    "molecule":    mol_name,
                    "strategy":    strategy,
                    "mode":        mode_key,
                    "final_energy": float(np.mean(efs)),
                    "final_energy_std": float(np.std(efs)),
                    "mean_coverage": float(np.mean(covs)),
                    "mean_acceptance": float(np.mean(accs)),
                })
                print(
                    f"  [{done:3d}/{total}] {mol_name:12} {strategy:22} "
                    f"{mode_key:15}  E={np.mean(efs):.1f}±{np.std(efs):.1f}"
                )

    return ann_results, ann_models


def plot_annealing_comparison(ann_results: list[dict], ann_models: dict) -> None:
    """Two-panel figure per molecule: bar chart (energy) + convergence curves."""
    n_s = len(STRATEGIES)
    x = np.arange(n_s)
    width = 0.25

    for mol_name, _ in MOLECULES:
        ref = REF_ENERGY[mol_name]
        mol_res = [r for r in ann_results if r["molecule"] == mol_name]

        fig, (ax_bar, ax_conv) = plt.subplots(1, 2, figsize=(16, 5.5))
        fig.suptitle(
            f"{mol_name.capitalize()} — impact of pre-minimisation + annealing  "
            f"(cutoff=4 Å, etkdg, mean ± std over 3 seeds)",
            fontsize=12, y=1.02,
        )

        # --- Bar chart: final energy per strategy × mode ---
        for k, (mode_key, mode_label) in enumerate(MODE_LABELS.items()):
            efs = [
                next((r["final_energy"] for r in mol_res
                      if r["strategy"] == s and r["mode"] == mode_key), np.nan)
                for s in STRATEGIES
            ]
            stds = [
                next((r["final_energy_std"] for r in mol_res
                      if r["strategy"] == s and r["mode"] == mode_key), 0.0)
                for s in STRATEGIES
            ]
            offset = (k - len(MODES) / 2 + 0.5) * width
            bars = ax_bar.bar(x + offset, efs, width, label=mode_label,
                              color=MODE_COLORS[mode_key], edgecolor="grey",
                              linewidth=0.5)
            ax_bar.errorbar(x + offset, efs, yerr=stds, fmt="none",
                            color="black", capsize=3, linewidth=1)

        ax_bar.axhline(ref, color="red", linestyle="--", linewidth=1.5,
                       label=f"200-conf+min ref ({ref:.1f})")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([s.replace("_", "\n") for s in STRATEGIES], fontsize=8)
        ax_bar.set_ylabel("Mean final E (kcal/mol)")
        ax_bar.set_title("Final energy per strategy and mode", fontsize=10)
        ax_bar.legend(fontsize=8)
        ax_bar.set_yscale("log")
        ax_bar.grid(axis="y", alpha=0.25, which="both")

        # --- Convergence curves: best mode (premin_anneal), all strategies ---
        for strategy in STRATEGIES:
            m = ann_models.get((mol_name, strategy, "premin_anneal"))
            if m is None:
                continue
            snaps = np.array(m.energy_snapshots, dtype=float)
            e_rel = np.maximum(snaps - ref, 0.01)
            ax_conv.plot(
                _smooth(e_rel),
                label=strategy.replace("_", " "),
                color=STRATEGY_COLORS[strategy],
                linewidth=1.8,
            )

        ax_conv.set_yscale("log")
        ax_conv.axhline(0.1, color="red", linestyle="--", linewidth=1,
                        label="ref (0.1 = exact min)")
        ax_conv.set_xlabel("Step")
        ax_conv.set_ylabel("E − E_ref  (kcal/mol, log)")
        ax_conv.set_title("Convergence — pre-min + anneal 800→300 K", fontsize=10)
        ax_conv.legend(fontsize=8)
        ax_conv.grid(alpha=0.25, which="both")

        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, f"annealing_{mol_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")

    # --- Combined bar: all molecules, best mode vs fixed ---
    fig, axes = plt.subplots(1, len(MOLECULES), figsize=(6 * len(MOLECULES), 5.5))
    for ax, (mol_name, _) in zip(axes, MOLECULES):
        ref = REF_ENERGY[mol_name]
        mol_res = [r for r in ann_results if r["molecule"] == mol_name]
        for k, (mode_key, mode_label) in enumerate(MODE_LABELS.items()):
            efs = [
                next((r["final_energy"] for r in mol_res
                      if r["strategy"] == s and r["mode"] == mode_key), np.nan)
                for s in STRATEGIES
            ]
            stds = [
                next((r["final_energy_std"] for r in mol_res
                      if r["strategy"] == s and r["mode"] == mode_key), 0.0)
                for s in STRATEGIES
            ]
            x = np.arange(len(STRATEGIES))
            offset = (k - len(MODES) / 2 + 0.5) * width
            ax.bar(x + offset, efs, width, label=mode_label,
                   color=MODE_COLORS[mode_key], edgecolor="grey", linewidth=0.5)
            ax.errorbar(x + offset, efs, yerr=stds, fmt="none",
                        color="black", capsize=2, linewidth=1)
        ax.axhline(ref, color="red", linestyle="--", linewidth=1.5)
        ax.set_xticks(np.arange(len(STRATEGIES)))
        ax.set_xticklabels([s.replace("_", "\n") for s in STRATEGIES], fontsize=7)
        ax.set_title(f"{mol_name}", fontsize=11)
        ax.set_ylabel("Final E (kcal/mol)")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25, which="both")
        if ax is axes[0]:
            ax.legend(fontsize=7)

    fig.suptitle(
        "Pre-minimisation + annealing comparison  (cutoff = 4.0 Å, init = etkdg)",
        fontsize=13,
    )
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "annealing_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t_start = time.time()
    results, models = run_grid()
    save_csv(results)

    print("\nGenerating main plots …")
    plot_convergence(models, results)
    plot_cutoff_effect(results)
    plot_proximity_dynamics(models)
    plot_heatmaps(results)
    plot_strategy_comparison(results)
    plot_convergence_speed(results)
    print_summary(results)

    print("\nRunning annealing comparison …")
    ann_results, ann_models = run_annealing_grid(n_seeds=3)
    plot_annealing_comparison(ann_results, ann_models)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.1f} min.  Results written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
