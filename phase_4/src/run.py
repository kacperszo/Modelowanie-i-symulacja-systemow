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
# Heterogeneous strategy experiments
# ---------------------------------------------------------------------------

# Majority strategy is always isolated (neutral baseline).
# We vary how many bonds use a minority strategy.
MINORITY_STRATEGIES = ["consensus", "gradient_exchange"]
HETERO_INITS = ["zeros", "etkdg"]

MINORITY_COLORS = {
    "consensus": "#d62728",
    "gradient_exchange": "#1f77b4",
}


def _n_bonds_for(smiles: str) -> int:
    """Return number of rotatable bonds without running a full simulation."""
    m = MoleculeModel(smiles, strategy="isolated", n_steps=0, seed=0)
    return len(m.bonds)


def run_heterogeneous_grid(n_seeds: int = 3) -> tuple:
    """
    For each molecule × minority_strategy × n_minority_agents × init × seed:
      - Build strategy list: first n_minority bonds = minority, rest = isolated.
      - Run 1000 steps at T=300K, cutoff=4.0Å.
    Returns (results_list, models_dict).
    """
    results = []
    models_dict = {}

    for mol_name, smiles in MOLECULES:
        n_bonds = _n_bonds_for(smiles)
        print(f"\n{mol_name} ({n_bonds} bonds)")

        for minority_strat in MINORITY_STRATEGIES:
            for n_minority in range(n_bonds + 1):
                strategy_list = (
                    [minority_strat] * n_minority + ["isolated"] * (n_bonds - n_minority)
                )
                fraction = n_minority / n_bonds

                for init in HETERO_INITS:
                    energies, acceptances_minority, acceptances_isolated = [], [], []

                    for seed in range(n_seeds):
                        m = MoleculeModel(
                            smiles,
                            strategy=strategy_list,
                            init=init,
                            comm_cutoff=4.0,
                            n_steps=N_STEPS,
                            temperature=TEMPERATURE,
                            seed=seed,
                        )
                        m.run()
                        s = m.summary()
                        energies.append(s["final_energy"])
                        pa = s["per_strategy_acceptance"]
                        if minority_strat in pa:
                            acceptances_minority.append(pa[minority_strat])
                        if "isolated" in pa:
                            acceptances_isolated.append(pa["isolated"])

                        if seed == 0:
                            models_dict[(mol_name, minority_strat, n_minority, init)] = m

                    label = (
                        minority_strat if n_minority == n_bonds
                        else ("isolated" if n_minority == 0 else f"{n_minority}x{minority_strat}")
                    )
                    results.append({
                        "molecule": mol_name,
                        "n_bonds": n_bonds,
                        "minority_strategy": minority_strat,
                        "n_minority": n_minority,
                        "fraction_minority": fraction,
                        "init": init,
                        "label": label,
                        "final_energy_mean": float(np.mean(energies)),
                        "final_energy_std": float(np.std(energies)),
                        "acc_minority_mean": float(np.mean(acceptances_minority)) if acceptances_minority else np.nan,
                        "acc_isolated_mean": float(np.mean(acceptances_isolated)) if acceptances_isolated else np.nan,
                    })
                    tag = f"n_minority={n_minority} init={init}"
                    print(f"  {minority_strat:20s} {tag:30s}  E={np.mean(energies):.1f}")

    return results, models_dict


def plot_heterogeneous(results: list, models_dict: dict) -> None:
    """
    Two figures:
    1. Deadlock threshold: E_final vs n_minority_agents, per molecule, per init.
    2. Per-strategy acceptance rate: how consensus/gradient_exchange acceptance
       differs from isolated neighbours in the same population.
    """

    # ---- Figure 1: final energy vs fraction minority agents ----
    fig, axes = plt.subplots(
        len(MINORITY_STRATEGIES), len(MOLECULES),
        figsize=(5.5 * len(MOLECULES), 4.5 * len(MINORITY_STRATEGIES)),
        squeeze=False,
    )

    init_styles = {"zeros": ("o", "-"), "etkdg": ("s", "--")}
    mol_colors = {
        "aspirin": "#2ca02c",
        "lidocaine": "#ff7f0e",
        "fluoxetine": "#9467bd",
    }

    for row, minority_strat in enumerate(MINORITY_STRATEGIES):
        for col, (mol_name, _) in enumerate(MOLECULES):
            ax = axes[row][col]
            subset = [
                r for r in results
                if r["molecule"] == mol_name and r["minority_strategy"] == minority_strat
            ]

            for init in HETERO_INITS:
                pts = sorted(
                    [r for r in subset if r["init"] == init],
                    key=lambda r: r["n_minority"],
                )
                if not pts:
                    continue
                xs = [p["fraction_minority"] for p in pts]
                ys = [p["final_energy_mean"] for p in pts]
                yerr = [p["final_energy_std"] for p in pts]
                marker, ls = init_styles[init]
                ax.errorbar(
                    xs, ys, yerr=yerr,
                    marker=marker, linestyle=ls, linewidth=1.8, markersize=5,
                    color=mol_colors[mol_name],
                    label=f"init={init}",
                    alpha=0.85,
                )

            ax.set_yscale("symlog", linthresh=500)
            ax.set_xlabel(f"Fraction of {minority_strat} agents")
            ax.set_ylabel("Final energy (kcal/mol, symlog)")
            ax.set_title(f"{mol_name}  |  minority = {minority_strat}", fontsize=10)
            ax.set_xlim(-0.05, 1.05)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)

    fig.suptitle(
        "Heterogeneous populations: final energy vs fraction of minority strategy\n"
        "(majority = isolated, cutoff=4.0Å, T=300K)",
        fontsize=12,
    )
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "heterogeneous_energy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

    # ---- Figure 2: acceptance rate split (minority vs isolated neighbours) ----
    fig, axes = plt.subplots(
        1, len(MINORITY_STRATEGIES),
        figsize=(7 * len(MINORITY_STRATEGIES), 5),
        squeeze=False,
    )

    for col, minority_strat in enumerate(MINORITY_STRATEGIES):
        ax = axes[0][col]
        for mol_name, _ in MOLECULES:
            subset = sorted(
                [r for r in results
                 if r["molecule"] == mol_name
                 and r["minority_strategy"] == minority_strat
                 and r["init"] == "etkdg"],
                key=lambda r: r["n_minority"],
            )
            # filter: only mixed (not pure)
            subset = [r for r in subset
                      if r["n_minority"] > 0 and r["fraction_minority"] < 1.0
                      and not np.isnan(r["acc_minority_mean"])]
            if not subset:
                continue
            xs = [r["fraction_minority"] for r in subset]
            ax.plot(xs, [r["acc_minority_mean"] for r in subset],
                    "o-", color=mol_colors[mol_name], linewidth=1.8,
                    label=f"{mol_name} ({minority_strat})")
            ax.plot(xs, [r["acc_isolated_mean"] for r in subset],
                    "s--", color=mol_colors[mol_name], linewidth=1.2, alpha=0.55)

        ax.set_xlabel(f"Fraction of {minority_strat} agents")
        ax.set_ylabel("Mean acceptance rate")
        ax.set_title(
            f"Acceptance: {minority_strat} (solid) vs isolated neighbours (dashed)\n"
            f"init=etkdg, cutoff=4.0Å",
            fontsize=10,
        )
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "heterogeneous_acceptance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

    # ---- Figure 3: energy trajectories for fluoxetine zeros, consensus sweep ----
    mol_name, smiles = MOLECULES[2]  # fluoxetine
    minority_strat = "consensus"
    n_bonds = _n_bonds_for(smiles)

    fig, axes = plt.subplots(1, n_bonds + 1, figsize=(3.5 * (n_bonds + 1), 4), squeeze=False)
    for n_min in range(n_bonds + 1):
        ax = axes[0][n_min]
        m = models_dict.get((mol_name, minority_strat, n_min, "zeros"))
        if m is None:
            ax.set_visible(False)
            continue
        snaps = np.array(m.energy_snapshots, dtype=float)
        e_ref = snaps.min()
        e_rel = np.maximum(snaps - e_ref, 0.01)
        ax.plot(_smooth(e_rel), color=MINORITY_COLORS[minority_strat], linewidth=1.5)
        ax.set_yscale("log")
        ax.set_title(f"{n_min}/{n_bonds} consensus", fontsize=9)
        ax.set_xlabel("Step")
        if n_min == 0:
            ax.set_ylabel("E − E_min  (kcal/mol, log)")
        ax.grid(alpha=0.25, which="both")

    fig.suptitle(
        f"Fluoxetine zeros init: increasing consensus agents\n"
        f"(cutoff=4.0Å, T=300K, seed=0)",
        fontsize=11,
    )
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "heterogeneous_deadlock_sweep.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Atom-agent model: grid + comparison with bond-agent model
# ---------------------------------------------------------------------------


def run_atom_grid(n_seeds: int = 3) -> list:
    """
    Run AtomMoleculeModel on same grid as run_grid():
    5 strategies × 4 inits × cutoff=4.0 Å × 3 molecules × n_seeds.
    Returns list of result dicts.
    """
    from atom_model import AtomMoleculeModel

    results = []
    ATOM_CUTOFF = 5.0  # atom-atom distance ~equivalent to 4.0 Å bond-midpoint

    for mol_name, smiles in MOLECULES:
        print(f"\n[atom] {mol_name}")
        for strategy in STRATEGIES:
            for init in INITS:
                energies, coverages, acceptances = [], [], []
                for seed in range(n_seeds):
                    m = AtomMoleculeModel(
                        smiles,
                        strategy=strategy,
                        init=init,
                        comm_cutoff=ATOM_CUTOFF,
                        n_steps=N_STEPS,
                        temperature=TEMPERATURE,
                        seed=seed,
                    )
                    m.run()
                    s = m.summary()
                    energies.append(s["final_energy"])
                    coverages.append(s["mean_coverage"])
                    acceptances.append(s["mean_acceptance"])

                results.append({
                    "molecule": mol_name,
                    "n_bonds": m.summary()["n_bonds"],
                    "n_agents": m.summary()["n_agents"],
                    "strategy": strategy,
                    "init": init,
                    "model": "atom",
                    "final_energy_mean": float(np.mean(energies)),
                    "final_energy_std": float(np.std(energies)),
                    "mean_coverage": float(np.mean(coverages)),
                    "mean_acceptance": float(np.mean(acceptances)),
                })
                print(f"  {strategy:20s} {init:8s}  E={np.mean(energies):.1f}")

    return results


def plot_atom_comparison(bond_results: list, atom_results: list) -> None:
    """
    Side-by-side comparison: bond-agent vs atom-agent model.
    One figure per molecule: median final energy per strategy × model,
    for each init separately.
    """
    model_styles = {
        "bond": {"color": "#1f77b4", "hatch": ""},
        "atom": {"color": "#ff7f0e", "hatch": "//"},
    }

    # Tag bond results with model="bond"
    for r in bond_results:
        r.setdefault("model", "bond")

    all_results = bond_results + atom_results

    fig, axes = plt.subplots(
        len(INITS), len(MOLECULES),
        figsize=(5.5 * len(MOLECULES), 4 * len(INITS)),
        squeeze=False,
    )

    x = np.arange(len(STRATEGIES))
    width = 0.38

    for row, init in enumerate(INITS):
        for col, (mol_name, _) in enumerate(MOLECULES):
            ax = axes[row][col]
            for k, model_type in enumerate(["bond", "atom"]):
                ys, yerrs = [], []
                for strat in STRATEGIES:
                    pts = [
                        r["final_energy_mean"]
                        for r in all_results
                        if r["molecule"] == mol_name
                        and r["strategy"] == strat
                        and r["init"] == init
                        and r.get("model") == model_type
                    ]
                    ys.append(np.median(pts) if pts else np.nan)
                    yerrs.append(np.std(pts) if pts else 0.0)

                offset = (k - 0.5) * width
                style = model_styles[model_type]
                bars = ax.bar(
                    x + offset, ys, width,
                    label=f"{model_type}-agent",
                    color=style["color"],
                    hatch=style["hatch"],
                    alpha=0.82,
                )
                ax.errorbar(
                    x + offset, ys, yerr=yerrs,
                    fmt="none", color="black", capsize=3, linewidth=1,
                )

            ax.set_yscale("symlog", linthresh=200)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [s.replace("_", "\n") for s in STRATEGIES], fontsize=7
            )
            ax.set_title(f"{mol_name}  |  init={init}", fontsize=9)
            ax.set_ylabel("Final energy (kcal/mol)" if col == 0 else "")
            ax.grid(axis="y", alpha=0.25, which="both")
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    fig.suptitle(
        "Bond-agent vs Atom-agent model\n"
        "(median over 3 seeds, cutoff bond=4.0Å / atom=5.0Å, T=300K)",
        fontsize=12,
    )
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "atom_vs_bond.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

    # --- Convergence curves: bond vs atom, etkdg init, cutoff default ---
    fig, axes = plt.subplots(
        1, len(MOLECULES),
        figsize=(6 * len(MOLECULES), 4.5),
        squeeze=False,
    )

    from atom_model import AtomMoleculeModel

    for col, (mol_name, smiles) in enumerate(MOLECULES):
        ax = axes[0][col]
        for strategy in STRATEGIES:
            color = STRATEGY_COLORS[strategy]
            label = strategy.replace("_", " ")

            # Bond model convergence (from bond_results, need model stored)
            # Use atom model convergence directly
            m_atom = AtomMoleculeModel(
                smiles, strategy=strategy, init="etkdg",
                comm_cutoff=5.0, n_steps=N_STEPS, temperature=TEMPERATURE, seed=42,
            )
            m_atom.run()
            snaps_a = np.array(m_atom.energy_snapshots)
            e_best_a = snaps_a.min()
            e_rel_a = np.maximum(snaps_a - e_best_a, 0.01)
            ax.plot(_smooth(e_rel_a), color=color, linewidth=1.8,
                    linestyle="--", alpha=0.7, label=f"{label} (atom)")

        ax.set_yscale("log")
        ax.set_xlabel("Step")
        ax.set_ylabel("E − E_min  (kcal/mol, log)" if col == 0 else "")
        ax.set_title(f"{mol_name} — atom-agent convergence (etkdg)", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25, which="both")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "atom_convergence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Atom-specific strategy comparison
# ---------------------------------------------------------------------------

ATOM_NEW_STRATEGIES = ["best_first", "coordinated", "lookahead"]
ATOM_ALL_STRATEGIES = STRATEGIES + ATOM_NEW_STRATEGIES

# colours for 8 strategies
ATOM_STRATEGY_COLORS = {
    **STRATEGY_COLORS,
    "best_first":   "#e377c2",
    "coordinated":  "#bcbd22",
    "lookahead":    "#17becf",
}


def run_atom_all_strategies(n_seeds: int = 3) -> list:
    """
    Run all 8 atom strategies (5 original + 3 new) on all molecules,
    etkdg + zeros inits, cutoff=5.0Å, 1000 steps.
    """
    from atom_model import AtomMoleculeModel

    results = []
    for mol_name, smiles in MOLECULES:
        print(f"\n[atom-all] {mol_name}")
        for strategy in ATOM_ALL_STRATEGIES:
            for init in ["etkdg", "zeros", "anti"]:
                energies, coverages, acceptances = [], [], []
                for seed in range(n_seeds):
                    m = AtomMoleculeModel(
                        smiles, strategy=strategy, init=init,
                        comm_cutoff=5.0, n_steps=N_STEPS,
                        temperature=TEMPERATURE, seed=seed,
                    )
                    m.run()
                    s = m.summary()
                    energies.append(s["final_energy"])
                    coverages.append(s["mean_coverage"])
                    acceptances.append(s["mean_acceptance"])
                results.append({
                    "molecule": mol_name,
                    "strategy": strategy,
                    "init": init,
                    "is_new": strategy in ATOM_NEW_STRATEGIES,
                    "final_energy_mean": float(np.mean(energies)),
                    "final_energy_std":  float(np.std(energies)),
                    "mean_coverage":     float(np.mean(coverages)),
                    "mean_acceptance":   float(np.mean(acceptances)),
                })
                tag = "NEW" if strategy in ATOM_NEW_STRATEGIES else "   "
                print(f"  {tag} {strategy:20s} {init:8s}  "
                      f"E={np.mean(energies):.1f}  cov={np.mean(coverages):.2f}")
    return results


def plot_atom_strategy_analysis(atom_all: list, bond_results: list) -> None:
    """
    Figure 1: bar chart — all 8 atom strategies + bond-agent baseline,
              per molecule × init.  New strategies highlighted.

    Figure 2: convergence curves — best new vs best original atom strategy
              vs bond isolated (etkdg init, each molecule).

    Figure 3: why original strategies underperform — scatter:
              effective steps-per-DOF vs final energy.
    """
    from atom_model import AtomMoleculeModel
    from model import MoleculeModel

    # ---- Figure 1: bar chart ----------------------------------------
    inits_show = ["etkdg", "zeros", "anti"]
    fig, axes = plt.subplots(
        len(inits_show), len(MOLECULES),
        figsize=(5.5 * len(MOLECULES), 4.5 * len(inits_show)),
        squeeze=False,
    )
    x = np.arange(len(ATOM_ALL_STRATEGIES))
    width = 0.38

    for row, init in enumerate(inits_show):
        for col, (mol_name, _) in enumerate(MOLECULES):
            ax = axes[row][col]

            # atom bars
            ys_atom, yerrs_atom = [], []
            for strat in ATOM_ALL_STRATEGIES:
                pts = [r["final_energy_mean"] for r in atom_all
                       if r["molecule"] == mol_name and r["strategy"] == strat
                       and r["init"] == init]
                ys_atom.append(np.mean(pts) if pts else np.nan)
                yerrs_atom.append(np.std(pts) if pts else 0.)

            colors_bar = [
                "#e8774e" if s in ATOM_NEW_STRATEGIES else "#7bafd4"
                for s in ATOM_ALL_STRATEGIES
            ]
            bars = ax.bar(x - width/2, ys_atom, width,
                          color=colors_bar, alpha=0.85, label="atom-agent")
            ax.errorbar(x - width/2, ys_atom, yerr=yerrs_atom,
                        fmt="none", color="black", capsize=3, linewidth=1)

            # bond baseline bars (isolated only, for reference)
            for r in bond_results:
                r.setdefault("model", "bond")
            bond_pts = {
                strat: np.mean([
                    float(r.get("final_energy_mean", r.get("final_energy", np.nan)))
                    for r in bond_results
                    if r.get("molecule") == mol_name
                    and r.get("strategy") == strat
                    and r.get("init") == init
                    and r.get("comm_cutoff", "4.0") in ("4.0", 4.0)
                ])
                for strat in STRATEGIES
            }
            ys_bond = [bond_pts.get(s, np.nan) for s in ATOM_ALL_STRATEGIES]
            ax.bar(x + width/2, ys_bond, width,
                   color="#aaaaaa", alpha=0.55, label="bond-agent")

            ax.set_yscale("symlog", linthresh=300)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [s.replace("_", "\n") for s in ATOM_ALL_STRATEGIES], fontsize=6.5
            )
            ax.set_title(f"{mol_name}  |  init={init}", fontsize=9)
            ax.set_ylabel("Final energy (kcal/mol)" if col == 0 else "")
            ax.grid(axis="y", alpha=0.2, which="both")
            if row == 0 and col == 0:
                from matplotlib.patches import Patch
                ax.legend(handles=[
                    Patch(color="#7bafd4", label="atom (original)"),
                    Patch(color="#e8774e", label="atom (new)"),
                    Patch(color="#aaaaaa", alpha=0.6, label="bond baseline"),
                ], fontsize=7)

    fig.suptitle(
        "All atom strategies vs bond-agent baseline\n"
        "(orange = atom-specific new strategies, cutoff=5.0Å, T=300K)",
        fontsize=12,
    )
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "atom_strategies_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

    # ---- Figure 2: convergence curves (etkdg) -----------------------
    fig, axes = plt.subplots(1, len(MOLECULES),
                             figsize=(6 * len(MOLECULES), 4.5), squeeze=False)

    for col, (mol_name, smiles) in enumerate(MOLECULES):
        ax = axes[0][col]

        for strategy in ATOM_ALL_STRATEGIES:
            m = AtomMoleculeModel(smiles, strategy=strategy, init="etkdg",
                                  comm_cutoff=5.0, n_steps=N_STEPS,
                                  temperature=TEMPERATURE, seed=42)
            m.run()
            snaps = np.array(m.energy_snapshots)
            e_rel = np.maximum(snaps - snaps.min(), 0.01)
            color = ATOM_STRATEGY_COLORS[strategy]
            ls = "-" if strategy in ATOM_NEW_STRATEGIES else "--"
            lw = 2.2 if strategy in ATOM_NEW_STRATEGIES else 1.2
            ax.plot(_smooth(e_rel), color=color, linewidth=lw, linestyle=ls,
                    label=strategy.replace("_", " ") +
                          (" ★" if strategy in ATOM_NEW_STRATEGIES else ""))

        # bond isolated reference
        mb = MoleculeModel(smiles, strategy="isolated", init="etkdg",
                           comm_cutoff=4.0, n_steps=N_STEPS,
                           temperature=TEMPERATURE, seed=42)
        mb.run()
        snaps_b = np.array(mb.energy_snapshots)
        e_rel_b = np.maximum(snaps_b - snaps_b.min(), 0.01)
        ax.plot(_smooth(e_rel_b), color="black", linewidth=1.5,
                linestyle=":", label="bond isolated (ref)")

        ax.set_yscale("log")
        ax.set_xlabel("Step")
        ax.set_ylabel("E − E_min  (kcal/mol)" if col == 0 else "")
        ax.set_title(f"{mol_name}  |  etkdg init", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2, which="both")

    fig.suptitle(
        "Convergence: atom strategies (solid=new ★, dashed=original) "
        "vs bond isolated (dotted)\netkdg init, cutoff=5.0Å, T=300K",
        fontsize=11,
    )
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "atom_strategies_convergence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

    # ---- Figure 3: effective steps-per-DOF vs final energy ----------
    fig, axes = plt.subplots(1, len(MOLECULES),
                             figsize=(5 * len(MOLECULES), 4.5), squeeze=False)

    for col, (mol_name, _) in enumerate(MOLECULES):
        ax = axes[0][col]
        n_bonds_mol = next(
            len(AtomMoleculeModel(smiles, n_steps=0, seed=0).bonds)
            for mn, smiles in MOLECULES if mn == mol_name
        )
        for strategy in ATOM_ALL_STRATEGIES:
            pts = [r for r in atom_all
                   if r["molecule"] == mol_name and r["strategy"] == strategy
                   and r["init"] == "etkdg"]
            if not pts:
                continue
            # Effective steps per DOF: an atom owning k bonds does k actions
            # in coordinated, 1 action in others → normalise by n_bonds
            steps_per_dof = N_STEPS / n_bonds_mol
            if strategy == "coordinated":
                # coordinated tries all bonds each step
                pass  # already full coverage
            fe = np.mean([p["final_energy_mean"] for p in pts])
            acc = np.mean([p["mean_acceptance"] for p in pts])
            color = ATOM_STRATEGY_COLORS[strategy]
            marker = "★" if strategy in ATOM_NEW_STRATEGIES else "o"
            ms = 12 if strategy in ATOM_NEW_STRATEGIES else 7
            ax.scatter(acc, fe, color=color, s=ms**2,
                       zorder=5, label=strategy.replace("_", " "))
            ax.annotate(strategy.split("_")[0], (acc, fe),
                        fontsize=6.5, ha="left", va="bottom")

        ax.set_yscale("symlog", linthresh=200)
        ax.set_xlabel("Mean acceptance rate")
        ax.set_ylabel("Final energy (kcal/mol)" if col == 0 else "")
        ax.set_title(f"{mol_name}  |  etkdg init", fontsize=10)
        ax.grid(alpha=0.2)

    fig.suptitle(
        "Acceptance rate vs final energy — atom strategies (etkdg init)\n"
        "★ = atom-specific new strategies",
        fontsize=11,
    )
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "atom_acceptance_vs_energy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


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

    print("\nRunning heterogeneous strategy experiments …")
    het_results, het_models = run_heterogeneous_grid(n_seeds=3)
    plot_heterogeneous(het_results, het_models)

    print("\nRunning atom-agent model comparison …")
    atom_results = run_atom_grid(n_seeds=3)
    # tag bond results with model="bond" for comparison
    for r in results:
        r.setdefault("model", "bond")
    plot_atom_comparison(results, atom_results)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.1f} min.  Results written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
