"""
optimize_pain.py — Random search + 2D grid search over PainModel parameters.

Fitness: mean final MMFF94 energy over all molecules × inits × seeds.

Usage:
    uv run python src/optimize_pain.py
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from pain_model import PARAM_BOUNDS, PARAM_NAMES, PainModel, PainParams

# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------

MOLECULES = [
    ("aspirin",    "CC(=O)Oc1ccccc1C(=O)O"),
    ("lidocaine",  "CCN(CC)CC(=O)Nc1c(C)cccc1C"),
    ("fluoxetine", "CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1"),
]
INITS = ["etkdg", "zeros"]
N_STEPS_OPT = 300      # steps during optimisation (fast)
N_STEPS_FULL = 1000    # steps for final analysis
N_SEEDS = 2
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Fitness function
# ---------------------------------------------------------------------------

def evaluate_params(
    params: PainParams,
    n_steps: int = N_STEPS_OPT,
    n_seeds: int = N_SEEDS,
) -> float:
    """Mean final MMFF94 energy over all molecules × inits × seeds."""
    energies = []
    for _, smiles in MOLECULES:
        for init in INITS:
            for seed in range(n_seeds):
                try:
                    m = PainModel(smiles, params, init=init,
                                  n_steps=n_steps, seed=seed)
                    m.run()
                    energies.append(m.final_energy())
                except Exception:
                    energies.append(1e9)
    return float(np.mean(energies))


# ---------------------------------------------------------------------------
# Random search
# ---------------------------------------------------------------------------

def random_search(n_samples: int = 100, seed: int = 0) -> List[dict]:
    rng = np.random.default_rng(seed)
    results = []
    t0 = time.time()

    for k in range(n_samples):
        arr = np.array([
            rng.uniform(*PARAM_BOUNDS[name]) for name in PARAM_NAMES
        ])
        params = PainParams.from_array(arr)
        fitness = evaluate_params(params)
        results.append({"params": params, "arr": arr.copy(), "fitness": fitness})

        if (k + 1) % 10 == 0:
            best = min(r["fitness"] for r in results)
            elapsed = time.time() - t0
            print(f"  [{k+1:3d}/{n_samples}]  best_so_far={best:.1f}  "
                  f"elapsed={elapsed:.0f}s  last={fitness:.1f}  {params}")

    results.sort(key=lambda r: r["fitness"])
    return results


# ---------------------------------------------------------------------------
# 2-D grid search around best params
# ---------------------------------------------------------------------------

def grid_search_2d(
    best_arr: np.ndarray,
    p1_idx: int,
    p2_idx: int,
    n_grid: int = 12,
    half_range: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vary parameters p1 and p2 on a grid centred on best_arr.
    Returns (p1_vals, p2_vals, fitness_matrix).
    """
    p1_name, p2_name = PARAM_NAMES[p1_idx], PARAM_NAMES[p2_idx]
    lo1, hi1 = PARAM_BOUNDS[p1_name]
    lo2, hi2 = PARAM_BOUNDS[p2_name]

    p1_c, p2_c = best_arr[p1_idx], best_arr[p2_idx]
    p1_span = (hi1 - lo1) * half_range
    p2_span = (hi2 - lo2) * half_range

    p1_vals = np.linspace(
        max(lo1, p1_c - p1_span), min(hi1, p1_c + p1_span), n_grid
    )
    p2_vals = np.linspace(
        max(lo2, p2_c - p2_span), min(hi2, p2_c + p2_span), n_grid
    )

    fitness = np.full((n_grid, n_grid), np.nan)
    total = n_grid * n_grid
    done = 0
    for ii, v1 in enumerate(p1_vals):
        for jj, v2 in enumerate(p2_vals):
            arr = best_arr.copy()
            arr[p1_idx] = v1
            arr[p2_idx] = v2
            fitness[ii, jj] = evaluate_params(PainParams.from_array(arr))
            done += 1
            if done % 20 == 0:
                print(f"  grid {done}/{total}  best={np.nanmin(fitness):.1f}")

    return p1_vals, p2_vals, fitness


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def sensitivity_analysis(results: List[dict]) -> Dict[str, float]:
    """
    Spearman correlation between each parameter and fitness.
    Higher |corr| → more important parameter.
    """
    from scipy.stats import spearmanr

    arrs = np.array([r["arr"] for r in results])
    fits = np.array([r["fitness"] for r in results])
    sensitivities = {}
    for k, name in enumerate(PARAM_NAMES):
        corr, _ = spearmanr(arrs[:, k], fits)
        sensitivities[name] = float(corr)
    return sensitivities


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    rs_results: List[dict],
    grid_data: dict,
    best_params: PainParams,
) -> None:

    # ---- Figure 1: parameter importance (scatter per param vs fitness) ---
    fig, axes = plt.subplots(1, len(PARAM_NAMES),
                             figsize=(4.5 * len(PARAM_NAMES), 4.5))
    fits = np.array([r["fitness"] for r in rs_results])
    fits_clipped = np.clip(fits, 0, np.percentile(fits, 95))

    for k, (ax, name) in enumerate(zip(axes, PARAM_NAMES)):
        vals = np.array([r["arr"][k] for r in rs_results])
        sc = ax.scatter(vals, fits_clipped, c=fits_clipped,
                        cmap="RdYlGn_r", s=20, alpha=0.7)
        ax.axvline(best_params.to_array()[k], color="black",
                   linestyle="--", linewidth=1.5, label="best")
        ax.set_xlabel(name)
        ax.set_ylabel("fitness (mean energy)" if k == 0 else "")
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        plt.colorbar(sc, ax=ax)

    fig.suptitle("Random search: parameter values vs fitness\n"
                 "(clipped at 95th percentile)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "pain_param_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

    # ---- Figure 2: 2D fitness landscape (grid search) ----------------
    n_grids = len(grid_data)
    fig, axes = plt.subplots(1, n_grids, figsize=(6 * n_grids, 5.5), squeeze=False)

    for col, (key, (p1v, p2v, fit_mat)) in enumerate(grid_data.items()):
        ax = axes[0][col]
        p1_name, p2_name = key
        im = ax.imshow(
            fit_mat.T,
            origin="lower",
            extent=[p1v[0], p1v[-1], p2v[0], p2v[-1]],
            aspect="auto",
            cmap="RdYlGn_r",
            vmin=np.nanpercentile(fit_mat, 5),
            vmax=np.nanpercentile(fit_mat, 95),
        )
        # Mark best point in grid
        best_idx = np.unravel_index(np.nanargmin(fit_mat), fit_mat.shape)
        ax.scatter(p1v[best_idx[0]], p2v[best_idx[1]],
                   color="blue", s=80, zorder=5, label="grid best")
        ax.scatter(best_params.to_array()[PARAM_NAMES.index(p1_name)],
                   best_params.to_array()[PARAM_NAMES.index(p2_name)],
                   color="white", marker="*", s=150, zorder=6, label="RS best")
        ax.set_xlabel(p1_name)
        ax.set_ylabel(p2_name)
        ax.set_title(f"Fitness landscape: {p1_name} × {p2_name}", fontsize=10)
        ax.legend(fontsize=8)
        plt.colorbar(im, ax=ax, label="mean final energy")

    fig.suptitle("2D grid search around random-search optimum", fontsize=12)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "pain_landscape.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

    # ---- Figure 3: convergence — best vs worst vs median params ------
    fig, axes = plt.subplots(1, len(MOLECULES),
                             figsize=(6 * len(MOLECULES), 4.5), squeeze=False)

    groups = {
        "best params":   (rs_results[0]["params"],  "#2ca02c", "-",  2.5),
        "median params": (rs_results[len(rs_results)//2]["params"], "#ff7f0e", "--", 1.5),
        "worst params":  (rs_results[-1]["params"], "#d62728", ":",  1.5),
    }

    for col, (mol_name, smiles) in enumerate(MOLECULES):
        ax = axes[0][col]
        for label, (params, color, ls, lw) in groups.items():
            m = PainModel(smiles, params, init="etkdg",
                          n_steps=N_STEPS_FULL, seed=42)
            m.run()
            snaps = np.array(m.energy_snapshots)
            ax.plot(snaps, color=color, linestyle=ls, linewidth=lw, label=label)

        ax.set_xlabel("Step")
        ax.set_ylabel("MMFF94 energy (kcal/mol)" if col == 0 else "")
        ax.set_title(mol_name, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle("Pain model convergence: best / median / worst parameter sets\n"
                 "(etkdg init, 1000 steps)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "pain_convergence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

    # ---- Figure 4: pain vs energy correlation over time (best params) ----
    fig, axes = plt.subplots(1, len(MOLECULES),
                             figsize=(5.5 * len(MOLECULES), 4.5), squeeze=False)

    for col, (mol_name, smiles) in enumerate(MOLECULES):
        ax = axes[0][col]
        m = PainModel(smiles, best_params, init="etkdg",
                      n_steps=N_STEPS_FULL, seed=42)
        m.run()
        steps = np.arange(len(m.energy_snapshots))
        e = np.array(m.energy_snapshots)
        p = np.array(m.pain_snapshots)

        ax2 = ax.twinx()
        ax.plot(steps, e, color="#1f77b4", linewidth=1.8, label="MMFF94 energy")
        ax2.plot(steps, p, color="#d62728", linewidth=1.5,
                 linestyle="--", label="total pain")
        ax.set_xlabel("Step")
        ax.set_ylabel("Energy (kcal/mol)", color="#1f77b4")
        ax2.set_ylabel("Pain signal (a.u.)", color="#d62728")
        ax.set_title(f"{mol_name}  |  rotations={m.n_rotations}", fontsize=10)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
        ax.grid(alpha=0.2)

    fig.suptitle("Pain signal vs MMFF94 energy over time\n"
                 "(best params, etkdg init)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "pain_vs_energy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()

    print("=== Random search (100 samples) ===")
    rs_results = random_search(n_samples=100, seed=7)

    best = rs_results[0]
    print(f"\nBest: fitness={best['fitness']:.2f}  {best['params']}")
    print(f"Top 5:")
    for r in rs_results[:5]:
        print(f"  {r['fitness']:.2f}  {r['params']}")

    best_arr = best["arr"]

    # Sensitivity
    try:
        sens = sensitivity_analysis(rs_results)
        print("\nSpearman correlation (param → fitness):")
        for name, corr in sorted(sens.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"  {name:20s}: {corr:+.3f}")
        # Pick 2 most important for grid search
        top2 = sorted(sens.items(), key=lambda x: abs(x[1]), reverse=True)[:2]
        p1_name, p2_name = top2[0][0], top2[1][0]
    except ImportError:
        print("scipy not available, using r_pain × pain_decay for grid")
        p1_name, p2_name = "r_pain", "pain_decay"

    p1_idx = PARAM_NAMES.index(p1_name)
    p2_idx = PARAM_NAMES.index(p2_name)

    print(f"\n=== 2D grid search: {p1_name} × {p2_name} ===")
    p1v, p2v, fit_mat = grid_search_2d(best_arr, p1_idx, p2_idx, n_grid=12)

    grid_data = {(p1_name, p2_name): (p1v, p2v, fit_mat)}

    print("\n=== Plotting ===")
    plot_results(rs_results, grid_data, best["params"])

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min.")


if __name__ == "__main__":
    main()
