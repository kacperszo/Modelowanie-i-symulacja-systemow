"""
pain_evo_large.py — Evolutionary pain model on large flexible molecules.

Tests whether the consensus-transplant advantage grows with molecular complexity.
Small molecules (3-7 bonds): consensus ≈ mutation (effects cancel).
Large molecules (12-16 bonds): hypothesis — per-bond adaptive explore/exploit
should pull ahead because identifying which DOFs are already solved is
increasingly valuable as dimensionality grows.

Molecules:
  salmeterol    — 16 rotatable bonds (long polyether chain + amine arm)
  atorvastatin  — 13 rotatable bonds (heptanoic acid chain + 4-ring scaffold)

Usage:
    uv run python src/pain_evo_large.py
"""

from __future__ import annotations

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from pain_model import PainModel, PainParams
from pain_evolution import (
    EvoPopulation, run_baseline, _circular_stats,
    STABILITY_DEG, SIGMA_UNSTABLE, SIGMA_MUT,
)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

BEST_PARAMS = PainParams(r_pain=5.06, pain_decay=0.62, step_size=2.91, vote_threshold=0.006)

MOLECULES = [
    ("salmeterol",   "CC(CCc1ccc(OCCOCCOCc2ccccc2)cc1)NCC(O)c1ccc(O)c(O)c1"),
    ("atorvastatin", "CC(C)c1c(C(=O)Nc2ccccc2F)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O"),
]

# Increased budget: more generations for higher-dimensional search space
POP_SIZE       = 20
N_PAIN_STEPS   = 15
N_GENERATIONS  = 60     # vs 40 for small molecules
N_MUT_BONDS    = 3      # mutate more bonds per offspring (was 2)
N_SINGLE_SEEDS = 8
N_SINGLE_STEPS = N_GENERATIONS * N_PAIN_STEPS   # = 900


def main() -> None:
    t0 = time.time()
    all_results = {}

    # Print bond counts
    print("Rotatable bond counts:")
    for mol_name, smiles in MOLECULES:
        m = PainModel(smiles, BEST_PARAMS, init="etkdg", n_steps=0, seed=42)
        print(f"  {mol_name}: {len(m.bonds)} bonds")

    print()

    for mol_name, smiles in MOLECULES:
        print(f"=== {mol_name} ===")

        print(f"  Baseline: {N_SINGLE_SEEDS} seeds × {N_SINGLE_STEPS} steps …")
        baseline = run_baseline(smiles, BEST_PARAMS, N_SINGLE_STEPS, N_SINGLE_SEEDS)
        print(f"  baseline E: min={min(baseline):.1f}  "
              f"mean={np.mean(baseline):.1f}  max={max(baseline):.1f}")

        print(f"  Evo (mutation) …")
        evo_mut = EvoPopulation(
            smiles, BEST_PARAMS,
            pop_size=POP_SIZE,
            n_pain_steps=N_PAIN_STEPS,
            n_mut_bonds=N_MUT_BONDS,
            sigma_mut=SIGMA_MUT,
            sigma_unstable=SIGMA_UNSTABLE,
            stability_deg=STABILITY_DEG,
            elite_frac=0.40,
            mode="mutation",
            seed=42,
        )
        evo_mut.run(N_GENERATIONS)
        bm = evo_mut.best_individual()
        print(f"    best E={bm.final_energy():.1f}")

        print(f"  Evo (consensus) …")
        evo_cons = EvoPopulation(
            smiles, BEST_PARAMS,
            pop_size=POP_SIZE,
            n_pain_steps=N_PAIN_STEPS,
            n_mut_bonds=N_MUT_BONDS,
            sigma_mut=SIGMA_MUT,
            sigma_unstable=SIGMA_UNSTABLE,
            stability_deg=STABILITY_DEG,
            elite_frac=0.40,
            mode="consensus",
            seed=42,
        )
        evo_cons.run(N_GENERATIONS)
        bc = evo_cons.best_individual()
        print(f"    best E={bc.final_energy():.1f}")

        if evo_cons.bond_stable_history:
            final_stable = evo_cons.bond_stable_history[-1]
            n_stable = int(final_stable.sum())
            n_total  = len(final_stable)
            print(f"    bonds stable at end: {n_stable}/{n_total}")
            first_stable_gen = []
            for b in range(n_total):
                col = [row[b] for row in evo_cons.bond_stable_history]
                first = next((g for g, v in enumerate(col) if v), None)
                first_stable_gen.append(first)
            print(f"    first stable gen per bond: {first_stable_gen}")

        all_results[mol_name] = {
            "baseline":  baseline,
            "evo_mut":   evo_mut,
            "evo_cons":  evo_cons,
        }

    print("\n=== Plotting ===")
    _plot(all_results)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min.")


def _plot(all_results: dict) -> None:
    n_mol = len(MOLECULES)
    gens  = range(1, N_GENERATIONS + 1)

    # ---- Fig 1: convergence ----
    fig, axes = plt.subplots(1, n_mol, figsize=(7 * n_mol, 5), squeeze=False)
    fig.suptitle(
        "Large molecules — mutation vs consensus-transplant evolution\n"
        f"(pop={POP_SIZE}, {N_GENERATIONS} gen × {N_PAIN_STEPS} pain steps, "
        f"n_mut_bonds={N_MUT_BONDS})",
        fontsize=11,
    )
    for col, (mol_name, _) in enumerate(MOLECULES):
        ax  = axes[0][col]
        res = all_results[mol_name]
        evm = res["evo_mut"]
        evc = res["evo_cons"]
        baseline = res["baseline"]

        ax.plot(gens, evm.best_energy_history,
                color="#1155aa", lw=2.2, label="mutation best")
        ax.plot(gens, evm.mean_energy_history,
                color="#1155aa", lw=1.0, ls=":", alpha=0.6, label="mutation mean")
        ax.plot(gens, evc.best_energy_history,
                color="#118844", lw=2.2, label="consensus best")
        ax.plot(gens, evc.mean_energy_history,
                color="#118844", lw=1.0, ls=":", alpha=0.6, label="consensus mean")

        b_mean = float(np.mean(baseline))
        b_min  = float(np.min(baseline))
        b_max  = float(np.max(baseline))
        ax.axhline(b_mean, color="#cc4422", lw=1.3, ls="--",
                   label=f"single mean ({b_mean:.0f})")
        ax.axhspan(b_min, b_max, alpha=0.10, color="#cc4422", label="single range")

        y_top = min(
            max(evm.mean_energy_history + evc.mean_energy_history),
            b_max * 2.0,
        )
        y_bot = min(evm.best_energy_history + evc.best_energy_history) * 0.95
        ax.set_ylim(y_bot, y_top)

        m = PainModel(MOLECULES[col][1], BEST_PARAMS, init="etkdg", n_steps=0, seed=42)
        ax.set_title(f"{mol_name}  ({len(m.bonds)} bonds)", fontsize=10)
        ax.set_xlabel("generation")
        ax.set_ylabel("MMFF94 energy (kcal/mol)" if col == 0 else "")
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, "evo_large_convergence.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")

    # ---- Fig 2: final energy boxplot ----
    fig2, axes2 = plt.subplots(1, n_mol, figsize=(6 * n_mol, 4.5), squeeze=False)
    fig2.suptitle("Large molecules — final energy distribution", fontsize=11)
    for col, (mol_name, _) in enumerate(MOLECULES):
        ax  = axes2[0][col]
        res = all_results[mol_name]
        pop_mut  = [m.final_energy() for m in res["evo_mut"].pop]
        pop_cons = [m.final_energy() for m in res["evo_cons"].pop]
        baseline = res["baseline"]

        bp = ax.boxplot(
            [baseline, pop_mut, pop_cons],
            tick_labels=["single\n(8 seeds)",
                         f"mutation\n(pop={POP_SIZE})",
                         f"consensus\n(pop={POP_SIZE})"],
            patch_artist=True,
            medianprops=dict(color="black", lw=2),
            showfliers=False,
        )
        bp["boxes"][0].set_facecolor("#ffbbaa")
        bp["boxes"][1].set_facecolor("#aabbff")
        bp["boxes"][2].set_facecolor("#aaddbb")

        bs = min(baseline);   bm = min(pop_mut);  bc = min(pop_cons)
        ax.scatter([1], [bs], color="#cc3300", s=70, zorder=5,
                   label=f"best single:    {bs:.1f}")
        ax.scatter([2], [bm], color="#0033cc", s=70, zorder=5,
                   label=f"best mutation:  {bm:.1f}")
        ax.scatter([3], [bc], color="#006633", s=70, zorder=5,
                   label=f"best consensus: {bc:.1f}")

        gap_pct = (bm - bc) / bm * 100 if bm > 0 else 0
        ax.set_title(f"{mol_name}\nconsensus vs mutation: {gap_pct:+.1f}%",
                     fontsize=10)
        ax.set_ylabel("MMFF94 energy (kcal/mol)" if col == 0 else "")
        ax.legend(fontsize=7.5)
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, "evo_large_comparison.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")

    # ---- Fig 3: bond stabilization heatmap (consensus only) ----
    fig3, axes3 = plt.subplots(1, n_mol, figsize=(7 * n_mol, 4.5), squeeze=False)
    fig3.suptitle(
        f"Bond stabilization — large molecules (consensus mode, threshold={STABILITY_DEG}°)\n"
        "color = circular std across elite; white squares = stable",
        fontsize=11,
    )
    for col, (mol_name, _) in enumerate(MOLECULES):
        ax  = axes3[0][col]
        evc = all_results[mol_name]["evo_cons"]
        if not evc.bond_std_history:
            ax.set_visible(False)
            continue
        std_arr    = np.array(evc.bond_std_history).T   # (n_bonds, n_gen)
        stable_arr = np.array(evc.bond_stable_history).T
        im = ax.imshow(std_arr, aspect="auto", origin="lower",
                       cmap="RdYlGn_r", vmin=0, vmax=90)
        ys, xs = np.where(stable_arr)
        ax.scatter(xs, ys, marker="s", s=10, color="white",
                   edgecolors="black", lw=0.3, alpha=0.7)
        ax.set_xlabel("generation")
        ax.set_ylabel("bond index" if col == 0 else "")
        ax.set_yticks(range(std_arr.shape[0]))
        m = PainModel(MOLECULES[col][1], BEST_PARAMS, init="etkdg", n_steps=0, seed=42)
        ax.set_title(f"{mol_name}  ({len(m.bonds)} bonds)", fontsize=10)
        plt.colorbar(im, ax=ax, label="circular std (°)")

    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, "evo_large_stabilization.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")

    # ---- Fig 4: gap consensus-vs-mutation vs n_bonds (cross-molecule summary) ----
    small_data = [
        # (mol_name, n_bonds, best_single, best_mut, best_cons)  from earlier run
        ("aspirin",    3,  43.5, 40.0, 38.5),
        ("lidocaine",  6, 105.9, 95.0, 95.1),
        ("fluoxetine", 7,  98.6, 94.3, 93.5),
    ]
    large_data = []
    for col, (mol_name, smiles) in enumerate(MOLECULES):
        res = all_results[mol_name]
        m = PainModel(smiles, BEST_PARAMS, init="etkdg", n_steps=0, seed=42)
        n_b = len(m.bonds)
        bs  = float(np.min(res["baseline"]))
        bm  = float(np.min([x.final_energy() for x in res["evo_mut"].pop]))
        bc  = float(np.min([x.final_energy() for x in res["evo_cons"].pop]))
        large_data.append((mol_name, n_b, bs, bm, bc))

    all_data = small_data + large_data
    n_bonds_list  = [d[1] for d in all_data]
    gap_cons_mut  = [(d[3] - d[4]) / d[3] * 100 for d in all_data]  # % by which cons beats mut
    gap_mut_single = [(d[2] - d[3]) / d[2] * 100 for d in all_data]  # % by which mut beats single

    fig4, ax4 = plt.subplots(figsize=(7, 4.5))
    ax4.scatter(n_bonds_list[:3], gap_cons_mut[:3],
                color="#118844", s=90, label="small molecules (Stage V)")
    ax4.scatter([d[1] for d in large_data], [gap_cons_mut[3+i] for i in range(len(large_data))],
                color="#118844", s=130, marker="*", label="large molecules (this run)")
    ax4.scatter(n_bonds_list[:3], gap_mut_single[:3],
                color="#1155aa", s=90, label="mutation vs single — small")
    ax4.scatter([d[1] for d in large_data], [gap_mut_single[3+i] for i in range(len(large_data))],
                color="#1155aa", s=130, marker="*", label="mutation vs single — large")

    for d, gc, gm in zip(all_data, gap_cons_mut, gap_mut_single):
        ax4.annotate(d[0], (d[1], gc), textcoords="offset points",
                     xytext=(4, 3), fontsize=7.5, color="#118844")
        ax4.annotate(d[0], (d[1], gm), textcoords="offset points",
                     xytext=(4, -10), fontsize=7.5, color="#1155aa")

    ax4.axhline(0, color="grey", lw=0.8, ls="--")
    ax4.set_xlabel("number of rotatable bonds")
    ax4.set_ylabel("energy improvement (%)\npositive = lower energy is better")
    ax4.set_title("Does the consensus advantage grow with molecule complexity?")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.25)
    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, "evo_large_scaling.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")


if __name__ == "__main__":
    main()
