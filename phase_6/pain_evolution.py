"""
pain_evolution.py — Evolutionary escape from local pain-minima.

Architecture (two-level ABM):
  Level 1 — within individual: PainModel (atoms emit pain → rotors respond)
  Level 2 — population:        EvoPopulation (select best → mutate → replace)

Each generation:
  1. Run N_PAIN_STEPS pain steps for every individual.
  2. Rank by total_pain (the pain model's own objective).
  3. Keep elite_frac of population unchanged.
  4. Replace the rest with mutated copies of randomly chosen elite parents.
     Mutation = perturb N_MUT_BONDS random dihedrals by N(0, sigma_mut).

Comparison: single PainModel (baseline) vs EvoPopulation on same budget.

Usage:
    uv run python src/pain_evolution.py
"""

from __future__ import annotations

import os
import sys
import time
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from pain_model import PainModel, PainParams

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MOLECULES = [
    ("aspirin",    "CC(=O)Oc1ccccc1C(=O)O"),
    ("lidocaine",  "CCN(CC)CC(=O)Nc1c(C)cccc1C"),
    ("fluoxetine", "CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1"),
]

BEST_PARAMS = PainParams(r_pain=5.06, pain_decay=0.62, step_size=2.91, vote_threshold=0.006)

# Evolution hyperparameters
POP_SIZE      = 20
N_PAIN_STEPS  = 15    # pain steps per generation
N_GENERATIONS = 40
N_MUT_BONDS   = 2     # dihedrals perturbed per offspring
SIGMA_MUT     = 30.0  # degrees — for mutation-only variant
ELITE_FRAC    = 0.40  # fraction of population kept each generation

# Consensus-transplant parameters
STABILITY_DEG    = 25.0   # circular std (deg) below which a bond is "stable"
SIGMA_UNSTABLE   = 60.0   # bigger mutation for bonds the elite hasn't agreed on

# Baseline: single PainModel with this many steps (comparable compute)
N_SINGLE_STEPS = N_PAIN_STEPS * N_GENERATIONS   # = 600
N_SINGLE_SEEDS = 8


# ---------------------------------------------------------------------------
# Evolutionary population
# ---------------------------------------------------------------------------

def _circular_stats(angles_deg: np.ndarray) -> tuple[float, float]:
    """
    Circular mean and circular std of an array of angles (degrees).
    Returns (mean_deg, std_deg).
    """
    rad = np.deg2rad(angles_deg)
    mean_sin = float(np.mean(np.sin(rad)))
    mean_cos = float(np.mean(np.cos(rad)))
    mean_deg = float(np.rad2deg(np.arctan2(mean_sin, mean_cos)))
    R = float(np.sqrt(mean_sin**2 + mean_cos**2))
    if R < 1e-6:
        std_deg = 360.0
    else:
        std_deg = float(np.rad2deg(np.sqrt(-2.0 * np.log(min(R, 1.0)))))
    return mean_deg, std_deg


class EvoPopulation:
    """
    Population of PainModel conformers evolving via (μ + λ) selection.

    Fitness = MMFF94 energy (lower = better).

    Two offspring strategies (mode):
      "mutation"  — each child = parent with N random dihedrals perturbed
      "consensus" — for each bond, compute circular mean/std across the elite.
                    Stable bonds (std < STABILITY_DEG) are transplanted into
                    every offspring; unstable bonds get a larger mutation.
                    This is per-bond crossover with adaptive exploration.
    """

    def __init__(
        self,
        smiles: str,
        params: PainParams,
        pop_size: int = POP_SIZE,
        n_pain_steps: int = N_PAIN_STEPS,
        n_mut_bonds: int = N_MUT_BONDS,
        sigma_mut: float = SIGMA_MUT,
        sigma_unstable: float = SIGMA_UNSTABLE,
        stability_deg: float = STABILITY_DEG,
        elite_frac: float = ELITE_FRAC,
        mode: str = "mutation",
        seed: int = 42,
    ) -> None:
        assert mode in ("mutation", "consensus")
        self.rng = np.random.default_rng(seed)
        self.pop_size      = pop_size
        self.n_pain_steps  = n_pain_steps
        self.n_mut_bonds   = n_mut_bonds
        self.sigma_mut     = sigma_mut
        self.sigma_unstable = sigma_unstable
        self.stability_deg = stability_deg
        self.n_elite       = max(2, int(pop_size * elite_frac))
        self.mode          = mode

        # Initialise with diverse conformers: half zeros, half random
        self.pop: List[PainModel] = []
        for k in range(pop_size):
            s = int(self.rng.integers(0, 2**31))
            init = "zeros" if k < pop_size // 2 else "random"
            self.pop.append(PainModel(smiles, params, init=init, n_steps=0, seed=s))

        self.n_bonds = len(self.pop[0].bonds)
        self.generation = 0
        self.best_energy_history:  List[float] = []
        self.mean_energy_history:  List[float] = []
        self.worst_energy_history: List[float] = []
        self.best_pain_history:    List[float] = []
        self.diversity_history:    List[float] = []
        # consensus mode only: (n_gen, n_bonds) arrays
        self.bond_std_history:    List[np.ndarray] = []
        self.bond_stable_history: List[np.ndarray] = []

    # ------------------------------------------------------------------

    def _run_pain(self, ind: PainModel) -> None:
        for _ in range(self.n_pain_steps):
            ind.step()

    def _elite_consensus(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-bond circular mean, std, and stability mask from the elite."""
        elite = self.pop[: self.n_elite]
        cons = np.zeros(self.n_bonds)
        stds = np.zeros(self.n_bonds)
        for b in range(self.n_bonds):
            angles = np.array([m.bonds[b].current_angle for m in elite])
            cons[b], stds[b] = _circular_stats(angles)
        is_stable = stds < self.stability_deg
        return cons, stds, is_stable

    def step(self) -> None:
        """One evolutionary generation."""
        # 1. Pain dynamics for every individual
        for ind in self.pop:
            self._run_pain(ind)

        # 2. Sort by MMFF94 energy
        self.pop.sort(key=lambda m: m.final_energy())

        # 3. Record stats
        energies = [m.final_energy() for m in self.pop]
        self.best_energy_history.append(float(np.min(energies)))
        self.mean_energy_history.append(float(np.mean(energies)))
        self.worst_energy_history.append(float(np.max(energies)))
        self.diversity_history.append(float(np.std(energies)))
        self.best_pain_history.append(float(self.pop[0].final_pain()))

        # 4. Offspring generation
        new_pop = list(self.pop[: self.n_elite])
        n_offspring = self.pop_size - self.n_elite

        if self.mode == "consensus":
            cons, stds, is_stable = self._elite_consensus()
            self.bond_std_history.append(stds.copy())
            self.bond_stable_history.append(is_stable.copy())
            for _ in range(n_offspring):
                parent_idx = int(self.rng.integers(0, self.n_elite))
                parent     = self.pop[parent_idx]
                parent_e   = parent.final_energy()
                for _attempt in range(5):
                    child_seed = int(self.rng.integers(0, 2**31))
                    child = parent.fork_consensus(
                        consensus_angles=cons,
                        is_stable=is_stable,
                        sigma_unstable=self.sigma_unstable,
                        seed=child_seed,
                    )
                    if child.final_energy() <= parent_e * 3.0:
                        break
                new_pop.append(child)
        else:   # mutation mode
            for _ in range(n_offspring):
                parent_idx = int(self.rng.integers(0, self.n_elite))
                parent     = self.pop[parent_idx]
                parent_e   = parent.final_energy()
                for _attempt in range(5):
                    child_seed = int(self.rng.integers(0, 2**31))
                    child = parent.fork(
                        perturb_bonds=self.n_mut_bonds,
                        sigma=self.sigma_mut,
                        seed=child_seed,
                    )
                    if child.final_energy() <= parent_e * 3.0:
                        break
                new_pop.append(child)

        self.pop = new_pop
        self.generation += 1

    def run(self, n_generations: int = N_GENERATIONS) -> None:
        for g in range(n_generations):
            self.step()
            if (g + 1) % 10 == 0:
                best_e = self.best_energy_history[-1]
                mean_e = self.mean_energy_history[-1]
                div    = self.diversity_history[-1]
                print(f"    gen {g+1:3d}  best_E={best_e:.1f}  "
                      f"mean_E={mean_e:.1f}  std={div:.1f}")

    def best_individual(self) -> PainModel:
        return min(self.pop, key=lambda m: m.final_energy())


# ---------------------------------------------------------------------------
# Baseline: plain PainModel over many seeds
# ---------------------------------------------------------------------------

def run_baseline(smiles: str, params: PainParams, n_steps: int, n_seeds: int):
    """Run single PainModel from zeros init across multiple seeds."""
    results = []
    for s in range(n_seeds):
        m = PainModel(smiles, params, init="zeros", n_steps=n_steps, seed=s)
        m.run()
        results.append(m.final_energy())
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(all_results: dict) -> None:
    """Three figures: convergence, final-energy box comparison, bond stabilization."""
    n_mol = len(MOLECULES)

    # ---- Figure 1: convergence — best lines for both evo modes ----
    fig1, axes = plt.subplots(1, n_mol, figsize=(6 * n_mol, 5), squeeze=False)
    fig1.suptitle(
        "Evolutionary pain model: mutation-only vs consensus-transplant\n"
        f"(pop={POP_SIZE}, elite={int(ELITE_FRAC*100)}%, "
        f"{N_GENERATIONS} generations × {N_PAIN_STEPS} pain steps)",
        fontsize=11,
    )

    for col, (mol_name, _) in enumerate(MOLECULES):
        ax  = axes[0][col]
        res = all_results[mol_name]
        evm = res["evo_mut"]
        evc = res["evo_cons"]
        gens = range(1, N_GENERATIONS + 1)

        ax.plot(gens, evm.best_energy_history,
                color="#1155aa", linewidth=2.2, label="mutation best")
        ax.plot(gens, evm.mean_energy_history,
                color="#1155aa", linewidth=1.0, linestyle=":", alpha=0.7,
                label="mutation mean")
        ax.plot(gens, evc.best_energy_history,
                color="#118844", linewidth=2.2, label="consensus best")
        ax.plot(gens, evc.mean_energy_history,
                color="#118844", linewidth=1.0, linestyle=":", alpha=0.7,
                label="consensus mean")

        baseline = res["baseline"]
        b_mean = float(np.mean(baseline))
        b_min  = float(np.min(baseline))
        b_max  = float(np.max(baseline))
        ax.axhline(b_mean, color="#cc4422", linewidth=1.3, linestyle="--",
                   label=f"single mean ({b_mean:.0f})")
        ax.axhspan(b_min, b_max, alpha=0.10, color="#cc4422",
                   label="single range")

        # zoom y-axis to interesting region (clip at 2× baseline max so spikes
        # from bad mutants don't compress the meaningful signal)
        y_top = min(
            max(evm.mean_energy_history + evc.mean_energy_history),
            b_max * 2.0,
        )
        y_bot = min(evm.best_energy_history + evc.best_energy_history) * 0.95
        ax.set_ylim(y_bot, y_top)

        ax.set_xlabel("generation")
        ax.set_ylabel("MMFF94 energy (kcal/mol)" if col == 0 else "")
        ax.set_title(mol_name, fontsize=10)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(alpha=0.25)

    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, "evo_convergence.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")

    # ---- Figure 2: final energy box comparison (3 groups) ----
    fig2, axes2 = plt.subplots(1, n_mol, figsize=(5.5 * n_mol, 4.5), squeeze=False)
    fig2.suptitle("Final energy: single runs vs mutation evo vs consensus evo",
                  fontsize=11)

    for col, (mol_name, _) in enumerate(MOLECULES):
        ax  = axes2[0][col]
        res = all_results[mol_name]
        baseline = res["baseline"]
        pop_mut  = [m.final_energy() for m in res["evo_mut"].pop]
        pop_cons = [m.final_energy() for m in res["evo_cons"].pop]

        bp = ax.boxplot(
            [baseline, pop_mut, pop_cons],
            tick_labels=["single\n(8 seeds)",
                         f"mutation\n(pop={POP_SIZE})",
                         f"consensus\n(pop={POP_SIZE})"],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            showfliers=False,
        )
        bp["boxes"][0].set_facecolor("#ffbbaa")
        bp["boxes"][1].set_facecolor("#aabbff")
        bp["boxes"][2].set_facecolor("#aaddbb")

        bs = min(baseline)
        bm = min(pop_mut)
        bc = min(pop_cons)
        ax.scatter([1], [bs], color="#cc3300", s=60, zorder=5,
                   label=f"best single:    {bs:.1f}")
        ax.scatter([2], [bm], color="#0033cc", s=60, zorder=5,
                   label=f"best mutation:  {bm:.1f}")
        ax.scatter([3], [bc], color="#006633", s=60, zorder=5,
                   label=f"best consensus: {bc:.1f}")
        ax.legend(fontsize=7.5)

        ax.set_ylabel("MMFF94 energy (kcal/mol)" if col == 0 else "")
        ax.set_title(mol_name, fontsize=10)
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, "evo_final_comparison.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")

    # ---- Figure 3: bond stabilization heatmap (consensus mode only) ----
    fig3, axes3 = plt.subplots(1, n_mol, figsize=(6 * n_mol, 4), squeeze=False)
    fig3.suptitle(
        f"Bond stabilization timeline (consensus mode, threshold = {STABILITY_DEG}°)\n"
        "color = circular std of bond angle across elite",
        fontsize=11,
    )

    for col, (mol_name, _) in enumerate(MOLECULES):
        ax  = axes3[0][col]
        evc = all_results[mol_name]["evo_cons"]
        if not evc.bond_std_history:
            ax.set_visible(False)
            continue

        std_arr = np.array(evc.bond_std_history).T   # (n_bonds, n_generations)
        im = ax.imshow(
            std_arr, aspect="auto", origin="lower",
            cmap="RdYlGn_r", vmin=0, vmax=90,
        )
        # Overlay stable-bond marker
        stable_arr = np.array(evc.bond_stable_history).T
        ys, xs = np.where(stable_arr)
        ax.scatter(xs, ys, marker="s", s=8, color="white",
                   edgecolors="black", linewidths=0.3, alpha=0.6)

        ax.set_xlabel("generation")
        ax.set_ylabel("bond index" if col == 0 else "")
        ax.set_yticks(range(std_arr.shape[0]))
        ax.set_title(mol_name, fontsize=10)
        plt.colorbar(im, ax=ax, label="circular std (°)")

    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, "evo_stabilization.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    all_results = {}

    for mol_name, smiles in MOLECULES:
        print(f"\n=== {mol_name} ===")

        print(f"  Baseline: {N_SINGLE_SEEDS} single runs × {N_SINGLE_STEPS} steps …")
        baseline = run_baseline(smiles, BEST_PARAMS, N_SINGLE_STEPS, N_SINGLE_SEEDS)
        print(f"  baseline E: min={min(baseline):.1f}  "
              f"mean={np.mean(baseline):.1f}  max={max(baseline):.1f}")

        print(f"  Evo (mutation only) …")
        evo_mut = EvoPopulation(smiles, BEST_PARAMS, mode="mutation", seed=42)
        evo_mut.run(N_GENERATIONS)
        bm = evo_mut.best_individual()
        print(f"    best E={bm.final_energy():.1f}")

        print(f"  Evo (consensus transplant) …")
        evo_cons = EvoPopulation(smiles, BEST_PARAMS, mode="consensus", seed=42)
        evo_cons.run(N_GENERATIONS)
        bc = evo_cons.best_individual()
        print(f"    best E={bc.final_energy():.1f}")

        # Per-bond stabilization summary
        if evo_cons.bond_stable_history:
            final_stable = evo_cons.bond_stable_history[-1]
            print(f"    bonds stable at end: "
                  f"{int(final_stable.sum())}/{len(final_stable)}")

        all_results[mol_name] = {
            "baseline": baseline,
            "evo_mut":  evo_mut,
            "evo_cons": evo_cons,
        }

    print("\n=== Plotting ===")
    plot_results(all_results)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
