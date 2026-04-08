"""
run.py — uruchamia ABM i porównuje z benchmarkiem RDKit ETKDG
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

from model import MoleculeModel
from molecule import find_rotatable_bonds, read_angles

MOLECULES = {
    "butane": "CCCC",
    "pentane": "CCCCC",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
}

TARGET_MOL = "aspirin"
SMILES = MOLECULES[TARGET_MOL]
N_STEPS = 500
TEMPERATURE = 300.0  # K — fizjologiczna, bariery rotacji ~1-3 kcal/mol
N_ETKDG_CONFS = 500


def run_etkdg_benchmark(smiles, n_confs):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xDEAD
    params.numThreads = 0
    AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)

    bonds = find_rotatable_bonds(mol)
    all_angles = {b.bond_idx: [] for b in bonds}
    for conf_id in range(mol.GetNumConformers()):
        for b in bonds:
            try:
                angle = rdMolTransforms.GetDihedralDeg(
                    mol.GetConformer(conf_id), *b.dihedral_atoms
                )
                all_angles[b.bond_idx].append(angle)
            except Exception:
                pass
    return bonds, all_angles


def plot_results(model, etkdg_bonds, etkdg_angles, mol_name):
    agents = list(model.schedule.agents)
    n = len(agents)
    if n == 0:
        return

    fig = plt.figure(figsize=(14, 4 * n + 3))
    gs = gridspec.GridSpec(n + 1, 3, figure=fig)
    fig.suptitle(
        f"Dihedral Agents vs ETKDG — {mol_name}\n"
        f"T={TEMPERATURE}K, {N_STEPS} kroków ABM, {N_ETKDG_CONFS} konf. ETKDG",
        fontsize=13,
        y=1.01,
    )

    for i, agent in enumerate(agents):
        bidx = agent.bond.bond_idx

        ax1 = fig.add_subplot(gs[i, 0])
        ax1.plot(agent.angle_history, lw=0.7, color="#3B8BD4", alpha=0.9)
        ax1.axhline(
            agent.angle_history[-1],
            color="#D85A30",
            lw=1,
            ls="--",
            label=f"Końcowy: {agent.angle_history[-1]:.1f}°",
        )
        ax1.set_ylabel(f"Bond {bidx}\nφ (°)", fontsize=9)
        ax1.set_ylim(-185, 185)
        ax1.legend(fontsize=8)
        if i == 0:
            ax1.set_title("Trajektoria ABM", fontsize=10)

        ax2 = fig.add_subplot(gs[i, 1])
        ax2.hist(
            agent.angle_history,
            bins=36,
            range=(-180, 180),
            color="#3B8BD4",
            alpha=0.7,
            density=True,
        )
        ax2.set_xlim(-180, 180)
        if i == 0:
            ax2.set_title("Rozkład ABM", fontsize=10)

        ax3 = fig.add_subplot(gs[i, 2])
        if bidx in etkdg_angles and etkdg_angles[bidx]:
            ax3.hist(
                etkdg_angles[bidx],
                bins=36,
                range=(-180, 180),
                color="#1D9E75",
                alpha=0.7,
                density=True,
            )
        ax3.set_xlim(-180, 180)
        if i == 0:
            ax3.set_title("Rozkład ETKDG (benchmark)", fontsize=10)

    ax_e = fig.add_subplot(gs[n, :])
    steps = [i * 10 for i in range(len(model.energy_snapshots))]
    ax_e.plot(steps, model.energy_snapshots, color="#D85A30", lw=1.2)
    ax_e.set_xlabel("Krok")
    ax_e.set_ylabel("Energia MMFF94 (kcal/mol)")
    ax_e.set_title("Zbieżność energii modelu", fontsize=10)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    out = f"results/{mol_name}_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wykres: {out}")
    plt.close()


def plot_graph_coloring(model, mol_name):
    coloring = model.coloring
    bonds = model.bonds
    dep_graph = model.dep_graph
    if not coloring:
        return

    n_colors = len(set(coloring.values()))
    palette = plt.cm.Set2(np.linspace(0, 1, max(n_colors, 3)))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(
        f"Graf zależności wiązań — kolorowanie grafowe\n{mol_name} | {n_colors} kolorów",
        fontsize=11,
    )

    bond_ids = [b.bond_idx for b in bonds]
    n = len(bond_ids)
    angles_pos = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {bid: (np.cos(a), np.sin(a)) for bid, a in zip(bond_ids, angles_pos)}

    drawn = set()
    for bid, nbrs in dep_graph.items():
        for nbr in nbrs:
            edge = tuple(sorted((bid, nbr)))
            if edge not in drawn and bid in pos and nbr in pos:
                ax.plot(
                    [pos[bid][0], pos[nbr][0]],
                    [pos[bid][1], pos[nbr][1]],
                    "k-",
                    alpha=0.2,
                    lw=1.2,
                    zorder=1,
                )
                drawn.add(edge)

    for bid in bond_ids:
        c = coloring.get(bid, 0)
        x, y = pos[bid]
        circle = plt.Circle(
            (x, y), 0.1, color=palette[c % len(palette)], zorder=3, ec="white", lw=1.5
        )
        ax.add_patch(circle)
        ax.text(
            x,
            y,
            str(bid),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            zorder=4,
            color="white",
        )
        ax.text(
            x * 1.22,
            y * 1.22,
            f"c={c}",
            ha="center",
            va="center",
            fontsize=7,
            color="gray",
        )

    for c in range(n_colors):
        ax.scatter(
            [],
            [],
            color=palette[c % len(palette)],
            s=80,
            label=f"Kolor {c} → równoległa grupa {c}",
        )
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")

    os.makedirs("results", exist_ok=True)
    out = f"results/{mol_name}_graph_coloring.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Graf: {out}")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print(f"Dihedral Agents — {TARGET_MOL}, T={TEMPERATURE}K, MMFF94")
    print("=" * 60 + "\n")

    model = MoleculeModel(
        smiles=SMILES,
        n_steps=N_STEPS,
        temperature=TEMPERATURE,
        seed=42,
    )
    model.run()

    print(f"\nBenchmark ETKDG ({N_ETKDG_CONFS} konformacji)...")
    etkdg_bonds, etkdg_angles = run_etkdg_benchmark(SMILES, N_ETKDG_CONFS)

    print("\nGeneruję wykresy...")
    plot_results(model, etkdg_bonds, etkdg_angles, TARGET_MOL)
    plot_graph_coloring(model, TARGET_MOL)
    print("Gotowe! Sprawdź results/")
