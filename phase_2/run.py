"""
run.py
------
Main entry point: runs ABM simulation and compares with RDKit ETKDG benchmark.

Usage:
    uv run python src/run.py

Outputs (written to results/):
    {mol}_comparison.png     -- ABM trajectory + histogram vs ETKDG histogram
    {mol}_graph_coloring.png -- dependency graph with color groups visualized
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms, rdForceFieldHelpers

from molecule import find_rotatable_bonds, read_angles
from model import MoleculeModel


# ── Configuration ─────────────────────────────────────────────────────────────

MOLECULES = {
    "butane":  "CCCC",
    "pentane": "CCCCC",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
}

TARGET_MOL    = "aspirin"
SMILES        = MOLECULES[TARGET_MOL]
N_STEPS       = 500
TEMPERATURE   = 300.0     # K — physiological temperature
N_ETKDG_CONFS = 500       # conformers generated as benchmark


# ── ETKDG benchmark ───────────────────────────────────────────────────────────

def run_etkdg_benchmark(smiles: str, n_confs: int):
    """Generate n_confs conformers with RDKit ETKDG and return dihedral distributions."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xDEAD
    params.numThreads = 0   # use all available cores
    AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)

    bonds = find_rotatable_bonds(mol)
    all_angles = {b.bond_idx: [] for b in bonds}

    for conf_id in range(mol.GetNumConformers()):
        for b in bonds:
            try:
                angle = rdMolTransforms.GetDihedralDeg(
                    mol.GetConformer(conf_id), *b.dihedral_atoms)
                all_angles[b.bond_idx].append(angle)
            except Exception:
                pass

    return bonds, all_angles


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_comparison(model, etkdg_bonds, etkdg_angles, mol_name: str):
    """Side-by-side comparison: ABM trajectory + histogram vs ETKDG histogram."""
    agents = list(model.schedule.agents)
    n = len(agents)
    if n == 0:
        return

    fig = plt.figure(figsize=(14, 4 * n + 3))
    gs = gridspec.GridSpec(n + 1, 3, figure=fig)
    fig.suptitle(
        f"Dihedral Agents vs ETKDG — {mol_name}\n"
        f"T={TEMPERATURE}K, {N_STEPS} ABM steps, {N_ETKDG_CONFS} ETKDG conformers",
        fontsize=13, y=1.01)

    for i, agent in enumerate(agents):
        bidx = agent.bond.bond_idx

        # Trajectory
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.plot(agent.angle_history, lw=0.7, color="#3B8BD4", alpha=0.9)
        ax1.axhline(agent.angle_history[-1], color="#D85A30", lw=1, ls="--",
                    label=f"Final: {agent.angle_history[-1]:.1f} deg")
        ax1.set_ylabel(f"Bond {bidx}\nphi (deg)", fontsize=9)
        ax1.set_ylim(-185, 185)
        ax1.legend(fontsize=8)
        if i == 0:
            ax1.set_title("ABM trajectory", fontsize=10)

        # ABM histogram
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.hist(agent.angle_history, bins=36, range=(-180, 180),
                 color="#3B8BD4", alpha=0.7, density=True)
        ax2.set_xlim(-180, 180)
        if i == 0:
            ax2.set_title("ABM distribution", fontsize=10)

        # ETKDG histogram (benchmark)
        ax3 = fig.add_subplot(gs[i, 2])
        if bidx in etkdg_angles and etkdg_angles[bidx]:
            ax3.hist(etkdg_angles[bidx], bins=36, range=(-180, 180),
                     color="#1D9E75", alpha=0.7, density=True)
        ax3.set_xlim(-180, 180)
        if i == 0:
            ax3.set_title("ETKDG distribution (benchmark)", fontsize=10)

    # Total energy convergence
    ax_e = fig.add_subplot(gs[n, :])
    steps = [i * 10 for i in range(len(model.energy_snapshots))]
    ax_e.plot(steps, model.energy_snapshots, color="#D85A30", lw=1.2)
    ax_e.set_xlabel("Step")
    ax_e.set_ylabel("MMFF94 energy (kcal/mol)")
    ax_e.set_title("Energy convergence", fontsize=10)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    out = f"results/{mol_name}_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_graph_coloring(model, mol_name: str):
    """Visualize the bond dependency graph with color groups."""
    coloring = model.coloring
    bonds = model.bonds
    dep_graph = model.dep_graph
    if not coloring:
        return

    n_colors = len(set(coloring.values()))
    palette = plt.cm.Set2(np.linspace(0, 1, max(n_colors, 3)))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(
        f"Bond dependency graph — graph coloring\n"
        f"{mol_name} | {n_colors} colors", fontsize=11)

    bond_ids = [b.bond_idx for b in bonds]
    n = len(bond_ids)
    angles_pos = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {bid: (np.cos(a), np.sin(a)) for bid, a in zip(bond_ids, angles_pos)}

    drawn = set()
    for bid, nbrs in dep_graph.items():
        for nbr in nbrs:
            edge = tuple(sorted((bid, nbr)))
            if edge not in drawn and bid in pos and nbr in pos:
                ax.plot([pos[bid][0], pos[nbr][0]], [pos[bid][1], pos[nbr][1]],
                        "k-", alpha=0.2, lw=1.2, zorder=1)
                drawn.add(edge)

    for bid in bond_ids:
        c = coloring.get(bid, 0)
        x, y = pos[bid]
        circle = plt.Circle((x, y), 0.1, color=palette[c % len(palette)],
                             zorder=3, ec="white", lw=1.5)
        ax.add_patch(circle)
        ax.text(x, y, str(bid), ha="center", va="center",
                fontsize=9, fontweight="bold", zorder=4, color="white")
        ax.text(x * 1.22, y * 1.22, f"c={c}", ha="center", va="center",
                fontsize=7, color="gray")

    for c in range(n_colors):
        ax.scatter([], [], color=palette[c % len(palette)], s=80,
                   label=f"Color {c} -> parallel group {c}")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")

    os.makedirs("results", exist_ok=True)
    out = f"results/{mol_name}_graph_coloring.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ── Minimum energy comparison ─────────────────────────────────────────────────

def run_rdkit_minimization(smiles: str, seed: int = 42):
    """
    Generate one ETKDG conformer and minimize with MMFF94.
    Returns (energy_kcal, bond_angles_dict) where bond_angles_dict maps bond_idx -> angle.
    This is the non-agent RDKit reference minimum.
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    AllChem.EmbedMolecule(mol, params)

    ff_props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
    ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, ff_props)
    ff.Minimize(maxIts=2000)

    energy = ff.CalcEnergy()
    bonds = find_rotatable_bonds(mol)
    read_angles(mol, bonds)
    angles = {b.bond_idx: b.current_angle for b in bonds}
    return energy, angles


def _build_abm_best_mol(model):
    """
    Reconstruct molecule geometry at the best ABM snapshot (lowest energy).
    Returns (mol_copy, snapshot_idx, best_angles_dict).
    """
    abm_min_idx = int(np.argmin(model.energy_snapshots))
    best_angles = model.angle_snapshots[abm_min_idx]

    mol = Chem.RWMol(model.mol)
    conf = mol.GetConformer(0)
    for i, agent in enumerate(model.schedule.agents):
        rdMolTransforms.SetDihedralDeg(conf, *agent.bond.dihedral_atoms, best_angles[i])

    angles_dict = {agent.bond.bond_idx: best_angles[i]
                   for i, agent in enumerate(model.schedule.agents)}
    return mol, abm_min_idx, angles_dict


def _draw_mol_3d(ax, mol, rot_bond_set, coloring, title):
    """
    Draw a 3D molecule on a matplotlib Axes3D.
    Heavy atoms only (H hidden for clarity).
    Rotatable bonds are colored by scheduler color group.
    """
    ELEMENT_COLOR = {6: "#888888", 8: "#E04040", 7: "#4060D8", 16: "#C8C820"}
    ELEMENT_SIZE  = {6: 120, 8: 160, 7: 140, 16: 200}
    ROT_COLORS    = {0: "#3B8BD4", 1: "#1D9E75", 2: "#D85A30"}

    conf = mol.GetConformer(0)

    # bonds
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ai, aj = mol.GetAtomWithIdx(i), mol.GetAtomWithIdx(j)
        if ai.GetAtomicNum() == 1 or aj.GetAtomicNum() == 1:
            continue
        pi, pj = conf.GetAtomPosition(i), conf.GetAtomPosition(j)
        bidx = bond.GetIdx()
        if bidx in rot_bond_set:
            c = coloring.get(bidx, 0)
            color, lw, alpha = ROT_COLORS.get(c, "#888888"), 3.0, 1.0
        else:
            color, lw, alpha = "#AAAAAA", 1.2, 0.6
        ax.plot([pi.x, pj.x], [pi.y, pj.y], [pi.z, pj.z],
                color=color, lw=lw, alpha=alpha, solid_capstyle="round")

    # atoms (heavy only)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        anum = atom.GetAtomicNum()
        ax.scatter([pos.x], [pos.y], [pos.z],
                   c=ELEMENT_COLOR.get(anum, "#888888"),
                   s=ELEMENT_SIZE.get(anum, 120),
                   edgecolors="white", linewidths=0.5,
                   depthshade=True, zorder=5)

    ax.set_title(title, fontsize=10, pad=8)
    ax.set_axis_off()


def plot_3d_conformer_comparison(model, mol_name: str):
    """
    Side-by-side 3D structures:
      Left  — ABM best snapshot (lowest energy during trajectory)
      Right — RDKit MMFF94-minimized conformer

    Both molecules are aligned to the same coordinate frame (AlignMol on
    heavy atoms) so the ring plane and substituents face the same direction.
    Both subplots use identical camera angle and axis limits.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from rdkit.Chem import rdMolAlign

    abm_mol, abm_idx, abm_angles = _build_abm_best_mol(model)
    rdkit_energy, rdkit_angles    = run_rdkit_minimization(SMILES)

    # Build minimized mol for 3D drawing
    rdkit_mol = Chem.MolFromSmiles(SMILES)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    params = AllChem.ETKDGv3(); params.randomSeed = 42
    AllChem.EmbedMolecule(rdkit_mol, params)
    ff_props = rdForceFieldHelpers.MMFFGetMoleculeProperties(rdkit_mol)
    ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(rdkit_mol, ff_props)
    ff.Minimize(maxIts=2000)

    # Align rdkit_mol onto abm_mol — same coordinate frame, same orientation
    heavy_map = [(i, i) for i in range(rdkit_mol.GetNumAtoms())
                 if rdkit_mol.GetAtomWithIdx(i).GetAtomicNum() != 1]
    rdMolAlign.AlignMol(rdkit_mol, abm_mol, atomMap=heavy_map)

    # Compute shared axis limits from both molecules combined
    def heavy_positions(mol):
        conf = mol.GetConformer(0)
        return np.array([
            [conf.GetAtomPosition(i).x,
             conf.GetAtomPosition(i).y,
             conf.GetAtomPosition(i).z]
            for i in range(mol.GetNumAtoms())
            if mol.GetAtomWithIdx(i).GetAtomicNum() != 1
        ])

    all_pos = np.vstack([heavy_positions(abm_mol), heavy_positions(rdkit_mol)])
    pad = 1.0
    lims = [(all_pos[:, k].min() - pad, all_pos[:, k].max() + pad) for k in range(3)]

    rot_bond_set = {b.bond_idx for b in model.bonds}
    coloring     = model.coloring
    abm_energy   = model.energy_snapshots[abm_idx]
    step_label   = abm_idx * 10

    fig = plt.figure(figsize=(13, 8))
    fig.suptitle(
        f"3D conformation comparison — {mol_name}\n"
        f"Rotatable bonds: blue = group 0, green = group 1",
        fontsize=12, y=0.98,
    )

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    _draw_mol_3d(ax1, abm_mol,   rot_bond_set, coloring,
                 f"ABM best  (step {step_label})\nE = {abm_energy:.2f} kcal/mol")
    _draw_mol_3d(ax2, rdkit_mol, rot_bond_set, coloring,
                 f"RDKit MMFF94 minimized\nE = {rdkit_energy:.2f} kcal/mol")

    # Identical camera angle and axis limits for both subplots
    ELEV, AZIM = 20, -70
    for ax in (ax1, ax2):
        ax.view_init(elev=ELEV, azim=AZIM)
        ax.set_xlim(*lims[0])
        ax.set_ylim(*lims[1])
        ax.set_zlim(*lims[2])

    # Dihedral angle table below each structure
    bond_ids = sorted(abm_angles.keys())
    def angle_table(ax, angles_dict):
        lines = ["φ (deg):"]
        for b in bond_ids:
            lines.append(f"  Bond {b}: {angles_dict.get(b, 0.0):+.1f}°")
        ax.text2D(0.5, -0.02, "\n".join(lines),
                  transform=ax.transAxes, ha="center", va="top",
                  fontsize=9, family="monospace",
                  bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray", alpha=0.85))

    angle_table(ax1, abm_angles)
    angle_table(ax2, rdkit_angles)

    delta = abm_energy - rdkit_energy
    fig.text(0.5, 0.01,
             f"ΔE (ABM − RDKit) = {delta:+.2f} kcal/mol",
             ha="center", fontsize=11,
             bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray", alpha=0.9))

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    os.makedirs("results", exist_ok=True)
    out = f"results/{mol_name}_3d_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_minimum_comparison(model, mol_name: str):
    """
    Single plot comparing:
      - ABM best conformation (lowest energy snapshot from trajectory)
      - RDKit MMFF94-minimized conformation (non-agent reference)

    Top panel:    bar chart of energies
    Bottom panel: grouped bar chart of dihedral angles per bond
    """
    # ABM best: snapshot with lowest recorded energy
    abm_min_idx = int(np.argmin(model.energy_snapshots))
    abm_energy  = model.energy_snapshots[abm_min_idx]
    abm_angles  = {
        agent.bond.bond_idx: model.angle_snapshots[abm_min_idx][i]
        for i, agent in enumerate(model.schedule.agents)
    }

    # RDKit reference minimum
    rdkit_energy, rdkit_angles = run_rdkit_minimization(SMILES)

    bond_ids = sorted(abm_angles.keys())
    x = np.arange(len(bond_ids))
    width = 0.35

    fig, (ax_e, ax_a) = plt.subplots(2, 1, figsize=(8, 8))
    fig.suptitle(
        f"Minimum energy conformation — {mol_name}\n"
        f"ABM best snapshot vs RDKit MMFF94 minimization",
        fontsize=13,
    )

    # ── Energy bar chart ──
    bars = ax_e.bar(
        ["ABM best\n(trajectory min)", "RDKit MMFF94\n(ff.Minimize)"],
        [abm_energy, rdkit_energy],
        color=["#3B8BD4", "#1D9E75"],
        width=0.4,
        edgecolor="white",
    )
    ax_e.bar_label(bars, fmt="%.2f kcal/mol", padding=4, fontsize=10)
    ax_e.set_ylabel("MMFF94 energy (kcal/mol)")
    ax_e.set_title("Total MMFF94 energy", fontsize=11)
    delta = abm_energy - rdkit_energy
    ax_e.text(
        0.98, 0.95,
        f"ΔE = {delta:+.2f} kcal/mol\n(ABM − RDKit)",
        transform=ax_e.transAxes,
        ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8),
    )
    ax_e.set_ylim(0, max(abm_energy, rdkit_energy) * 1.15)

    # ── Dihedral angles grouped bar chart ──
    abm_vals   = [abm_angles[b]   for b in bond_ids]
    rdkit_vals = [rdkit_angles.get(b, 0.0) for b in bond_ids]

    ax_a.bar(x - width / 2, abm_vals,   width, label="ABM best",          color="#3B8BD4", alpha=0.85)
    ax_a.bar(x + width / 2, rdkit_vals, width, label="RDKit MMFF94 min",  color="#1D9E75", alpha=0.85)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f"Bond {b}" for b in bond_ids], fontsize=10)
    ax_a.set_ylabel("Dihedral angle φ (deg)")
    ax_a.set_title("Dihedral angles at minimum", fontsize=11)
    ax_a.axhline(0, color="gray", lw=0.8, ls="--")
    ax_a.set_ylim(-200, 200)
    ax_a.set_yticks([-180, -90, 0, 90, 180])
    ax_a.legend(fontsize=10)

    for i, (av, rv) in enumerate(zip(abm_vals, rdkit_vals)):
        ax_a.text(i - width / 2, av + (6 if av >= 0 else -12),
                  f"{av:.1f}°", ha="center", fontsize=8, color="#3B8BD4")
        ax_a.text(i + width / 2, rv + (6 if rv >= 0 else -12),
                  f"{rv:.1f}°", ha="center", fontsize=8, color="#1D9E75")

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    out = f"results/{mol_name}_minimum_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

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

    print(f"\nRunning ETKDG benchmark ({N_ETKDG_CONFS} conformers)...")
    etkdg_bonds, etkdg_angles = run_etkdg_benchmark(SMILES, N_ETKDG_CONFS)

    print("\nGenerating plots...")
    plot_comparison(model, etkdg_bonds, etkdg_angles, TARGET_MOL)
    plot_graph_coloring(model, TARGET_MOL)
    plot_minimum_comparison(model, TARGET_MOL)
    plot_3d_conformer_comparison(model, TARGET_MOL)

    # Save conformers as SDF for external 3D viewing
    abm_mol, abm_idx, _ = _build_abm_best_mol(model)
    rdkit_mol2 = Chem.MolFromSmiles(SMILES)
    rdkit_mol2 = Chem.AddHs(rdkit_mol2)
    p = AllChem.ETKDGv3(); p.randomSeed = 42
    AllChem.EmbedMolecule(rdkit_mol2, p)
    ff2 = rdForceFieldHelpers.MMFFGetMoleculeForceField(
        rdkit_mol2, rdForceFieldHelpers.MMFFGetMoleculeProperties(rdkit_mol2))
    ff2.Minimize(maxIts=2000)

    os.makedirs("results", exist_ok=True)
    w = Chem.SDWriter("results/aspirin_abm_best.sdf")
    abm_mol.SetProp("_Name", f"ABM_best_step{abm_idx*10}")
    w.write(abm_mol); w.close()

    w2 = Chem.SDWriter("results/aspirin_rdkit_min.sdf")
    rdkit_mol2.SetProp("_Name", "RDKit_MMFF94_min")
    w2.write(rdkit_mol2); w2.close()

    print("Saved: results/aspirin_abm_best.sdf")
    print("Saved: results/aspirin_rdkit_min.sdf")
    print("\nDone! Check results/")
