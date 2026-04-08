"""
visualize_mol.py
----------------
Diagnostic tool: energy profiles and 2D structure visualization.

Note: this script creates its own mol and force field instances,
independent of MoleculeModel. This is intentional — it operates on
a fresh ETKDG geometry, not on post-simulation state.

Outputs (written to results/):
    aspirin_energy_profiles.png -- E(phi) scans for each rotatable bond
    aspirin_structure.svg       -- 2D structure with highlighted bonds
                                   (color = scheduler color group)

Usage:
    uv run python src/visualize_mol.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Note: this script creates its own mol and force field instances,
# independent of MoleculeModel. This is intentional — it operates on
# a fresh ETKDG geometry, not on post-simulation state.

import numpy as np
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms, rdForceFieldHelpers
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from molecule import find_rotatable_bonds, read_angles, build_dependency_graph, greedy_graph_coloring

SMILES = "CC(=O)Oc1ccccc1C(=O)O"

# Scheduler color -> hex color for plots
SCHED_COLORS = {
    0: "#3B8BD4",   # blue  -- parallel group 0
    1: "#1D9E75",   # green -- parallel group 1
    2: "#D85A30",   # coral -- parallel group 2 (if needed)
}

BOND_LABELS = {
    2: "acetyl C=O\n(ester)",
    3: "O->Ar\n(ester oxygen)",
    9: "Ar->COOH\n(carboxyl)",
}


def get_bonds_with_coloring(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMolecule(mol_h, params)

    bonds = find_rotatable_bonds(mol_h)
    dep_graph = build_dependency_graph(bonds, mol_h)
    coloring = greedy_graph_coloring(dep_graph)
    read_angles(mol_h, bonds, conf_id=0)

    return mol, mol_h, bonds, coloring


def plot_energy_profiles(mol_h, bonds, coloring, output_path):
    """
    For each rotatable bond: scan phi from -180 to 180 deg in 5 deg steps,
    compute MMFF94 energy (all other bonds frozen), plot E(phi) profile.

    The kT line at 300K shows which barriers are thermally accessible.
    """
    ff_props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol_h)
    ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol_h, ff_props)
    conf = mol_h.GetConformer(0)

    n = len(bonds)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    angles_range = np.linspace(-180, 175, 72)   # 5 deg resolution

    for ax, rb in zip(axes, bonds):
        color_idx = coloring.get(rb.bond_idx, 0)
        color = SCHED_COLORS.get(color_idx, "#888888")

        # Save current angles of all bonds
        saved = {b.bond_idx: rdMolTransforms.GetDihedralDeg(conf, *b.dihedral_atoms)
                 for b in bonds}

        energies = []
        for phi in angles_range:
            rdMolTransforms.SetDihedralDeg(conf, *rb.dihedral_atoms, phi)
            energies.append(ff.CalcEnergy())

        # Restore all angles
        for b in bonds:
            rdMolTransforms.SetDihedralDeg(conf, *b.dihedral_atoms, saved[b.bond_idx])

        energies = np.array(energies)
        e_rel = energies - energies.min()

        ax.plot(angles_range, e_rel, color=color, lw=2.5, zorder=3)
        ax.fill_between(angles_range, e_rel, alpha=0.12, color=color)

        # Mark energy minimum
        min_phi = angles_range[np.argmin(energies)]
        ax.axvline(min_phi, color="darkred", lw=1.5, ls="--", alpha=0.7,
                   label=f"min: {min_phi:.0f} deg")

        # kT line — barriers above this are thermally inaccessible at 300K
        kT = 0.001987 * 300
        ax.axhline(kT, color="gray", lw=1, ls=":", alpha=0.6,
                   label=f"kT (300K) = {kT:.2f} kcal/mol")

        label = BOND_LABELS.get(rb.bond_idx, f"Bond {rb.bond_idx}")
        sched_label = f"Scheduler color: {color_idx}"
        ax.set_title(f"Bond {rb.bond_idx} — {label}\n{sched_label}", fontsize=10)
        ax.set_xlabel("Dihedral angle phi (deg)", fontsize=9)
        ax.set_ylabel("Delta E (kcal/mol)" if bonds.index(rb) == 0 else "", fontsize=9)
        ax.set_xlim(-180, 180)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

        n_minima = max(1, np.sum(np.diff(np.sign(np.diff(e_rel))) < 0))
        ax.text(0.02, 0.97, f"Local minima: ~{n_minima}",
                transform=ax.transAxes, fontsize=8, va="top", color="gray")

    fig.suptitle(
        "MMFF94 energy profiles — Aspirin\n"
        "E(phi) with all other bonds frozen",
        fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Energy profiles: {output_path}")
    plt.close()


def draw_structure_svg(mol, bonds, coloring, output_path):
    """
    2D structure as SVG with highlighted rotatable bonds.
    Bond color = scheduler color group (same color = updated in parallel).
    Open in any browser to view.
    """
    rdDepictor.Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DSVG(600, 380)
    drawer.drawOptions().addAtomIndices = False
    drawer.drawOptions().addStereoAnnotation = False
    drawer.drawOptions().padding = 0.15

    highlight_bonds = []
    bond_colors = {}
    highlight_atoms = []
    atom_colors = {}

    for rb in bonds:
        bond = mol.GetBondBetweenAtoms(rb.atom_i, rb.atom_j)
        if bond is None:
            continue
        bidx = bond.GetIdx()
        highlight_bonds.append(bidx)

        color_idx = coloring.get(rb.bond_idx, 0)
        hex_color = SCHED_COLORS.get(color_idx, "#888888")
        r = int(hex_color[1:3], 16) / 255
        g = int(hex_color[3:5], 16) / 255
        b = int(hex_color[5:7], 16) / 255
        bond_colors[bidx] = (r, g, b)

        for atom_idx in [rb.atom_i, rb.atom_j]:
            if atom_idx not in highlight_atoms:
                highlight_atoms.append(atom_idx)
            atom_colors[atom_idx] = (r, g, b)

    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    svg_text = drawer.GetDrawingText()

    with open(output_path, "w") as f:
        f.write(svg_text)
    print(f"2D structure (SVG): {output_path}")


def print_summary(bonds, coloring, dep_graph):
    from collections import defaultdict
    print("\n=== Bond dependency graph summary ===")
    for b in bonds:
        c = coloring.get(b.bond_idx, "?")
        nbrs = sorted(dep_graph.get(b.bond_idx, set()))
        label = BOND_LABELS.get(b.bond_idx, "")
        print(f"  Bond {b.bond_idx:2d}  [{label:20s}]  "
              f"color={c}  phi={b.current_angle:7.1f} deg  neighbors={nbrs}")

    print("\n=== Scheduler execution order ===")
    groups = defaultdict(list)
    for bidx, c in coloring.items():
        groups[c].append(bidx)
    for c in sorted(groups):
        print(f"  Round {c}: update bonds {sorted(groups[c])} in parallel")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    print("Preparing molecule...")
    mol, mol_h, bonds, coloring = get_bonds_with_coloring(SMILES)
    dep_graph = build_dependency_graph(bonds, mol_h)

    print_summary(bonds, coloring, dep_graph)

    print("\nGenerating energy profiles...")
    plot_energy_profiles(mol_h, bonds, coloring, "results/aspirin_energy_profiles.png")

    print("Generating 2D structure...")
    draw_structure_svg(mol, bonds, coloring, "results/aspirin_structure.svg")

    print("\nDone! Check results/")
    print("  aspirin_energy_profiles.png -- why Bond 2/9 jump in trajectory")
    print("  aspirin_structure.svg       -- open in browser")
