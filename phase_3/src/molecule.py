"""
molecule.py — RDKit utilities: bond detection, dependency graph, graph coloring.

Heavy-atom indices are stable across AddHs/RemoveHs, so SMARTS detection runs on
the H-stripped molecule while the resulting indices remain valid on mol_h.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from rdkit import Chem
from rdkit.Chem import rdMolTransforms

# Lipinski-style rotatable-bond SMARTS (excludes terminal heavy atoms,
# triple bonds, haloform groups, and neopentyl-type quaternary centres).
_ROTATABLE_SMARTS = (
    "[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)"
    "&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])]"
    "-&!@"
    "[!$(*#*)&!D1]"
)


@dataclass
class RotatableBond:
    bond_idx: int
    atom_i: int
    atom_j: int
    dihedral_atoms: Tuple[int, int, int, int]  # (a, i, j, b) for SetDihedralDeg
    current_angle: float = 0.0
    neighbors: List[int] = field(default_factory=list)


def find_rotatable_bonds(mol_h: Chem.Mol) -> List[RotatableBond]:
    """
    Detect rotatable bonds.

    SMARTS is applied to *RemoveHs(mol_h)* to avoid false positives: after
    AddHs(), CH3/OH atoms lose D1 status on the hydrogen-stripped graph and
    would pass the !D1 filter.  Heavy-atom indices are identical in both
    representations, so dihedral_atoms indices work on mol_h too.
    """
    mol_noh = Chem.RemoveHs(mol_h)
    pattern = Chem.MolFromSmarts(_ROTATABLE_SMARTS)
    matches = mol_noh.GetSubstructMatches(pattern)

    seen: Set[int] = set()
    bonds: List[RotatableBond] = []

    for match in matches:
        i, j = match[0], match[1]
        bond_obj = mol_noh.GetBondBetweenAtoms(i, j)
        if bond_obj is None:
            continue
        bidx = bond_obj.GetIdx()
        if bidx in seen:
            continue
        seen.add(bidx)

        # Collect heavy-atom anchor candidates for dihedral (a, i, j, b).
        i_nbrs = [
            n.GetIdx()
            for n in mol_noh.GetAtomWithIdx(i).GetNeighbors()
            if n.GetIdx() != j
        ]
        j_nbrs = [
            n.GetIdx()
            for n in mol_noh.GetAtomWithIdx(j).GetNeighbors()
            if n.GetIdx() != i
        ]
        if not i_nbrs or not j_nbrs:
            continue  # no anchor available

        bonds.append(
            RotatableBond(
                bond_idx=bidx,
                atom_i=i,
                atom_j=j,
                dihedral_atoms=(i_nbrs[0], i, j, j_nbrs[0]),
            )
        )

    return bonds


def build_dependency_graph(
    bonds: List[RotatableBond], mol: Chem.Mol
) -> Dict[int, Set[int]]:
    """
    Build bond dependency graph.

    Two bonds are dependent if:
      (a) they share an atom directly, OR
      (b) any endpoint of one bond is a graph-neighbour of any endpoint of the
          other bond (i.e. separated by exactly one atom).

    Condition (b) is required so that, e.g., aspirin's O-Ar and Ar-COOH bonds
    (which meet at a common aromatic carbon neighbour) are correctly linked.

    Parameters
    ----------
    bonds : detected rotatable bonds (indices refer to mol)
    mol   : heavy-atom molecule (no Hs)
    """
    # Quick lookup: atom index → list of bond_idx
    atom_to_bonds: Dict[int, List[int]] = {}
    for b in bonds:
        for atom in (b.atom_i, b.atom_j):
            atom_to_bonds.setdefault(atom, []).append(b.bond_idx)

    graph: Dict[int, Set[int]] = {b.bond_idx: set() for b in bonds}

    def _link(a: int, b: int) -> None:
        if a != b:
            graph[a].add(b)
            graph[b].add(a)

    for ba in bonds:
        endpoints_a = {ba.atom_i, ba.atom_j}
        for atom in endpoints_a:
            # (a) direct sharing
            for other_bidx in atom_to_bonds.get(atom, []):
                _link(ba.bond_idx, other_bidx)

            # (b) one-hop separation
            for nbr in mol.GetAtomWithIdx(atom).GetNeighbors():
                nbr_idx = nbr.GetIdx()
                if nbr_idx in endpoints_a:
                    continue  # already covered by (a)
                for other_bidx in atom_to_bonds.get(nbr_idx, []):
                    _link(ba.bond_idx, other_bidx)

    return graph


def greedy_graph_coloring(graph: Dict[int, Set[int]]) -> Dict[int, int]:
    """
    Standard greedy graph coloring (O(V+E)).

    Returns {node: color_int}.  Nodes with the same color form an independent
    set and can be updated in parallel without race conditions.
    """
    coloring: Dict[int, int] = {}
    for node in sorted(graph):
        used = {coloring[nbr] for nbr in graph[node] if nbr in coloring}
        color = 0
        while color in used:
            color += 1
        coloring[node] = color
    return coloring


def read_angles(
    mol: Chem.Mol, bonds: List[RotatableBond], conf_id: int = 0
) -> None:
    """Read current dihedral angles from a RDKit conformer into RotatableBond objects."""
    conf = mol.GetConformer(conf_id)
    for b in bonds:
        a, i, j, bk = b.dihedral_atoms
        b.current_angle = rdMolTransforms.GetDihedralDeg(conf, a, i, j, bk)
