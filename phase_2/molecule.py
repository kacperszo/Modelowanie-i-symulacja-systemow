"""
molecule.py
-----------
Builds the bond dependency graph from an RDKit molecule.

Key design decision: rotatable bond detection runs on mol WITHOUT hydrogens
so that terminal groups (CH3, OH) correctly have degree D1 and are filtered out.
Atom indices are preserved after AddHs(), so they map directly to mol_h.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms


@dataclass
class RotatableBond:
    """Single rotatable bond — future agent state carrier."""
    bond_idx: int
    atom_i: int
    atom_j: int
    dihedral_atoms: Tuple   # (a, i, j, b) — four atoms defining the dihedral angle
    current_angle: float = 0.0
    neighbors: List[int] = field(default_factory=list)  # dependent bond indices


# Standard RDKit SMARTS for rotatable bonds — applied on mol WITHOUT H
# so terminal atoms correctly appear as D1 (one heavy neighbor)
_ROTATABLE_SMARTS = Chem.MolFromSmarts(
    "[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)"
    "&!$(C([CH3])([CH3])[CH3])]-&!@[!$(*#*)&!D1]"
)


def find_rotatable_bonds(mol_h: Chem.Mol) -> List[RotatableBond]:
    """
    Detect rotatable bonds in mol_h (molecule WITH hydrogens).

    Detection runs on mol WITHOUT H to ensure terminal atoms are D1.
    Heavy atom indices are identical before and after AddHs().
    """
    mol_no_h = Chem.RemoveHs(mol_h)
    matches = mol_no_h.GetSubstructMatches(_ROTATABLE_SMARTS)

    bonds = []
    seen = set()

    for match in matches:
        i, j = match[0], match[1]
        bond = mol_no_h.GetBondBetweenAtoms(i, j)
        if bond is None:
            continue

        bond_h = mol_h.GetBondBetweenAtoms(i, j)
        if bond_h is None:
            continue
        bidx = bond_h.GetIdx()
        if bidx in seen:
            continue
        seen.add(bidx)

        dihedral = _find_dihedral_atoms(mol_h, i, j)
        if dihedral is None:
            continue

        bonds.append(RotatableBond(
            bond_idx=bidx,
            atom_i=i,
            atom_j=j,
            dihedral_atoms=dihedral,
        ))

    return bonds


def _find_dihedral_atoms(mol: Chem.Mol, i: int, j: int):
    """Find the (a, i, j, b) quartet for the dihedral at bond i-j."""
    neighbors_i = [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() != j]
    neighbors_j = [n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != i]
    if not neighbors_i or not neighbors_j:
        return None

    def heavy_first(nbrs):
        heavy = [n for n in nbrs if mol.GetAtomWithIdx(n).GetAtomicNum() > 1]
        return heavy[0] if heavy else nbrs[0]

    return (heavy_first(neighbors_i), i, j, heavy_first(neighbors_j))


def build_dependency_graph(bonds: List[RotatableBond], mol: Chem.Mol = None) -> Dict[int, Set[int]]:
    """
    Build bond dependency graph: bond_idx -> set of dependent bond indices.

    Two bonds are dependent if:
      (a) they share an atom directly, OR
      (b) they are separated by exactly one atom in the molecular graph

    Condition (b) is essential for aspirin: Bond 3 (O-Ar) and Bond 9 (Ar-COOH)
    are separated by the aromatic carbon — without it, Bond 9 appears isolated.
    """
    atom_to_bonds: Dict[int, List[int]] = {}
    for b in bonds:
        for atom in [b.atom_i, b.atom_j]:
            atom_to_bonds.setdefault(atom, []).append(b.bond_idx)

    graph: Dict[int, Set[int]] = {b.bond_idx: set() for b in bonds}

    # (a) direct shared atom
    for atom, bond_list in atom_to_bonds.items():
        for bi in bond_list:
            for bj in bond_list:
                if bi != bj:
                    graph[bi].add(bj)

    # (b) separated by one atom
    if mol is not None:
        for b in bonds:
            for end_atom in [b.atom_i, b.atom_j]:
                for nbr in mol.GetAtomWithIdx(end_atom).GetNeighbors():
                    nbr_idx = nbr.GetIdx()
                    if nbr_idx in atom_to_bonds:
                        for other_bidx in atom_to_bonds[nbr_idx]:
                            if other_bidx != b.bond_idx:
                                graph[b.bond_idx].add(other_bidx)

    for b in bonds:
        b.neighbors = list(graph[b.bond_idx])

    return graph


def greedy_graph_coloring(graph: Dict[int, Set[int]]) -> Dict[int, int]:
    """
    Greedy graph coloring — O(V + E).
    Returns {bond_idx: color}.

    Bonds with the same color form an independent set in the dependency graph
    and can be updated in parallel without race conditions.

    For molecular graphs (sparse, low vertex degree <= 4 for sp3),
    greedy coloring achieves optimal or near-optimal chromatic number.
    """
    colors: Dict[int, int] = {}
    for node in sorted(graph.keys()):
        neighbor_colors = {colors[n] for n in graph[node] if n in colors}
        color = 0
        while color in neighbor_colors:
            color += 1
        colors[node] = color
    return colors


def read_angles(mol: Chem.Mol, bonds: List[RotatableBond], conf_id: int = 0) -> None:
    """Read current dihedral angles from an RDKit conformer into RotatableBond objects."""
    conf = mol.GetConformer(conf_id)
    for b in bonds:
        b.current_angle = rdMolTransforms.GetDihedralDeg(conf, *b.dihedral_atoms)
