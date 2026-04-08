"""
molecule.py v3
--------------
Poprawka dependency graph: dwa wiązania są zależne jeśli
są SĄSIEDNIE w grafie molekularnym (dzielą atom LUB są
oddzielone dokładnie jednym atomem). Poprzednia wersja
używała tylko bezpośredniego współdzielenia atomu,
przez co Bond 9 (carboxyl) wypadał jako izolowany.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms


@dataclass
class RotatableBond:
    bond_idx: int
    atom_i: int
    atom_j: int
    dihedral_atoms: Tuple
    current_angle: float = 0.0
    neighbors: List[int] = field(default_factory=list)


_ROTATABLE_SMARTS = Chem.MolFromSmarts(
    "[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)"
    "&!$(C([CH3])([CH3])[CH3])]-&!@[!$(*#*)&!D1]"
)


def find_rotatable_bonds(mol_h: Chem.Mol) -> List[RotatableBond]:
    """Wykryj obracalne wiązania na mol bez H, mapuj na mol z H."""
    mol_no_h = Chem.RemoveHs(mol_h)
    matches = mol_no_h.GetSubstructMatches(_ROTATABLE_SMARTS)

    bonds = []
    seen = set()

    for match in matches:
        i_noh, j_noh = match[0], match[1]
        bond = mol_no_h.GetBondBetweenAtoms(i_noh, j_noh)
        if bond is None:
            continue

        bond_h = mol_h.GetBondBetweenAtoms(i_noh, j_noh)
        if bond_h is None:
            continue
        bidx = bond_h.GetIdx()
        if bidx in seen:
            continue
        seen.add(bidx)

        dihedral = _find_dihedral_atoms(mol_h, i_noh, j_noh)
        if dihedral is None:
            continue

        bonds.append(
            RotatableBond(
                bond_idx=bidx,
                atom_i=i_noh,
                atom_j=j_noh,
                dihedral_atoms=dihedral,
            )
        )

    return bonds


def _find_dihedral_atoms(mol: Chem.Mol, i: int, j: int):
    neighbors_i = [
        n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() != j
    ]
    neighbors_j = [
        n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != i
    ]
    if not neighbors_i or not neighbors_j:
        return None

    def heavy_first(nbrs):
        heavy = [n for n in nbrs if mol.GetAtomWithIdx(n).GetAtomicNum() > 1]
        return heavy[0] if heavy else nbrs[0]

    return (heavy_first(neighbors_i), i, j, heavy_first(neighbors_j))


def build_dependency_graph(
    bonds: List[RotatableBond], mol: Chem.Mol = None
) -> Dict[int, Set[int]]:
    """
    Graf zależności: bond_idx -> zbiór zależnych wiązań.

    Definicja zależności (konserwatywna):
      Dwa wiązania są zależne jeśli:
      (a) współdzielą atom (bezpośrednie sąsiedztwo), LUB
      (b) są oddzielone jednym atomem w grafie mol (atom jest
          końcem obu wiązań przez jednego sąsiada)

    Wariant (b) jest kluczowy dla aspiryny: Bond 3 (O-Ar) i
    Bond 9 (Ar-COOH) są oddzielone atomem węgla pierścienia —
    bez tego warunku Bond 9 wypada jako izolowany.
    """
    # Zbiory atomów każdego wiązania (i, j)
    bond_atoms: Dict[int, Set[int]] = {b.bond_idx: {b.atom_i, b.atom_j} for b in bonds}

    # Mapa atom → lista wiązań które go zawierają
    atom_to_bonds: Dict[int, List[int]] = {}
    for b in bonds:
        for atom in [b.atom_i, b.atom_j]:
            atom_to_bonds.setdefault(atom, []).append(b.bond_idx)

    graph: Dict[int, Set[int]] = {b.bond_idx: set() for b in bonds}

    # (a) Bezpośrednie współdzielenie atomu
    for atom, bond_list in atom_to_bonds.items():
        for bi in bond_list:
            for bj in bond_list:
                if bi != bj:
                    graph[bi].add(bj)

    # (b) Zależność przez jednego sąsiada (jeśli mol dostępny)
    if mol is not None:
        bond_map = {b.bond_idx: b for b in bonds}
        for b in bonds:
            for end_atom in [b.atom_i, b.atom_j]:
                # Sprawdź sąsiadów tego atomu
                for nbr in mol.GetAtomWithIdx(end_atom).GetNeighbors():
                    nbr_idx = nbr.GetIdx()
                    if nbr_idx in atom_to_bonds:
                        for other_bidx in atom_to_bonds[nbr_idx]:
                            if other_bidx != b.bond_idx:
                                graph[b.bond_idx].add(other_bidx)

    # Zapisz sąsiadów w obiektach
    for b in bonds:
        b.neighbors = list(graph[b.bond_idx])

    return graph


def greedy_graph_coloring(graph: Dict[int, Set[int]]) -> Dict[int, int]:
    """
    Zachłanne kolorowanie grafowe → {bond_idx: kolor}.
    Wiązania tego samego koloru mogą aktualizować się równolegle.
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
    conf = mol.GetConformer(conf_id)
    for b in bonds:
        b.current_angle = rdMolTransforms.GetDihedralDeg(conf, *b.dihedral_atoms)
