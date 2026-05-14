"""
tests.py — Unit tests for dihedral-agents.

Group 1 (pure Python, no RDKit): graph coloring correctness.
Group 2 (requires RDKit): molecular detection and sync round-trip.

Exit code 0 if all tests pass, 1 if any fail.
"""

from __future__ import annotations

import random
import sys
import traceback
from typing import Callable, List, Tuple


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


def run_suite(tests: List[Tuple[str, Callable]]) -> int:
    n_pass = n_fail = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            n_pass += 1
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc()
            n_fail += 1
    return n_fail


# ===========================================================================
# Group 1 — Pure Python graph coloring (no RDKit)
# ===========================================================================


def test_path3_coloring() -> None:
    """A 3-node path graph is 2-chromatic: endpoints get the same colour."""
    from molecule import greedy_graph_coloring

    graph = {0: {1}, 1: {0, 2}, 2: {1}}
    colors = greedy_graph_coloring(graph)

    assert colors[0] != colors[1], "Adjacent nodes 0-1 must differ"
    assert colors[1] != colors[2], "Adjacent nodes 1-2 must differ"
    assert colors[0] == colors[2], "Non-adjacent endpoints must share a colour (2-chromatic)"


def test_triangle_coloring() -> None:
    """K3 (triangle) requires exactly 3 colours with greedy coloring."""
    from molecule import greedy_graph_coloring

    graph = {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}
    colors = greedy_graph_coloring(graph)

    assert colors[0] != colors[1], "Adjacent 0-1 must differ"
    assert colors[1] != colors[2], "Adjacent 1-2 must differ"
    assert colors[0] != colors[2], "Adjacent 0-2 must differ"
    assert len(set(colors.values())) == 3, "K3 needs exactly 3 colours"


def test_aspirin_coloring() -> None:
    """
    Aspirin dependency graph {2:{3}, 3:{2,9}, 9:{3}}.

    Bond2 and Bond9 are connected only through Bond3 (a path), so they form
    the parallel colour group: greedy coloring must give them the same colour.
    """
    from molecule import greedy_graph_coloring

    graph = {2: {3}, 3: {2, 9}, 9: {3}}
    colors = greedy_graph_coloring(graph)

    assert colors[2] != colors[3], "Bond2 and Bond3 must have different colours"
    assert colors[9] != colors[3], "Bond9 and Bond3 must have different colours"
    assert colors[2] == colors[9], "Bond2 and Bond9 must share a colour (parallel group)"


def test_fuzz_coloring() -> None:
    """1000 random graphs: greedy coloring must produce a valid (proper) coloring."""
    from molecule import greedy_graph_coloring

    rng = random.Random(2024)
    for trial in range(1000):
        n = rng.randint(2, 25)
        graph = {v: set() for v in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.30:
                    graph[i].add(j)
                    graph[j].add(i)

        colors = greedy_graph_coloring(graph)

        for v, nbrs in graph.items():
            for nbr in nbrs:
                assert colors[v] != colors[nbr], (
                    f"Trial {trial}: adjacent nodes {v} and {nbr} "
                    f"have the same colour {colors[v]}"
                )


# ===========================================================================
# Group 2 — Molecular tests (requires RDKit)
# ===========================================================================


def _embed(smiles: str):
    """Return (mol_h, mol_noh) with a valid ETKDG conformer."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Cannot parse SMILES: {smiles!r}"
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    rc = AllChem.EmbedMolecule(mol_h, params)
    if rc == -1:
        AllChem.EmbedMolecule(mol_h)
    mol_noh = Chem.RemoveHs(mol_h)
    return mol_h, mol_noh


def test_aspirin_bond_count() -> None:
    """Aspirin (CC(=O)Oc1ccccc1C(=O)O) must have exactly 3 rotatable bonds."""
    from molecule import find_rotatable_bonds

    mol_h, _ = _embed("CC(=O)Oc1ccccc1C(=O)O")
    bonds = find_rotatable_bonds(mol_h)
    assert len(bonds) == 3, f"Expected 3 rotatable bonds, got {len(bonds)}"


def test_dependency_graph_connected() -> None:
    """Aspirin's bond dependency graph must be connected (one component)."""
    from molecule import build_dependency_graph, find_rotatable_bonds

    mol_h, mol_noh = _embed("CC(=O)Oc1ccccc1C(=O)O")
    bonds = find_rotatable_bonds(mol_h)
    dep_graph = build_dependency_graph(bonds, mol_noh)

    # BFS from the first node
    start = next(iter(dep_graph))
    visited = {start}
    queue = [start]
    while queue:
        node = queue.pop()
        for nbr in dep_graph[node]:
            if nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)

    assert len(visited) == len(dep_graph), (
        f"Dependency graph has {len(dep_graph)} nodes but only {len(visited)} reachable"
    )


def test_bond_detection_counts() -> None:
    """Verify bond counts for canonical molecules."""
    from molecule import find_rotatable_bonds

    cases = [
        ("CCCC", 1),                            # butane: one central C-C bond
        ("CCCCC", 2),                           # pentane: two inner C-C bonds
        ("CC(=O)Oc1ccccc1C(=O)O", 3),           # aspirin: three rotatable bonds
        ("c1ccccc1", 0),                        # benzene: aromatic, none rotatable
    ]
    for smiles, expected in cases:
        mol_h, _ = _embed(smiles)
        bonds = find_rotatable_bonds(mol_h)
        assert len(bonds) == expected, (
            f"SMILES={smiles!r}: expected {expected} rotatable bonds, got {len(bonds)}"
        )


def test_sync_round_trip() -> None:
    """
    sync_angle_from_model followed by sync_angle_to_model must preserve the
    dihedral angle within 0.01°.
    """
    from rdkit.Chem import rdMolTransforms

    from model import MoleculeModel

    model = MoleculeModel(
        smiles="CC(=O)Oc1ccccc1C(=O)O",
        strategy="isolated",
        init="etkdg",
        n_steps=0,
        seed=42,
    )
    agent = model.bond_agents[0]
    a, i, j, b = agent.bond.dihedral_atoms

    # Set a specific angle in master_mol
    target = 47.3
    conf = model.master_mol.GetConformer(0)
    rdMolTransforms.SetDihedralDeg(conf, a, i, j, b, target)

    # Round-trip
    agent.sync_angle_from_model(model.master_mol)
    assert abs(agent.bond.current_angle - target) < 0.01, (
        f"sync_from: expected {target:.3f}°, got {agent.bond.current_angle:.3f}°"
    )

    agent.sync_angle_to_model(model.master_mol)
    recovered = rdMolTransforms.GetDihedralDeg(conf, a, i, j, b)
    assert abs(recovered - target) < 0.01, (
        f"sync_to: expected {target:.3f}°, got {recovered:.3f}°"
    )


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    group1 = [
        ("path-3 coloring (2-chromatic)", test_path3_coloring),
        ("triangle K3 coloring (3-chromatic)", test_triangle_coloring),
        ("aspirin dependency graph coloring", test_aspirin_coloring),
        ("fuzz coloring (1000 random graphs)", test_fuzz_coloring),
    ]
    group2 = [
        ("aspirin has 3 rotatable bonds", test_aspirin_bond_count),
        ("aspirin dependency graph connected", test_dependency_graph_connected),
        ("bond detection counts (butane/pentane/aspirin/benzene)", test_bond_detection_counts),
        ("sync round-trip preserves angle ≤0.01°", test_sync_round_trip),
    ]

    print("=" * 60)
    print("Group 1 — Pure Python graph coloring")
    print("=" * 60)
    f1 = run_suite(group1)

    print()
    print("=" * 60)
    print("Group 2 — Molecular (RDKit)")
    print("=" * 60)
    f2 = run_suite(group2)

    total_fail = f1 + f2
    total_pass = (len(group1) + len(group2)) - total_fail

    print()
    print("=" * 60)
    print(f"Results: {total_pass} passed, {total_fail} failed")
    print("=" * 60)

    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()
