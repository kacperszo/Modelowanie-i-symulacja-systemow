"""
tests.py
--------
Unit and validation tests for Dihedral Agents.

Usage:
    uv run python src/tests.py

Test groups:
  1. Graph coloring correctness (pure Python, no RDKit)
  2. Bond dependency graph — aspirin has path topology Bond2-Bond3-Bond9
  3. Rotatable bond detection — aspirin has exactly 3 rotatable bonds
  4. Boltzmann validation on butane: simulation visits both gauche and anti regions
  5. Angle synchronization: sync_from/to keeps agent and master_mol consistent

Exit code: 0 if all tests pass, 1 if any fail.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    results.append(condition)
    return condition


# ── Test 1: Graph coloring (pure Python) ──────────────────────────────────────

print("\n=== Test 1: Graph coloring correctness ===")

from molecule import greedy_graph_coloring

def coloring_is_valid(graph, coloring):
    for node, neighbors in graph.items():
        for nbr in neighbors:
            if coloring.get(node) == coloring.get(nbr):
                return False, f"Bond {node} and Bond {nbr} share color {coloring[node]}"
    return True, f"{len(set(coloring.values()))} colors for {len(graph)} nodes"

# Path graph: 0-1-2 (aspirin-like topology)
path = {0: {1}, 1: {0, 2}, 2: {1}}
col = greedy_graph_coloring(path)
ok, detail = coloring_is_valid(path, col)
check("path 3-node: valid coloring", ok, detail)
check("path 3-node: 2-chromatic", len(set(col.values())) == 2, str(col))

# Complete graph K3 (triangle): needs 3 colors
k3 = {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}
col3 = greedy_graph_coloring(k3)
ok3, detail3 = coloring_is_valid(k3, col3)
check("triangle K3: valid coloring", ok3, detail3)
check("triangle K3: 3-chromatic", len(set(col3.values())) == 3, str(col3))

# Single node
check("single node: 1 color",
      len(set(greedy_graph_coloring({42: set()}).values())) == 1)

# Aspirin bond graph (path 2-3-9)
asp = {2: {3}, 3: {2, 9}, 9: {3}}
ca = greedy_graph_coloring(asp)
ok_a, detail_a = coloring_is_valid(asp, ca)
check("aspirin: valid coloring", ok_a, detail_a)
check("aspirin: 2-chromatic", len(set(ca.values())) == 2, str(ca))
check("aspirin: Bond2 and Bond9 same color (parallel group)",
      ca[2] == ca[9], f"Bond2={ca[2]}, Bond9={ca[9]}")

# Star graph: center must differ from all leaves; leaves can share color
star = {0: {1, 2, 3, 4}, 1: {0}, 2: {0}, 3: {0}, 4: {0}}
cs = greedy_graph_coloring(star)
check("star: center differs from all leaves",
      all(cs[0] != cs[i] for i in [1, 2, 3, 4]), str(cs))
check("star: 2-chromatic", len(set(cs.values())) == 2, str(cs))

# Linear chain 10 nodes: should be 2-chromatic (alternating 0,1,0,1,...)
chain = {i: {i - 1, i + 1} & set(range(10)) for i in range(10)}
cc = greedy_graph_coloring(chain)
ok_cc, detail_cc = coloring_is_valid(chain, cc)
check("chain 10-node: valid coloring", ok_cc, detail_cc)
check("chain 10-node: 2-chromatic", len(set(cc.values())) == 2, str(cc))

# Fuzz test: 1000 random graphs
import random
random.seed(42)
fuzz_ok = 0
for _ in range(1000):
    n = random.randint(2, 20)
    graph = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.25:
                graph[i].add(j)
                graph[j].add(i)
    col = greedy_graph_coloring(graph)
    if all(col.get(u) != col.get(v) for u, vs in graph.items() for v in vs):
        fuzz_ok += 1
check(f"fuzz test: 1000 random graphs valid",
      fuzz_ok == 1000, f"{fuzz_ok}/1000 valid")


# ── Tests 2–5: require RDKit ──────────────────────────────────────────────────

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolTransforms, rdForceFieldHelpers
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("\n  [SKIP] Tests 2-5 skipped: RDKit not available in this environment")
    print("         Run with: uv run python src/tests.py")

if RDKIT_AVAILABLE:
    from molecule import find_rotatable_bonds, build_dependency_graph, read_angles

    # ── Test 2: Bond dependency graph — Aspirin ────────────────────────────────

    print("\n=== Test 2: Bond dependency graph — Aspirin ===")

    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    mol_h = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())

    bonds = find_rotatable_bonds(mol_h)
    dep_graph = build_dependency_graph(bonds, mol_h)
    coloring = greedy_graph_coloring(dep_graph)
    bond_ids = sorted([b.bond_idx for b in bonds])

    check("aspirin: exactly 3 rotatable bonds",
          len(bonds) == 3, f"found: {len(bonds)} ({bond_ids})")

    def connected_components(graph):
        visited, components = set(), []
        for start in graph:
            if start not in visited:
                comp, stack = set(), [start]
                while stack:
                    node = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        comp.add(node)
                        stack.extend(graph[node] - visited)
                components.append(comp)
        return components

    components = connected_components(dep_graph)
    check("aspirin: dependency graph is connected (1 component)",
          len(components) == 1, f"components: {components}")
    check("aspirin: graph coloring is 2-chromatic",
          len(set(coloring.values())) == 2,
          f"used {len(set(coloring.values()))} colors")
    ok_col, detail_col = coloring_is_valid(dep_graph, coloring)
    check("aspirin: graph coloring is valid", ok_col, detail_col)

    # ── Test 3: Rotatable bond detection ──────────────────────────────────────

    print("\n=== Test 3: Rotatable bond detection ===")

    test_cases = [
        ("CCCC",                        1, "butane"),
        ("CCCCC",                       2, "pentane"),
        ("CC(=O)Oc1ccccc1C(=O)O",       3, "aspirin"),
        ("c1ccccc1",                     0, "benzene -- no rotatable bonds"),
        ("CC(=O)O",                      1, "acetic acid"),
        ("CCOCC",                        2, "diethyl ether"),
    ]

    for smiles, expected, name in test_cases:
        m = Chem.MolFromSmiles(smiles)
        m = Chem.AddHs(m)
        AllChem.EmbedMolecule(m, AllChem.ETKDGv3())
        b = find_rotatable_bonds(m)
        check(f"{name}: {expected} bond(s)",
              len(b) == expected, f"found {len(b)}")

    # ── Test 4: Boltzmann validation — butane gauche/anti ─────────────────────

    print("\n=== Test 4: Boltzmann validation — butane gauche/anti ===")
    print("  (long test, ~2000 MC steps)")

    from model import MoleculeModel

    model_butane = MoleculeModel(smiles="CCCC", n_steps=2000, temperature=500.0, seed=123)
    model_butane.run()

    agent = model_butane.schedule.agents[0]
    angles = np.array(agent.angle_history)

    # Acceptance rate: too low = step size too large, too high = too small
    acc = agent.acceptance_rate
    check("butane: acceptance rate in range 10-70%",
          0.10 < acc < 0.70, f"acc = {acc:.3f}")

    # Simulation must spend >5% of steps in both anti and gauche regions
    anti_frac   = np.mean(np.abs(np.abs(angles) - 180) < 40)
    gauche_frac = np.mean(np.abs(np.abs(angles) - 60)  < 40)

    visited_anti   = anti_frac   > 0.05
    visited_gauche = gauche_frac > 0.05

    check("butane: >5% of steps in anti region (|phi| ~ 180 deg)",
          visited_anti,   f"{anti_frac * 100:.1f}% of steps in anti")
    check("butane: >5% of steps in gauche region (|phi| ~ 60 deg)",
          visited_gauche, f"{gauche_frac * 100:.1f}% of steps in gauche")

    e_start = model_butane.energy_snapshots[0]
    e_end   = np.mean(model_butane.energy_snapshots[-10:])
    check("butane: final energy <= starting energy",
          e_end <= e_start + 0.5,
          f"E_start={e_start:.2f}, E_end={e_end:.2f} kcal/mol")

    # ── Test 5: Angle synchronization ─────────────────────────────────────────

    print("\n=== Test 5: Angle synchronization agent <-> master_mol ===")

    model_asp = MoleculeModel("CC(=O)Oc1ccccc1C(=O)O", n_steps=1, seed=42)
    agents_asp = model_asp.schedule.agents

    # After sync_from: agent angle == master_mol angle
    for agent in agents_asp:
        agent.sync_angle_from_model(model_asp.mol)
        conf = model_asp.mol.GetConformer(0)
        master_angle = rdMolTransforms.GetDihedralDeg(conf, *agent.bond.dihedral_atoms)
        check(f"Bond {agent.bond.bond_idx}: sync_from correct",
              abs(agent.angle - master_angle) < 0.01,
              f"agent={agent.angle:.2f} master={master_angle:.2f} deg")

    # Manual angle change + sync_to -> master_mol updated
    test_agent = agents_asp[0]
    test_angle = 42.0
    test_agent.bond.current_angle = test_angle
    conf_own = test_agent.mol.GetConformer(0)
    rdMolTransforms.SetDihedralDeg(conf_own, *test_agent.bond.dihedral_atoms, test_angle)
    test_agent.sync_angle_to_model(model_asp.mol)

    conf_master = model_asp.mol.GetConformer(0)
    master_after = rdMolTransforms.GetDihedralDeg(
        conf_master, *test_agent.bond.dihedral_atoms)
    check("sync_to writes angle to master_mol",
          abs(master_after - test_angle) < 0.01,
          f"wrote {test_angle} deg, read back {master_after:.2f} deg")


# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'=' * 50}")
n_pass  = sum(results)
n_total = len(results)
n_fail  = n_total - n_pass
print(f"Results: {n_pass}/{n_total} passed"
      + (f" -- {n_fail} FAILED" if n_fail else " -- all OK"))

sys.exit(0 if n_fail == 0 else 1)
