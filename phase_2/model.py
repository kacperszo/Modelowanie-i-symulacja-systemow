"""
model.py
--------
MoleculeModel: Mesa model representing a single molecule as a collection
of BondAgents updated by a graph-coloring-based scheduler.

Key CS contribution: GraphColoringScheduler implements parallel update
semantics through a three-phase sync protocol:
  1. sync_from: all agents in a color group read state from master_mol
  2. step:      each agent runs Metropolis on its own private mol copy
  3. sync_to:   accepted angles are written back to master_mol

Bonds of the same color form an independent set in the dependency graph,
so their updates do not interfere — correct parallel semantics even in
the current sequential implementation. ThreadPoolExecutor in Stage 4.

Known limitation: sync_angle_from_model copies only the agent's own
dihedral angle, not the full 3D geometry. For molecules with many bonds
updated across multiple rounds, the private copy geometry may drift
slightly from master_mol in degrees of freedom not owned by this agent.
Effect is negligible for aspirin (3 bonds, 2 rounds) — open issue for Stage 4.
"""

import numpy as np
from mesa import Model
from typing import List, Dict
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import AllChem, rdForceFieldHelpers

from molecule import (
    find_rotatable_bonds, build_dependency_graph,
    greedy_graph_coloring, read_angles
)
from agents import BondAgent


class GraphColoringScheduler:
    """
    Custom Mesa scheduler — pure Python, no dependency on mesa.time
    (removed in Mesa 3.x).

    One model step = one pass over all color groups:
      for each color c in ascending order:
        phase 1 — sync_from: agents read current state from master_mol
        phase 2 — step:      each agent runs Metropolis on its private copy
        phase 3 — sync_to:   accepted angles written back to master_mol
    """

    def __init__(self, coloring: Dict[int, int], master_mol):
        self.coloring = coloring
        self.master_mol = master_mol
        self._agents: List = []
        self._color_groups: Dict[int, List] = defaultdict(list)
        self.steps = 0

    def add(self, agent):
        self._agents.append(agent)
        color = self.coloring.get(agent.bond.bond_idx, 0)
        self._color_groups[color].append(agent)

    @property
    def agents(self):
        return self._agents

    def step(self):
        for color in sorted(self._color_groups.keys()):
            group = self._color_groups[color]

            # Phase 1: all agents in group read current state
            for agent in group:
                agent.sync_angle_from_model(self.master_mol)

            # Phase 2: each agent runs Metropolis on its private mol copy
            for agent in group:
                agent.step()

            # Phase 3: accepted angles written back to master_mol
            for agent in group:
                agent.sync_angle_to_model(self.master_mol)

        self.steps += 1


class MoleculeModel(Model):
    """
    ABM of a single small molecule.

    Parameters
    ----------
    smiles : str
        SMILES string of the molecule.
    n_steps : int
        Number of simulation steps.
    temperature : float
        Simulation temperature in Kelvin. Controls Metropolis acceptance:
        low T = exploitation (stays near minima),
        high T = exploration (crosses barriers).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        smiles: str,
        n_steps: int = 500,
        temperature: float = 300.0,
        seed: int = 42,
    ):
        super().__init__(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.n_steps = n_steps
        self.temperature = temperature
        self.step_count = 0

        # Build molecule and generate starting conformer with ETKDG
        self.mol = Chem.MolFromSmiles(smiles)
        if self.mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        self.mol = Chem.AddHs(self.mol)

        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        if AllChem.EmbedMolecule(self.mol, params) == -1:
            raise ValueError("EmbedMolecule failed — molecule may be too strained")

        # Build dependency graph and compute graph coloring
        self.bonds = find_rotatable_bonds(self.mol)
        if not self.bonds:
            raise ValueError("Molecule has no rotatable bonds")

        self.dep_graph = build_dependency_graph(self.bonds, self.mol)
        self.coloring = greedy_graph_coloring(self.dep_graph)

        n_colors = len(set(self.coloring.values()))
        print(f"Molecule:          {smiles}")
        print(f"Rotatable bonds:   {len(self.bonds)}")
        print(f"Graph coloring:    {n_colors} colors "
              f"-> {n_colors} rounds/step vs {len(self.bonds)} sequential")
        print(f"  coloring: {self.coloring}")
        print(f"Temperature:       {temperature} K  "
              f"(kT = {0.001987 * temperature:.4f} kcal/mol)")

        read_angles(self.mol, self.bonds, conf_id=0)

        # Create agents — each gets mol_template and builds its own private copy
        self.schedule = GraphColoringScheduler(self.coloring, self.mol)
        for i, bond in enumerate(self.bonds):
            agent = BondAgent(
                unique_id=i,
                model=self,
                bond=bond,
                mol_template=self.mol,
                temperature=temperature,
            )
            self.schedule.add(agent)

        # Master force field for monitoring total energy (read-only)
        ff_props = rdForceFieldHelpers.MMFFGetMoleculeProperties(self.mol)
        self._master_ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(self.mol, ff_props)

        self.energy_snapshots: list = []
        self.angle_snapshots: list = []

    def step(self):
        self.schedule.step()
        self.step_count += 1
        if self.step_count % 10 == 0:
            self.energy_snapshots.append(self._master_ff.CalcEnergy())
            self.angle_snapshots.append([a.angle for a in self.schedule.agents])

    def run(self):
        print(f"\nRunning {self.n_steps} steps...")
        for i in range(self.n_steps):
            self.step()
            if (i + 1) % 100 == 0:
                e = self._master_ff.CalcEnergy()
                rates = [f"{a.acceptance_rate:.2f}" for a in self.schedule.agents]
                print(f"  step {i+1:4d}  E={e:.2f} kcal/mol  acc={rates}")

        print("\n=== Final state ===")
        for agent in self.schedule.agents:
            print(f"  Bond {agent.bond.bond_idx}: "
                  f"phi={agent.angle:.1f} deg  "
                  f"acceptance={agent.acceptance_rate:.2f}")

    def get_final_angles(self) -> List[float]:
        return [a.angle for a in self.schedule.agents]
