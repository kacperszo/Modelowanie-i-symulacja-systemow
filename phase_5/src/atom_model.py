"""
atom_model.py — AtomMoleculeModel: each atom_i of a rotatable bond is an agent.

Differences from MoleculeModel (bond-as-agent):
  - N_agents <= N_bonds (branch atoms own multiple bonds)
  - Communication by atom-to-atom Euclidean distance (not bond midpoints)
  - An agent selects which of its owned bonds to rotate each step
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from mesa import Model
from rdkit import Chem
from rdkit.Chem import AllChem, rdForceFieldHelpers, rdMolTransforms

from atom_agents import ATOM_AGENT_CLASSES, AtomAgent
from molecule import (
    build_dependency_graph,
    find_rotatable_bonds,
    greedy_graph_coloring,
    read_angles,
)
from molecule import RotatableBond


class AtomProximityScheduler:
    """Same protocol as ProximityScheduler but proximity = atom-to-atom distance."""

    def __init__(self, agents: List[AtomAgent], master_mol: Chem.Mol,
                 coloring: Dict[int, int], comm_cutoff: float) -> None:
        self.agents = agents
        self.master_mol = master_mol
        self.comm_cutoff = comm_cutoff
        colors = sorted(set(coloring.values()))
        self._color_groups: List[List[AtomAgent]] = [
            [a for a in agents if coloring[a.atom_idx] == c] for c in colors
        ]
        self.proximity_edge_history: List[int] = []

    def _compute_proximity(self) -> None:
        pos = self.master_mol.GetConformer(0).GetPositions()
        for agent in self.agents:
            agent.active_neighbors = []
        n_edges = 0
        for k, a in enumerate(self.agents):
            for b in self.agents[k + 1:]:
                if np.linalg.norm(pos[a.atom_idx] - pos[b.atom_idx]) <= self.comm_cutoff:
                    a.active_neighbors.append(b)
                    b.active_neighbors.append(a)
                    n_edges += 1
        self.proximity_edge_history.append(n_edges)

    def step(self) -> None:
        self._compute_proximity()
        for group in self._color_groups:
            for agent in group:
                agent.sync_from_model(self.master_mol)
            for agent in group:
                agent.step()
            for agent in group:
                agent.sync_to_model(self.master_mol)


class AtomMoleculeModel(Model):
    """ABM where each unique atom_i across rotatable bonds is one agent."""

    def __init__(self, smiles: str, strategy: str = "isolated",
                 init: str = "etkdg", comm_cutoff: float = 5.0,
                 n_steps: int = 1000, temperature: float = 300.0,
                 annealing: bool = False, T_start: float = 3000.0,
                 pre_minimize: bool = False, seed: int = 42) -> None:
        super().__init__(seed=seed)

        self.strategy = strategy
        self.init = init
        self.comm_cutoff = comm_cutoff
        self.n_steps = n_steps
        self.temperature = temperature
        self.annealing = annealing
        self.T_start = T_start if annealing else temperature
        self.current_temperature: float = T_start if annealing else temperature

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Cannot parse SMILES: {smiles!r}")
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        if AllChem.EmbedMolecule(mol_h, params) == -1:
            AllChem.EmbedMolecule(mol_h)
        self.master_mol: Chem.RWMol = Chem.RWMol(mol_h)

        self.pre_minimize = pre_minimize
        if pre_minimize:
            _fp = rdForceFieldHelpers.MMFFGetMoleculeProperties(self.master_mol)
            _ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(self.master_mol, _fp)
            _ff.Minimize(maxIts=2000)

        mol_noh = Chem.RemoveHs(mol_h)
        self.bonds: List[RotatableBond] = find_rotatable_bonds(mol_h)
        if not self.bonds:
            raise ValueError(f"No rotatable bonds in {smiles!r}")

        conf = self.master_mol.GetConformer(0)
        if init == "etkdg":
            read_angles(self.master_mol, self.bonds)
        elif init == "random":
            for b in self.bonds:
                a, i, j, bk = b.dihedral_atoms
                rdMolTransforms.SetDihedralDeg(conf, a, i, j, bk,
                                               float(self.rng.uniform(-180., 180.)))
            read_angles(self.master_mol, self.bonds)
        elif init == "zeros":
            for b in self.bonds:
                a, i, j, bk = b.dihedral_atoms
                rdMolTransforms.SetDihedralDeg(conf, a, i, j, bk, 0.)
            read_angles(self.master_mol, self.bonds)
        elif init == "anti":
            for b in self.bonds:
                a, i, j, bk = b.dihedral_atoms
                rdMolTransforms.SetDihedralDeg(conf, a, i, j, bk, 180.)
            read_angles(self.master_mol, self.bonds)
        else:
            raise ValueError(f"Unknown init: {init!r}")

        # Group bonds by atom_i — each unique atom_i becomes one agent
        atom_to_bonds: Dict[int, List[RotatableBond]] = {}
        for bond in self.bonds:
            atom_to_bonds.setdefault(bond.atom_i, []).append(bond)

        # Atom dependency graph derived from bond dependency graph
        bond_dep = build_dependency_graph(self.bonds, mol_noh)
        atom_dep: Dict[int, set] = {a: set() for a in atom_to_bonds}
        for atom_a, bonds_a in atom_to_bonds.items():
            for atom_b, bonds_b in atom_to_bonds.items():
                if atom_a == atom_b:
                    continue
                for ba in bonds_a:
                    for bb in bonds_b:
                        if bb.bond_idx in bond_dep.get(ba.bond_idx, set()):
                            atom_dep[atom_a].add(atom_b)
                            break
                    else:
                        continue
                    break

        self.coloring: Dict[int, int] = greedy_graph_coloring(atom_dep)

        agent_cls = ATOM_AGENT_CLASSES[strategy]
        self.atom_agents: List[AtomAgent] = [
            agent_cls(self, atom_idx, bonds, self.master_mol, temperature)
            for atom_idx, bonds in atom_to_bonds.items()
        ]

        self.scheduler = AtomProximityScheduler(
            self.atom_agents, self.master_mol, self.coloring, comm_cutoff
        )

        self._master_ff_props = rdForceFieldHelpers.MMFFGetMoleculeProperties(self.master_mol)
        self.energy_snapshots: List[float] = [self._master_energy()]
        self.current_step: int = 0

    def _master_energy(self) -> float:
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(
            self.master_mol, self._master_ff_props)
        return ff.CalcEnergy() if ff else 0.0

    def step(self) -> None:
        if self.annealing and self.n_steps > 0:
            progress = self.current_step / self.n_steps
            self.current_temperature = self.T_start * (
                self.temperature / self.T_start) ** progress
        self.scheduler.step()
        self.current_step += 1
        self.energy_snapshots.append(self._master_energy())

    def run(self) -> None:
        for _ in range(self.n_steps):
            self.step()

    def summary(self) -> dict:
        snaps = self.energy_snapshots
        final_e = snaps[-1]
        initial_e = snaps[0]

        converge_step = self.n_steps
        for idx, e in enumerate(snaps):
            if e <= final_e + 0.5:
                converge_step = idx
                break

        coverages = []
        for agent in self.atom_agents:
            for bond in agent.owned_bonds:
                hist = agent.angle_histories.get(bond.bond_idx, [])
                if hist:
                    bins = {int((a + 180.) / 10.) % 36 for a in hist}
                    coverages.append(len(bins) / 36.)
        mean_coverage = float(np.mean(coverages)) if coverages else 0.

        acc_rates = [a.acceptance_rate() for a in self.atom_agents]
        mean_acceptance = float(np.mean(acc_rates)) if acc_rates else 0.

        edges = self.scheduler.proximity_edge_history
        mean_neighbors = float(np.mean(edges)) if edges else 0.

        return {
            "strategy": self.strategy,
            "init": self.init,
            "comm_cutoff": self.comm_cutoff,
            "n_agents": len(self.atom_agents),
            "n_bonds": len(self.bonds),
            "final_energy": final_e,
            "energy_drop": initial_e - final_e,
            "converge_step": converge_step,
            "mean_coverage": mean_coverage,
            "mean_acceptance": mean_acceptance,
            "mean_neighbors": mean_neighbors,
        }
