"""
model.py — Mesa 3.x, MMFF94 force field
"""

from typing import Dict, List

import numpy as np
from mesa import Model
from rdkit import Chem
from rdkit.Chem import AllChem, rdForceFieldHelpers

from agents import BondAgent
from molecule import (
    build_dependency_graph,
    find_rotatable_bonds,
    greedy_graph_coloring,
    read_angles,
)


class GraphColoringScheduler:
    def __init__(self, coloring: Dict[int, int]):
        self.coloring = coloring
        self._agents: List = []
        self._color_groups: Dict[int, List] = {}
        self.steps = 0

    def add(self, agent):
        self._agents.append(agent)
        color = self.coloring.get(agent.bond.bond_idx, 0)
        self._color_groups.setdefault(color, []).append(agent)

    @property
    def agents(self):
        return self._agents

    def step(self):
        for color in sorted(self._color_groups.keys()):
            for agent in self._color_groups[color]:
                agent.step()
        self.steps += 1


class MoleculeModel(Model):
    def __init__(
        self,
        smiles: str,
        n_steps: int = 500,
        temperature: float = 300.0,
        scheduler_type: str = "graph_coloring",
        seed: int = 42,
    ):
        super().__init__(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.n_steps = n_steps
        self.temperature = temperature
        self.step_count = 0

        # Cząsteczka + konformacja startowa
        self.mol = Chem.MolFromSmiles(smiles)
        if self.mol is None:
            raise ValueError(f"Nieprawidłowy SMILES: {smiles}")
        self.mol = Chem.AddHs(self.mol)

        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        result = AllChem.EmbedMolecule(self.mol, params)
        if result == -1:
            raise ValueError("EmbedMolecule failed")

        # MMFF94 force field — tworzony raz, współdzielony przez agentów
        ff_props = rdForceFieldHelpers.MMFFGetMoleculeProperties(self.mol)
        if ff_props is None:
            raise ValueError("MMFF nie obsługuje tej cząsteczki")
        self.ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(self.mol, ff_props)

        # Graf wiązań + kolorowanie
        self.bonds = find_rotatable_bonds(self.mol)
        if not self.bonds:
            raise ValueError("Brak obracalnych wiązań!")

        self.dep_graph = build_dependency_graph(self.bonds, self.mol)
        self.coloring = greedy_graph_coloring(self.dep_graph)

        n_colors = len(set(self.coloring.values()))
        print(f"Cząsteczka: {smiles}")
        print(f"Obracalne wiązania: {len(self.bonds)}")
        print(
            f"Kolorowanie grafowe: {n_colors} kolorów → {n_colors} rund/krok zamiast {len(self.bonds)}"
        )
        print(f"  kolorowanie: {self.coloring}")
        print(f"Temperatura: {temperature} K")
        print()

        read_angles(self.mol, self.bonds, conf_id=0)

        # Agenci
        self.schedule = GraphColoringScheduler(self.coloring)
        for i, bond in enumerate(self.bonds):
            agent = BondAgent(
                unique_id=i,
                model=self,
                bond=bond,
                mol=self.mol,
                ff=self.ff,
                temperature=temperature,
            )
            self.schedule.add(agent)

        self.energy_snapshots = []
        self.angle_snapshots = []

    def step(self):
        self.schedule.step()
        self.step_count += 1
        if self.step_count % 10 == 0:
            self.energy_snapshots.append(self.ff.CalcEnergy())
            self.angle_snapshots.append([a.angle for a in self.schedule.agents])

    def run(self):
        print(f"Uruchamiam {self.n_steps} kroków...")
        for i in range(self.n_steps):
            self.step()
            if (i + 1) % 100 == 0:
                e = self.ff.CalcEnergy()
                print(f"  krok {i+1}/{self.n_steps}  E={e:.2f} kcal/mol")

        print("\n=== Wyniki ===")
        for agent in self.schedule.agents:
            print(
                f"Bond {agent.bond.bond_idx} "
                f"(atomy {agent.bond.atom_i}-{agent.bond.atom_j}): "
                f"φ={agent.angle:.1f}°  acc={agent.acceptance_rate:.2f}"
            )

    def get_final_angles(self) -> List[float]:
        return [a.angle for a in self.schedule.agents]
