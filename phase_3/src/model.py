"""
model.py — MoleculeModel (Mesa Model subclass) and ProximityScheduler.

ProximityScheduler computes a dynamic communication graph based on bond
midpoint distances in the *current* master_mol conformer, then steps agents
in graph-coloring order so each color group can be treated as a parallel
update batch.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from mesa import Model
from rdkit import Chem
from rdkit.Chem import AllChem, rdForceFieldHelpers, rdMolTransforms

from agents import AGENT_CLASSES, BondAgent
from molecule import (
    RotatableBond,
    build_dependency_graph,
    find_rotatable_bonds,
    greedy_graph_coloring,
    read_angles,
)

INITIALIZERS = ["etkdg", "random", "zeros", "anti"]


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------


def init_etkdg(mol: Chem.Mol, bonds: List[RotatableBond]) -> None:
    """Keep the ETKDG conformer as-is; just read angles into bond objects."""
    read_angles(mol, bonds, conf_id=0)


def init_random(
    mol: Chem.Mol, bonds: List[RotatableBond], rng: np.random.Generator
) -> None:
    """Randomise all rotatable dihedrals uniformly in [-180, 180)."""
    conf = mol.GetConformer(0)
    for b in bonds:
        a, i, j, bk = b.dihedral_atoms
        rdMolTransforms.SetDihedralDeg(conf, a, i, j, bk, float(rng.uniform(-180.0, 180.0)))
    read_angles(mol, bonds, conf_id=0)


def init_zeros(mol: Chem.Mol, bonds: List[RotatableBond]) -> None:
    """Set all rotatable dihedrals to 0°."""
    conf = mol.GetConformer(0)
    for b in bonds:
        a, i, j, bk = b.dihedral_atoms
        rdMolTransforms.SetDihedralDeg(conf, a, i, j, bk, 0.0)
    read_angles(mol, bonds, conf_id=0)


def init_anti(mol: Chem.Mol, bonds: List[RotatableBond]) -> None:
    """Set all rotatable dihedrals to 180° (anti conformation)."""
    conf = mol.GetConformer(0)
    for b in bonds:
        a, i, j, bk = b.dihedral_atoms
        rdMolTransforms.SetDihedralDeg(conf, a, i, j, bk, 180.0)
    read_angles(mol, bonds, conf_id=0)


# ---------------------------------------------------------------------------
# Proximity Scheduler
# ---------------------------------------------------------------------------


class ProximityScheduler:
    """
    Manages the dynamic communication topology and step protocol.

    Before every full step:
      1. Compute pairwise distances between bond midpoints in master_mol.
      2. Populate agent.active_neighbors for all pairs within comm_cutoff Å.

    For each graph-coloring group (in colour order):
      a. sync_angle_from_model  — all agents read the current master state.
      b. agent.step()           — each agent decides independently.
      c. sync_angle_to_model    — accepted angles are written back to master.

    The proximity graph is recomputed once per full scheduler step (before all
    colour groups), so it reflects the conformation at the START of each step.
    """

    def __init__(
        self,
        agents: List[BondAgent],
        master_mol: Chem.Mol,
        coloring: Dict[int, int],
        comm_cutoff: float,
    ) -> None:
        self.agents = agents
        self.master_mol = master_mol
        self.comm_cutoff = comm_cutoff
        self.coloring = coloring

        # Pre-build colour groups (sorted for determinism)
        colors = sorted(set(coloring.values()))
        self._color_groups: List[List[BondAgent]] = [
            [a for a in agents if coloring[a.bond.bond_idx] == c]
            for c in colors
        ]

        self.proximity_edge_history: List[int] = []

    # ------------------------------------------------------------------
    # Proximity computation
    # ------------------------------------------------------------------

    def _compute_proximity(self) -> None:
        """Populate active_neighbors from current master_mol geometry."""
        conf = self.master_mol.GetConformer(0)
        pos = conf.GetPositions()

        midpoints: Dict[int, np.ndarray] = {}
        for agent in self.agents:
            mi = pos[agent.bond.atom_i]
            mj = pos[agent.bond.atom_j]
            midpoints[agent.unique_id] = (mi + mj) * 0.5

        for agent in self.agents:
            agent.active_neighbors = []

        n_edges = 0
        for k, agent_a in enumerate(self.agents):
            for agent_b in self.agents[k + 1 :]:
                d = float(
                    np.linalg.norm(
                        midpoints[agent_a.unique_id] - midpoints[agent_b.unique_id]
                    )
                )
                if d <= self.comm_cutoff:
                    agent_a.active_neighbors.append(agent_b)
                    agent_b.active_neighbors.append(agent_a)
                    n_edges += 1

        self.proximity_edge_history.append(n_edges)

    # ------------------------------------------------------------------
    # Full step
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Execute one complete simulation step across all colour groups."""
        # Step 1: build proximity graph from current master_mol state
        self._compute_proximity()

        # Step 2: colour groups in order
        for group in self._color_groups:
            # (a) Sync all agents in group from master
            for agent in group:
                agent.sync_angle_from_model(self.master_mol)

            # (b) Each agent decides
            for agent in group:
                agent.step()

            # (c) Write accepted angles back to master
            for agent in group:
                agent.sync_angle_to_model(self.master_mol)


# ---------------------------------------------------------------------------
# Molecule Model
# ---------------------------------------------------------------------------


class MoleculeModel(Model):
    """
    ABM of a small organic molecule where each rotatable bond is an agent.

    Parameters
    ----------
    smiles      : SMILES string of the molecule
    strategy    : agent decision strategy (see agents.AGENT_CLASSES)
    init        : initial conformation ('etkdg', 'random', 'zeros', 'anti')
    comm_cutoff : communication radius in Å (bond midpoint distance)
    n_steps     : number of simulation steps to run
    temperature : Metropolis temperature in K
    seed        : RNG seed for reproducibility
    """

    def __init__(
        self,
        smiles: str,
        strategy: str = "isolated",
        init: str = "etkdg",
        comm_cutoff: float = 4.0,
        n_steps: int = 1000,
        temperature: float = 300.0,
        annealing: bool = False,
        T_start: float = 3000.0,
        pre_minimize: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__(seed=seed)

        self.strategy = strategy
        self.init = init
        self.comm_cutoff = comm_cutoff
        self.n_steps = n_steps
        self.temperature = temperature

        # Simulated annealing: geometric cooling from T_start → temperature
        self.annealing = annealing
        self.T_start = T_start if annealing else temperature
        self.current_temperature: float = T_start if annealing else temperature

        # ------------------------------------------------------------------
        # Build and embed molecule
        # ------------------------------------------------------------------
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Cannot parse SMILES: {smiles!r}")
        mol_h = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        rc = AllChem.EmbedMolecule(mol_h, params)
        if rc == -1:
            # Fallback: distance-geometry without random seed
            AllChem.EmbedMolecule(mol_h)

        self.master_mol: Chem.RWMol = Chem.RWMol(mol_h)

        # ------------------------------------------------------------------
        # Optional: MMFF94 L-BFGS pre-minimisation
        # Fixes bond-length and angle contributions before agents start.
        # Agents then explore *dihedral* space from a proper local minimum.
        # ------------------------------------------------------------------
        self.pre_minimize = pre_minimize
        self.e_after_premin: float = 0.0
        if pre_minimize:
            _fp = rdForceFieldHelpers.MMFFGetMoleculeProperties(self.master_mol)
            _ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(self.master_mol, _fp)
            _ff.Minimize(maxIts=2000)
            self.e_after_premin = _ff.CalcEnergy()

        # ------------------------------------------------------------------
        # Detect rotatable bonds (SMARTS on H-stripped mol)
        # ------------------------------------------------------------------
        mol_noh = Chem.RemoveHs(mol_h)
        self.bonds: List[RotatableBond] = find_rotatable_bonds(mol_h)
        if not self.bonds:
            raise ValueError(f"No rotatable bonds found in {smiles!r}")

        # ------------------------------------------------------------------
        # Apply init transform to master_mol
        # ------------------------------------------------------------------
        if init == "etkdg":
            init_etkdg(self.master_mol, self.bonds)
        elif init == "random":
            init_random(self.master_mol, self.bonds, self.rng)
        elif init == "zeros":
            init_zeros(self.master_mol, self.bonds)
        elif init == "anti":
            init_anti(self.master_mol, self.bonds)
        else:
            raise ValueError(f"Unknown init: {init!r}")

        # ------------------------------------------------------------------
        # Dependency graph + coloring
        # ------------------------------------------------------------------
        dep_graph = build_dependency_graph(self.bonds, mol_noh)
        self.coloring: Dict[int, int] = greedy_graph_coloring(dep_graph)

        # ------------------------------------------------------------------
        # Create agents (one per rotatable bond)
        # ------------------------------------------------------------------
        agent_cls = AGENT_CLASSES[strategy]
        self.bond_agents: List[BondAgent] = []
        for bond in self.bonds:
            agent = agent_cls(self, bond, self.master_mol, temperature)
            self.bond_agents.append(agent)

        # ------------------------------------------------------------------
        # Proximity scheduler
        # ------------------------------------------------------------------
        self.scheduler = ProximityScheduler(
            self.bond_agents, self.master_mol, self.coloring, comm_cutoff
        )

        # ------------------------------------------------------------------
        # Master force field (energy snapshots only — not used by agents)
        # ------------------------------------------------------------------
        self._master_ff_props = rdForceFieldHelpers.MMFFGetMoleculeProperties(
            self.master_mol
        )
        self.energy_snapshots: List[float] = [self._master_energy()]
        self.current_step: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _master_energy(self) -> float:
        """Total MMFF94 energy of master_mol (current accepted conformation)."""
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(
            self.master_mol, self._master_ff_props
        )
        if ff is None:
            return 0.0
        return ff.CalcEnergy()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance the simulation by one step."""
        # Update temperature before agents act
        if self.annealing and self.n_steps > 0:
            progress = self.current_step / self.n_steps
            self.current_temperature = self.T_start * (
                self.temperature / self.T_start
            ) ** progress
        self.scheduler.step()
        self.current_step += 1
        self.energy_snapshots.append(self._master_energy())

    def run(self) -> None:
        """Run for self.n_steps steps."""
        for _ in range(self.n_steps):
            self.step()

    def summary(self) -> dict:
        """
        Return a summary dictionary with key metrics.

        Keys
        ----
        strategy, init, comm_cutoff : experiment identifiers
        final_energy    : last recorded total energy (kcal/mol)
        energy_drop     : initial_energy - final_energy
        converge_step   : first step where E <= final_E + 0.5 kcal/mol
        mean_coverage   : mean fraction of [-180, 180] explored (10-deg bins)
        mean_acceptance : mean acceptance rate across all agents
        mean_neighbors  : mean number of active proximity edges per step
        """
        snaps = self.energy_snapshots
        initial_e = snaps[0]
        final_e = snaps[-1]

        # Convergence step: first snapshot index where E <= final + 0.5
        converge_step = self.n_steps
        for idx, e in enumerate(snaps):
            if e <= final_e + 0.5:
                converge_step = idx  # snapshot at every step
                break

        # Angular coverage per agent (fraction of 36 bins of width 10°)
        coverages = []
        for agent in self.bond_agents:
            if agent.angle_history:
                bins = {int((a + 180.0) / 10.0) % 36 for a in agent.angle_history}
                coverages.append(len(bins) / 36.0)
        mean_coverage = float(np.mean(coverages)) if coverages else 0.0

        # Mean acceptance rate
        acc_rates = [a.acceptance_rate() for a in self.bond_agents]
        mean_acceptance = float(np.mean(acc_rates)) if acc_rates else 0.0

        # Mean proximity edges per step
        edges = self.scheduler.proximity_edge_history
        mean_neighbors = float(np.mean(edges)) if edges else 0.0

        return {
            "strategy": self.strategy,
            "init": self.init,
            "comm_cutoff": self.comm_cutoff,
            "annealing": self.annealing,
            "pre_minimize": self.pre_minimize,
            "final_energy": final_e,
            "energy_drop": initial_e - final_e,
            "converge_step": converge_step,
            "mean_coverage": mean_coverage,
            "mean_acceptance": mean_acceptance,
            "mean_neighbors": mean_neighbors,
        }
