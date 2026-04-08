"""
agents.py
---------
BondAgent: one rotatable bond as a Mesa 3.x agent.

Each agent owns a private copy of the molecule and force field.
This eliminates race conditions when bonds in the same color group
are updated in parallel (ThreadPoolExecutor in Stage 4).

Memory cost: N copies of the mol object (~tens of kB each for drug-like molecules).
Acceptable for typical ligands (<100 heavy atoms).
"""

import numpy as np
from mesa import Agent
from rdkit import Chem
from rdkit.Chem import rdMolTransforms, rdForceFieldHelpers


class BondAgent(Agent):
    """
    Agent representing a single rotatable bond.

    State:  dihedral angle phi in [-180, 180] degrees
    Action: random perturbation of phi + Metropolis acceptance criterion

    The agent evaluates moves using the full MMFF94 force field energy
    of the whole molecule — not a local approximation. This is chemically
    correct: dihedral energy is inherently non-local (van der Waals and
    electrostatics couple all non-bonded atom pairs).
    """

    def __init__(self, unique_id, model, bond, mol_template, temperature=300.0):
        super().__init__(model)
        self.unique_id = unique_id
        self.bond = bond
        self.temperature = temperature
        self.step_size = 15.0   # degrees — perturbation std dev (sigma)

        # Private RNG seeded from model seed + unique_id — thread-safe for Stage 4
        self._rng = np.random.default_rng(model.rng.integers(2**31) + unique_id)

        # Private molecule copy and force field — no shared mutable state
        self.mol = Chem.RWMol(mol_template)
        ff_props = rdForceFieldHelpers.MMFFGetMoleculeProperties(self.mol)
        if ff_props is None:
            raise ValueError(f"MMFF94 does not support agent {unique_id}")
        self.ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(self.mol, ff_props)

        self.energy_history: list = []
        self.angle_history: list = []
        self._accepted = 0
        self._proposed = 0

    @property
    def angle(self) -> float:
        return self.bond.current_angle

    def get_energy(self) -> float:
        """MMFF94 energy of the current geometry of this agent's private mol copy."""
        return self.ff.CalcEnergy()

    def energy_at_angle(self, angle: float) -> float:
        """
        MMFF94 energy when this bond's dihedral is set to `angle`.
        Temporarily sets the angle, measures energy, then restores.
        Used for energy profile plots (visualize_mol.py).
        """
        conf = self.mol.GetConformer(0)
        current = rdMolTransforms.GetDihedralDeg(conf, *self.bond.dihedral_atoms)
        rdMolTransforms.SetDihedralDeg(conf, *self.bond.dihedral_atoms, angle)
        e = self.ff.CalcEnergy()
        rdMolTransforms.SetDihedralDeg(conf, *self.bond.dihedral_atoms, current)
        return e

    def sync_angle_from_model(self, master_mol: Chem.Mol):
        """
        Copy this bond's dihedral angle from master_mol into this agent's private mol.
        Called by the scheduler at the start of each color-group round.
        Ensures all agents in a group start from the same consistent state.
        """
        conf_master = master_mol.GetConformer(0)
        conf_own = self.mol.GetConformer(0)
        angle = rdMolTransforms.GetDihedralDeg(conf_master, *self.bond.dihedral_atoms)
        rdMolTransforms.SetDihedralDeg(conf_own, *self.bond.dihedral_atoms, angle)
        self.bond.current_angle = angle

    def sync_angle_to_model(self, master_mol: Chem.Mol):
        """
        Write the accepted angle back to master_mol.
        Called by the scheduler after each color-group round completes.
        """
        conf_master = master_mol.GetConformer(0)
        rdMolTransforms.SetDihedralDeg(
            conf_master, *self.bond.dihedral_atoms, self.bond.current_angle
        )

    def step(self):
        """
        One Metropolis step on this agent's private mol copy.

        1. Read current energy E_curr
        2. Propose phi' = phi + delta, delta ~ N(0, step_size)
        3. Compute E_proposed
        4. Accept with probability min(1, exp(-dE/kT))
        5. Revert if rejected
        """
        conf = self.mol.GetConformer(0)
        current_angle = self.angle
        current_e = self.get_energy()

        delta = self._rng.normal(0, self.step_size)
        proposed_angle = ((current_angle + delta) + 180) % 360 - 180

        rdMolTransforms.SetDihedralDeg(conf, *self.bond.dihedral_atoms, proposed_angle)
        proposed_e = self.get_energy()

        delta_e = proposed_e - current_e
        kT = 0.001987 * self.temperature   # kcal/mol

        if delta_e < 0 or self._rng.random() < np.exp(-delta_e / kT):
            self.bond.current_angle = proposed_angle
            self._accepted += 1
        else:
            # Revert to previous angle
            rdMolTransforms.SetDihedralDeg(conf, *self.bond.dihedral_atoms, current_angle)
            self.bond.current_angle = current_angle

        self._proposed += 1
        self.energy_history.append(self.get_energy())
        self.angle_history.append(self.bond.current_angle)

    @property
    def acceptance_rate(self) -> float:
        if self._proposed == 0:
            return 0.0
        return self._accepted / self._proposed
