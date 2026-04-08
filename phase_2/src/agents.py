"""
agents.py v3
------------
Poprawka: usunięto ff.Initialize() po każdym kroku.
ff.CalcEnergy() działa bezpośrednio na aktualnej geometrii
konformacji — Initialize() był niepotrzebny i powodował
artefakty w trajektorii.
"""

import numpy as np
from mesa import Agent
from rdkit.Chem import rdMolTransforms


class BondAgent(Agent):
    def __init__(self, unique_id, model, bond, mol, ff, temperature=300.0):
        super().__init__(model)
        self.unique_id = unique_id
        self.bond = bond
        self.mol = mol
        self.ff = ff
        self.temperature = temperature
        self.step_size = 15.0
        self.energy_history = []
        self.angle_history = []
        self._accepted = 0
        self._proposed = 0

    @property
    def angle(self) -> float:
        return self.bond.current_angle

    def local_energy(self, angle: float) -> float:
        return self.ff.CalcEnergy()

    def step(self):
        conf = self.mol.GetConformer(0)
        current_angle = self.angle
        current_e = self.ff.CalcEnergy()

        # Propozycja
        delta = self.model.rng.normal(0, self.step_size)
        proposed_angle = ((current_angle + delta) + 180) % 360 - 180

        # Ustaw proponowany kąt
        rdMolTransforms.SetDihedralDeg(conf, *self.bond.dihedral_atoms, proposed_angle)
        proposed_e = self.ff.CalcEnergy()

        delta_e = proposed_e - current_e
        kT = 0.001987 * self.temperature

        if delta_e < 0 or self.model.rng.random() < np.exp(-delta_e / kT):
            self.bond.current_angle = proposed_angle
            self._accepted += 1
        else:
            # Odrzuć — przywróć
            rdMolTransforms.SetDihedralDeg(
                conf, *self.bond.dihedral_atoms, current_angle
            )
            self.bond.current_angle = current_angle

        self._proposed += 1
        self.energy_history.append(self.ff.CalcEnergy())
        self.angle_history.append(self.bond.current_angle)

    @property
    def acceptance_rate(self) -> float:
        if self._proposed == 0:
            return 0.0
        return self._accepted / self._proposed
