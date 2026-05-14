"""
pain_model.py — Pain-signal ABM of molecular conformational dynamics.

No Metropolis, no energy evaluation during simulation.

Architecture:
  - Every heavy atom is a pain sensor: pain_i = sum_j max(0, r_pain - d_ij)
  - Every atom_i of a rotatable bond is a rotor
  - Each step, each rotor tries +step_size and -step_size for each owned bond
  - Weighted pain = sum_i pain_i * exp(-pain_decay * d(atom_i, bond_midpoint))
  - Rotate in the direction that reduces weighted pain by more than vote_threshold
  - MMFF94 energy recorded every step for analysis only

Parameters (PainParams):
  r_pain          Å     distance below which atom-pair contributes pain
  pain_decay      Å⁻¹   decay rate of pain signal with distance from bond
  step_size       deg   rotation per step
  vote_threshold        minimum weighted-pain reduction to trigger rotation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from mesa import Model
from rdkit import Chem
from rdkit.Chem import AllChem, rdForceFieldHelpers, rdMolTransforms

from molecule import RotatableBond, find_rotatable_bonds, read_angles


@dataclass
class PainParams:
    r_pain: float = 4.5
    pain_decay: float = 0.5
    step_size: float = 5.0
    vote_threshold: float = 0.005

    def to_array(self) -> np.ndarray:
        return np.array([self.r_pain, self.pain_decay,
                         self.step_size, self.vote_threshold])

    @classmethod
    def from_array(cls, arr) -> "PainParams":
        return cls(r_pain=float(arr[0]), pain_decay=float(arr[1]),
                   step_size=float(arr[2]), vote_threshold=float(arr[3]))

    def __str__(self) -> str:
        return (f"r={self.r_pain:.2f} dec={self.pain_decay:.2f} "
                f"step={self.step_size:.2f} thr={self.vote_threshold:.3f}")


# Parameter bounds for optimisation
PARAM_BOUNDS = {
    "r_pain":          (2.0, 5.5),
    "pain_decay":      (0.1, 3.0),
    "step_size":       (1.0, 15.0),
    "vote_threshold":  (0.0001, 0.05),
}
PARAM_NAMES = list(PARAM_BOUNDS.keys())


class PainModel(Model):
    """
    Pain-signal conformational model.

    Agents are implicit (no Mesa Agent subclass needed) — the model directly
    iterates over rotor atoms and applies decisions.  Mesa is used only for
    reproducible RNG via self.rng.
    """

    def __init__(
        self,
        smiles: str,
        params: PainParams,
        init: str = "etkdg",
        n_steps: int = 500,
        seed: int = 42,
    ) -> None:
        super().__init__(seed=seed)
        self.params = params
        self.init = init
        self.n_steps = n_steps

        # Build and embed molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Bad SMILES: {smiles!r}")
        mol_h = Chem.AddHs(mol)
        p = AllChem.ETKDGv3()
        p.randomSeed = seed
        if AllChem.EmbedMolecule(mol_h, p) == -1:
            AllChem.EmbedMolecule(mol_h)
        self.mol: Chem.RWMol = Chem.RWMol(mol_h)

        # Detect rotatable bonds
        self.bonds: List[RotatableBond] = find_rotatable_bonds(mol_h)
        if not self.bonds:
            raise ValueError(f"No rotatable bonds in {smiles!r}")

        # Apply init
        conf = self.mol.GetConformer(0)
        if init == "etkdg":
            read_angles(self.mol, self.bonds)
        elif init == "random":
            for b in self.bonds:
                a, i, j, bk = b.dihedral_atoms
                rdMolTransforms.SetDihedralDeg(
                    conf, a, i, j, bk, float(self.rng.uniform(-180.0, 180.0))
                )
            read_angles(self.mol, self.bonds)
        elif init == "zeros":
            for b in self.bonds:
                a, i, j, bk = b.dihedral_atoms
                rdMolTransforms.SetDihedralDeg(conf, a, i, j, bk, 0.0)
            read_angles(self.mol, self.bonds)
        elif init == "anti":
            for b in self.bonds:
                a, i, j, bk = b.dihedral_atoms
                rdMolTransforms.SetDihedralDeg(conf, a, i, j, bk, 180.0)
            read_angles(self.mol, self.bonds)
        else:
            raise ValueError(f"Unknown init: {init!r}")

        # Group bonds by atom_i (rotor atoms)
        self._rotors: Dict[int, List[RotatableBond]] = {}
        for bond in self.bonds:
            self._rotors.setdefault(bond.atom_i, []).append(bond)

        # MMFF94 for energy tracking (not used in decisions)
        self._ff_props = rdForceFieldHelpers.MMFFGetMoleculeProperties(self.mol)

        # History
        self.energy_snapshots: List[float] = [self._mmff_energy()]
        self.pain_snapshots: List[float] = [self._total_pain()]
        self.n_rotations: int = 0
        self.current_step: int = 0

    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------

    def _mmff_energy(self) -> float:
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(self.mol, self._ff_props)
        return ff.CalcEnergy() if ff else 0.0

    def _total_pain(self) -> float:
        """Unweighted total pairwise pain (global observable)."""
        pos = self.mol.GetConformer(0).GetPositions()
        # Vectorised: all pairwise distances
        diff = pos[:, None, :] - pos[None, :, :]        # N×N×3
        d = np.linalg.norm(diff, axis=2)                # N×N
        np.fill_diagonal(d, np.inf)
        r = self.params.r_pain
        return float(np.maximum(0.0, r - d).sum() / 2.0)   # divide by 2 (pairs)

    # ------------------------------------------------------------------
    # Core: weighted pain for a bond at a given angle
    # ------------------------------------------------------------------

    def _weighted_pain_at(
        self,
        bond: RotatableBond,
        phi: float,
        midpoint: np.ndarray,
    ) -> float:
        """
        Non-destructively probe weighted pain if bond is set to phi.

        weighted_pain = sum_i  pain_i * exp(-decay * d(atom_i, midpoint))
        pain_i = sum_{j != i} max(0, r_pain - d_ij)
        """
        conf = self.mol.GetConformer(0)
        a, i, j, bk = bond.dihedral_atoms
        cur = rdMolTransforms.GetDihedralDeg(conf, a, i, j, bk)
        rdMolTransforms.SetDihedralDeg(conf, a, i, j, bk, phi)

        pos = conf.GetPositions()                         # N×3 (already updated)
        r = self.params.r_pain
        decay = self.params.pain_decay

        # Pairwise pain matrix (vectorised)
        diff = pos[:, None, :] - pos[None, :, :]         # N×N×3
        d_mat = np.linalg.norm(diff, axis=2)             # N×N
        np.fill_diagonal(d_mat, np.inf)
        pain_per_atom = np.maximum(0.0, r - d_mat).sum(axis=1)  # N

        # Weight by distance to bond midpoint
        d_to_mid = np.linalg.norm(pos - midpoint, axis=1)       # N
        weights = np.exp(-decay * d_to_mid)

        result = float((pain_per_atom * weights).sum())

        rdMolTransforms.SetDihedralDeg(conf, a, i, j, bk, cur)  # restore
        return result

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self) -> None:
        conf = self.mol.GetConformer(0)
        pos = conf.GetPositions()

        for atom_idx, bonds in self._rotors.items():
            for bond in bonds:
                a, i, j, bk = bond.dihedral_atoms
                phi = bond.current_angle
                mid = (pos[bond.atom_i] + pos[bond.atom_j]) * 0.5

                p_cur   = self._weighted_pain_at(bond, phi, mid)
                p_plus  = self._weighted_pain_at(bond, phi + self.params.step_size, mid)
                p_minus = self._weighted_pain_at(bond, phi - self.params.step_size, mid)

                best = min(p_plus, p_minus)
                if p_cur - best > self.params.vote_threshold:
                    new_phi = (phi + self.params.step_size
                               if p_plus <= p_minus
                               else phi - self.params.step_size)
                    new_phi = ((new_phi + 180.0) % 360.0) - 180.0
                    rdMolTransforms.SetDihedralDeg(conf, a, i, j, bk, new_phi)
                    bond.current_angle = new_phi
                    # Update positions for subsequent bonds in this step
                    pos = conf.GetPositions()
                    self.n_rotations += 1

        self.current_step += 1
        self.energy_snapshots.append(self._mmff_energy())
        self.pain_snapshots.append(self._total_pain())

    def run(self) -> None:
        for _ in range(self.n_steps):
            self.step()

    def final_energy(self) -> float:
        return self.energy_snapshots[-1]

    def final_pain(self) -> float:
        return self.pain_snapshots[-1]
