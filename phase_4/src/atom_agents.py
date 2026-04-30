"""
atom_agents.py — AtomAgent subclasses: each heavy atom_i is an agent.

Key differences from BondAgent:
  - An agent owns ALL rotatable bonds where it is atom_i (can be 1 or more).
  - Each step the agent picks WHICH bond to rotate (not predetermined).
  - Communication by atom-to-atom Euclidean distance.
  - Shared signal is mean_gradient (scalar summary across owned bonds).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List

import numpy as np
from mesa import Agent
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers, rdMolTransforms

from molecule import RotatableBond

_KB: float = 1.987204259e-3


def _wrap(phi: float) -> float:
    return ((phi + 180.0) % 360.0) - 180.0


class AtomAgent(Agent):
    """
    Base class for an atom agent.

    Owns all rotatable bonds where this atom is atom_i.  At each step the
    agent selects one bond to act on (strategy-dependent) and proposes a
    rotation, accepting/rejecting via Metropolis at model.current_temperature.
    """

    strategy: str = "base"

    def __init__(
        self,
        model,
        atom_idx: int,
        owned_bonds: List[RotatableBond],
        mol_template: Chem.Mol,
        temperature: float,
        step_size: float = 15.0,
    ) -> None:
        super().__init__(model)
        self.atom_idx = atom_idx
        self.owned_bonds = owned_bonds
        self.temperature = temperature
        self.step_size = step_size

        self.mol = Chem.RWMol(mol_template)
        self._ff_props = rdForceFieldHelpers.MMFFGetMoleculeProperties(self.mol)
        self.energy: float = self._calc_energy()

        # Per-bond numerical gradient; mean_gradient is the shared signal
        self.gradients: Dict[int, float] = {b.bond_idx: 0.0 for b in owned_bonds}
        self.mean_gradient: float = 0.0

        self.active_neighbors: List[AtomAgent] = []

        self.energy_history: List[float] = []
        self.angle_histories: Dict[int, List[float]] = {
            b.bond_idx: [] for b in owned_bonds
        }
        self._accepted_count: int = 0
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _calc_energy(self) -> float:
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(self.mol, self._ff_props)
        return ff.CalcEnergy() if ff else 0.0

    def energy_at_angle(self, bond: RotatableBond, phi: float) -> float:
        conf = self.mol.GetConformer(0)
        a, i, j, b = bond.dihedral_atoms
        cur = rdMolTransforms.GetDihedralDeg(conf, a, i, j, b)
        rdMolTransforms.SetDihedralDeg(conf, a, i, j, b, phi)
        e = self._calc_energy()
        rdMolTransforms.SetDihedralDeg(conf, a, i, j, b, cur)
        return e

    def _compute_gradients(self) -> None:
        for bond in self.owned_bonds:
            phi = bond.current_angle
            self.gradients[bond.bond_idx] = (
                self.energy_at_angle(bond, phi + 5.0)
                - self.energy_at_angle(bond, phi - 5.0)
            ) / 10.0
        if self.gradients:
            self.mean_gradient = float(np.mean(list(self.gradients.values())))

    def _metropolis(self, delta_e: float) -> bool:
        if delta_e <= 0.0:
            return True
        T = self.model.current_temperature
        return float(self.model.rng.random()) < math.exp(-delta_e / (_KB * T))

    def _apply_angle(self, bond: RotatableBond, phi: float) -> None:
        conf = self.mol.GetConformer(0)
        a, i, j, b = bond.dihedral_atoms
        rdMolTransforms.SetDihedralDeg(conf, a, i, j, b, phi)
        bond.current_angle = phi
        self.energy = self._calc_energy()
        self._accepted_count += 1

    def _pick_random_bond(self) -> RotatableBond:
        return self.owned_bonds[int(self.model.rng.integers(len(self.owned_bonds)))]

    # ------------------------------------------------------------------
    # Scheduler interface
    # ------------------------------------------------------------------

    def sync_from_model(self, master_mol: Chem.Mol) -> None:
        conf_m = master_mol.GetConformer(0)
        conf_s = self.mol.GetConformer(0)
        for bond in self.owned_bonds:
            a, i, j, b = bond.dihedral_atoms
            phi = rdMolTransforms.GetDihedralDeg(conf_m, a, i, j, b)
            rdMolTransforms.SetDihedralDeg(conf_s, a, i, j, b, phi)
            bond.current_angle = phi
        self.energy = self._calc_energy()

    def sync_to_model(self, master_mol: Chem.Mol) -> None:
        conf_m = master_mol.GetConformer(0)
        conf_s = self.mol.GetConformer(0)
        for bond in self.owned_bonds:
            a, i, j, b = bond.dihedral_atoms
            phi = rdMolTransforms.GetDihedralDeg(conf_s, a, i, j, b)
            rdMolTransforms.SetDihedralDeg(conf_m, a, i, j, b, phi)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def _decide(self) -> None:
        raise NotImplementedError

    def step(self) -> None:
        if self.owned_bonds:
            self._compute_gradients()
            self._decide()
        self.energy_history.append(self.energy)
        for bond in self.owned_bonds:
            self.angle_histories[bond.bond_idx].append(bond.current_angle)
        self._step_count += 1

    def acceptance_rate(self) -> float:
        return self._accepted_count / self._step_count if self._step_count else 0.0


# ---------------------------------------------------------------------------
# Strategy 1 — Isolated atom
# ---------------------------------------------------------------------------


class IsolatedAtomAgent(AtomAgent):
    """
    Pick a random owned bond, propose a random rotation, pure Metropolis.
    No social influence — baseline for the atom model.
    """

    strategy = "isolated"

    def _decide(self) -> None:
        bond = self._pick_random_bond()
        proposed = _wrap(
            bond.current_angle + float(self.model.rng.normal(0.0, self.step_size))
        )
        delta_e = self.energy_at_angle(bond, proposed) - self.energy
        if self._metropolis(delta_e):
            self._apply_angle(bond, proposed)


# ---------------------------------------------------------------------------
# Strategy 2 — Local greed atom
# ---------------------------------------------------------------------------


class LocalGreedAtomAgent(AtomAgent):
    """
    Selects the owned bond with the steepest downhill gradient.
    If a neighbour has lower energy, follows that gradient more aggressively.
    """

    strategy = "local_greed"

    def _decide(self) -> None:
        # Prefer the bond most likely to improve energy (most negative gradient)
        bond = min(self.owned_bonds, key=lambda b: self.gradients[b.bond_idx])
        g = self.gradients[bond.bond_idx]

        if self.active_neighbors:
            best = min(self.active_neighbors, key=lambda n: n.energy)
            if best.energy < self.energy and float(self.model.rng.random()) < 0.40:
                bias = float(np.clip(-g * 3.0, -self.step_size, self.step_size))
                proposed = _wrap(
                    bond.current_angle + float(self.model.rng.normal(bias, 6.0))
                )
            else:
                proposed = _wrap(
                    bond.current_angle + float(self.model.rng.normal(0.0, self.step_size))
                )
        else:
            proposed = _wrap(
                bond.current_angle + float(self.model.rng.normal(0.0, self.step_size))
            )

        delta_e = self.energy_at_angle(bond, proposed) - self.energy
        if self._metropolis(delta_e):
            self._apply_angle(bond, proposed)


# ---------------------------------------------------------------------------
# Strategy 3 — Consensus atom
# ---------------------------------------------------------------------------


class ConsensusAtomAgent(AtomAgent):
    """Veto Metropolis-valid move if >40% of neighbour atoms are stressed."""

    strategy = "consensus"

    def _decide(self) -> None:
        bond = self._pick_random_bond()
        proposed = _wrap(
            bond.current_angle + float(self.model.rng.normal(0.0, self.step_size))
        )
        delta_e = self.energy_at_angle(bond, proposed) - self.energy

        if not self._metropolis(delta_e):
            return

        if self.active_neighbors:
            nbr_e = [n.energy for n in self.active_neighbors]
            mean_e = float(np.mean(nbr_e))
            n_stressed = sum(1 for e in nbr_e if e > mean_e + 0.5)
            if n_stressed / len(self.active_neighbors) > 0.40:
                return

        self._apply_angle(bond, proposed)


# ---------------------------------------------------------------------------
# Strategy 4 — Adaptive density atom
# ---------------------------------------------------------------------------


class AdaptiveDensityAtomAgent(AtomAgent):
    """Step size scales inversely with neighbourhood density."""

    strategy = "adaptive_density"

    def __init__(self, model, atom_idx, owned_bonds, mol_template,
                 temperature, step_size=15.0):
        super().__init__(model, atom_idx, owned_bonds, mol_template,
                         temperature, step_size)
        self._recent: deque = deque(maxlen=25)
        self._sigma_scale: float = 1.0

    def _decide(self) -> None:
        n = len(self.active_neighbors)
        sigma = (25.0 + (5.0 - 25.0) * n / (n + 3.0)) * self._sigma_scale

        bond = self._pick_random_bond()
        proposed = _wrap(
            bond.current_angle + float(self.model.rng.normal(0.0, sigma))
        )
        delta_e = self.energy_at_angle(bond, proposed) - self.energy
        accepted = self._metropolis(delta_e)

        self._recent.append(accepted)
        if accepted:
            self._apply_angle(bond, proposed)

        if len(self._recent) == 25:
            acc = sum(self._recent) / 25
            if acc > 0.30:
                self._sigma_scale = min(self._sigma_scale * 1.02, 3.0)
            else:
                self._sigma_scale = max(self._sigma_scale * 0.98, 0.1)


# ---------------------------------------------------------------------------
# Strategy 5 — Gradient exchange atom
# ---------------------------------------------------------------------------


class GradientExchangeAtomAgent(AtomAgent):
    """
    Selects the owned bond with the largest absolute gradient.
    Blends its own gradient with neighbours' mean_gradient scalar signal.
    """

    strategy = "gradient_exchange"

    def _decide(self) -> None:
        # Select the most informative bond (steepest slope in either direction)
        bond = max(self.owned_bonds, key=lambda b: abs(self.gradients[b.bond_idx]))
        g_self = self.gradients[bond.bond_idx]

        if self.active_neighbors:
            n = len(self.active_neighbors)
            alpha = 0.60 * n / (n + 4.0)
            g_nbrs = float(np.mean([nb.mean_gradient for nb in self.active_neighbors]))
            g_blended = (1.0 - alpha) * g_self + alpha * g_nbrs
        else:
            g_blended = g_self

        bias = float(np.clip(-g_blended * 3.0, -self.step_size, self.step_size))
        proposed = _wrap(
            bond.current_angle + float(self.model.rng.normal(bias, self.step_size))
        )
        delta_e = self.energy_at_angle(bond, proposed) - self.energy
        if self._metropolis(delta_e):
            self._apply_angle(bond, proposed)


# ---------------------------------------------------------------------------
# Strategy 6 — Best-first (atom-specific: always pick steepest-gradient bond)
# ---------------------------------------------------------------------------


class BestFirstAtomAgent(AtomAgent):
    """
    Atom-specific strategy: always select the owned bond whose gradient
    points most steeply downhill (most negative dE/dphi).

    A bond agent cannot do this — it has only one bond.  An atom that owns
    multiple bonds has a genuine choice; this strategy exploits that choice
    optimally at each step.
    """

    strategy = "best_first"

    def _decide(self) -> None:
        # Pick bond with most negative gradient → most likely to improve energy
        bond = min(self.owned_bonds, key=lambda b: self.gradients[b.bond_idx])
        g = self.gradients[bond.bond_idx]
        bias = float(np.clip(-g * 3.0, -self.step_size, self.step_size))
        proposed = _wrap(
            bond.current_angle + float(self.model.rng.normal(bias, self.step_size))
        )
        delta_e = self.energy_at_angle(bond, proposed) - self.energy
        if self._metropolis(delta_e):
            self._apply_angle(bond, proposed)


# ---------------------------------------------------------------------------
# Strategy 7 — Coordinated (atom-specific: rotate ALL owned bonds per step)
# ---------------------------------------------------------------------------


class CoordinatedAtomAgent(AtomAgent):
    """
    Atom-specific strategy: attempt to rotate every owned bond in a single
    step (sequential, each accepting independently).

    For a single-bond atom this is identical to isolated.  For a multi-bond
    atom (e.g. a branching nitrogen) it explores all its degrees of freedom
    simultaneously, moving in N-dimensional dihedral space rather than 1D.
    """

    strategy = "coordinated"

    def _decide(self) -> None:
        # Shuffle order to avoid always favouring the same bond
        order = list(range(len(self.owned_bonds)))
        self.model.rng.shuffle(order)
        for idx in order:
            bond = self.owned_bonds[idx]
            proposed = _wrap(
                bond.current_angle + float(self.model.rng.normal(0.0, self.step_size))
            )
            delta_e = self.energy_at_angle(bond, proposed) - self.energy
            if self._metropolis(delta_e):
                self._apply_angle(bond, proposed)


# ---------------------------------------------------------------------------
# Strategy 8 — Lookahead (atom-specific: sample K proposals, pick best)
# ---------------------------------------------------------------------------

_LOOKAHEAD_K: int = 5   # number of candidate angles to evaluate before deciding


class LookaheadAtomAgent(AtomAgent):
    """
    Atom-specific strategy: sample K candidate angles for the selected bond,
    evaluate the energy of each (non-destructively), then apply Metropolis
    against the single best candidate.

    A bond agent proposes one angle blindly.  An atom can afford to screen
    K options first because it controls the bond exclusively — no wasted
    per-agent coordination overhead for the extra evaluations.
    """

    strategy = "lookahead"

    def _decide(self) -> None:
        # Prefer the bond most in need of improvement
        bond = min(self.owned_bonds, key=lambda b: self.gradients[b.bond_idx])

        # Sample K candidates and pick the lowest-energy one
        phi = bond.current_angle
        candidates = [
            _wrap(phi + float(self.model.rng.normal(0.0, self.step_size)))
            for _ in range(_LOOKAHEAD_K)
        ]
        energies = [self.energy_at_angle(bond, c) for c in candidates]
        best_idx = int(np.argmin(energies))
        proposed = candidates[best_idx]

        delta_e = energies[best_idx] - self.energy
        if self._metropolis(delta_e):
            self._apply_angle(bond, proposed)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ATOM_AGENT_CLASSES = {
    "isolated": IsolatedAtomAgent,
    "local_greed": LocalGreedAtomAgent,
    "consensus": ConsensusAtomAgent,
    "adaptive_density": AdaptiveDensityAtomAgent,
    "gradient_exchange": GradientExchangeAtomAgent,
    "best_first": BestFirstAtomAgent,
    "coordinated": CoordinatedAtomAgent,
    "lookahead": LookaheadAtomAgent,
}

ATOM_STRATEGIES_NEW = ["best_first", "coordinated", "lookahead"]
ATOM_STRATEGIES_ALL = list(ATOM_AGENT_CLASSES.keys())
