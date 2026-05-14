"""
agents.py — Five BondAgent subclasses implementing different local decision strategies.

Each agent owns a *private* copy of the molecule (Chem.RWMol) and its own
MMFF94 force field.  Private copies eliminate race conditions when agents run
in parallel within the same graph-coloring group.

The master_mol (shared) is used only for inter-agent coordination via the
ProximityScheduler; agents read from it via sync_angle_from_model() and write
back via sync_angle_to_model().
"""

from __future__ import annotations

import math
from collections import deque
from typing import List

import numpy as np
from mesa import Agent
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers, rdMolTransforms

from molecule import RotatableBond

# Boltzmann constant in kcal / (mol · K)
_KB: float = 1.987204259e-3


def _wrap(phi: float) -> float:
    """Wrap an angle to the interval [-180, 180)."""
    return ((phi + 180.0) % 360.0) - 180.0


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------


class BondAgent(Agent):
    """
    Base class for a rotatable-bond agent.

    Subclasses must implement _decide().
    """

    strategy: str = "base"

    def __init__(
        self,
        model,
        bond: RotatableBond,
        mol_template: Chem.Mol,
        temperature: float,
        step_size: float = 15.0,
    ) -> None:
        super().__init__(model)
        self.bond = bond
        self.temperature = temperature
        self.step_size = step_size

        # Private molecule copy — no shared mutable state
        self.mol = Chem.RWMol(mol_template)
        self._ff_props = rdForceFieldHelpers.MMFFGetMoleculeProperties(self.mol)

        # Current energy (total MMFF94 of private mol)
        self.energy: float = self._calc_energy()

        # Numerical gradient dE/dphi (kcal/mol/deg) — shared via neighbor observation
        self.gradient: float = 0.0

        # Populated by ProximityScheduler before each step
        self.active_neighbors: List[BondAgent] = []

        # Histories
        self.energy_history: List[float] = []
        self.angle_history: List[float] = []
        self.n_neighbors_history: List[int] = []

        # Acceptance tracking
        self._accepted_count: int = 0
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _calc_energy(self) -> float:
        """Compute MMFF94 energy from the current private conformer."""
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(self.mol, self._ff_props)
        if ff is None:
            return 0.0
        return ff.CalcEnergy()

    def energy_at_angle(self, phi: float) -> float:
        """
        Non-destructive probe: total MMFF94 energy if the bond were at *phi*.

        Saves the current angle, sets phi, measures, then restores.  The
        private mol is never left in a half-modified state.
        """
        conf = self.mol.GetConformer(0)
        a, i, j, b = self.bond.dihedral_atoms
        current = rdMolTransforms.GetDihedralDeg(conf, a, i, j, b)
        rdMolTransforms.SetDihedralDeg(conf, a, i, j, b, phi)
        e = self._calc_energy()
        rdMolTransforms.SetDihedralDeg(conf, a, i, j, b, current)
        return e

    def _compute_gradient(self) -> float:
        """Central-difference numerical gradient dE/dphi in kcal/mol/deg."""
        phi = self.bond.current_angle
        return (self.energy_at_angle(phi + 5.0) - self.energy_at_angle(phi - 5.0)) / 10.0

    def _metropolis(self, delta_e: float) -> bool:
        """Standard Metropolis acceptance criterion at the model's current temperature."""
        if delta_e <= 0.0:
            return True
        T = self.model.current_temperature
        return float(self.model.rng.random()) < math.exp(-delta_e / (_KB * T))

    def _apply_angle(self, phi: float) -> None:
        """Commit *phi* to the private conformer and update energy / counters."""
        conf = self.mol.GetConformer(0)
        a, i, j, b = self.bond.dihedral_atoms
        rdMolTransforms.SetDihedralDeg(conf, a, i, j, b, phi)
        self.bond.current_angle = phi
        self.energy = self._calc_energy()
        self._accepted_count += 1

    # ------------------------------------------------------------------
    # Scheduler interface
    # ------------------------------------------------------------------

    def sync_angle_from_model(self, master_mol: Chem.Mol) -> None:
        """Copy the current dihedral angle from master_mol into the private copy."""
        a, i, j, b = self.bond.dihedral_atoms
        phi = rdMolTransforms.GetDihedralDeg(master_mol.GetConformer(0), a, i, j, b)
        rdMolTransforms.SetDihedralDeg(self.mol.GetConformer(0), a, i, j, b, phi)
        self.bond.current_angle = phi
        self.energy = self._calc_energy()

    def sync_angle_to_model(self, master_mol: Chem.Mol) -> None:
        """Write the accepted dihedral angle from the private copy back to master_mol."""
        a, i, j, b = self.bond.dihedral_atoms
        phi = rdMolTransforms.GetDihedralDeg(self.mol.GetConformer(0), a, i, j, b)
        rdMolTransforms.SetDihedralDeg(master_mol.GetConformer(0), a, i, j, b, phi)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def _decide(self) -> None:
        raise NotImplementedError

    def step(self) -> None:
        self.gradient = self._compute_gradient()
        self._decide()
        self.energy_history.append(self.energy)
        self.angle_history.append(self.bond.current_angle)
        self.n_neighbors_history.append(len(self.active_neighbors))
        self._step_count += 1

    def acceptance_rate(self) -> float:
        if self._step_count == 0:
            return 0.0
        return self._accepted_count / self._step_count


# ---------------------------------------------------------------------------
# Strategy 1 — Isolated (pure Metropolis, no social influence)
# ---------------------------------------------------------------------------


class IsolatedAgent(BondAgent):
    """
    The blind agent.  Observes only its own energy gradient; ignores all
    neighbors even when they are close.  Baseline: what emerges from zero
    social influence?
    """

    strategy = "isolated"

    def _decide(self) -> None:
        phi = self.bond.current_angle
        proposed = _wrap(phi + float(self.model.rng.normal(0.0, self.step_size)))
        delta_e = self.energy_at_angle(proposed) - self.energy
        if self._metropolis(delta_e):
            self._apply_angle(proposed)


# ---------------------------------------------------------------------------
# Strategy 2 — Local Greed (imitate the most successful nearby neighbor)
# ---------------------------------------------------------------------------

_P_IMITATE: float = 0.40  # probability of copying a better neighbor's angle


class LocalGreedAgent(BondAgent):
    """
    The opportunist.  If a nearby neighbor has lower energy, it biases its
    proposal toward that neighbor's dihedral angle with probability P_IMITATE.
    Rule: "do what the successful neighbor nearby is doing."
    """

    strategy = "local_greed"

    def _decide(self) -> None:
        phi = self.bond.current_angle

        if self.active_neighbors:
            best = min(self.active_neighbors, key=lambda n: n.energy)
            if best.energy < self.energy and float(self.model.rng.random()) < _P_IMITATE:
                delta = _wrap(best.bond.current_angle - phi)
                proposed = _wrap(
                    phi + 0.45 * delta + float(self.model.rng.normal(0.0, 6.0))
                )
            else:
                proposed = _wrap(phi + float(self.model.rng.normal(0.0, self.step_size)))
        else:
            proposed = _wrap(phi + float(self.model.rng.normal(0.0, self.step_size)))

        delta_e = self.energy_at_angle(proposed) - self.energy
        if self._metropolis(delta_e):
            self._apply_angle(proposed)


# ---------------------------------------------------------------------------
# Strategy 3 — Spatial Consensus (veto move if neighborhood is stressed)
# ---------------------------------------------------------------------------

_VETO_THRESHOLD: float = 0.40  # fraction of stressed neighbours that blocks a move
_STRESS_MARGIN: float = 0.5    # kcal/mol above mean → "stressed"


class SpatialConsensusAgent(BondAgent):
    """
    The cautious democrat.  Accepts only if Metropolis is satisfied AND fewer
    than VETO_THRESHOLD fraction of close neighbours are in a high-energy state.
    Rule: "don't make waves when the neighbourhood is already stressed."
    """

    strategy = "consensus"

    def _decide(self) -> None:
        phi = self.bond.current_angle
        proposed = _wrap(phi + float(self.model.rng.normal(0.0, self.step_size)))
        delta_e = self.energy_at_angle(proposed) - self.energy

        if not self._metropolis(delta_e):
            return

        if self.active_neighbors:
            nbr_energies = [n.energy for n in self.active_neighbors]
            mean_e = float(np.mean(nbr_energies))
            n_stressed = sum(1 for e in nbr_energies if e > mean_e + _STRESS_MARGIN)
            veto_rate = n_stressed / len(self.active_neighbors)
            if veto_rate > _VETO_THRESHOLD:
                return

        self._apply_angle(proposed)


# ---------------------------------------------------------------------------
# Strategy 4 — Adaptive Density (step size scales with neighbourhood crowd)
# ---------------------------------------------------------------------------

_SIGMA_SPARSE: float = 25.0   # large step when few/no neighbours (deg)
_SIGMA_DENSE: float = 5.0     # small step when many neighbours (deg)
_DENSITY_SCALE: float = 3.0   # half-saturation constant (n neighbours)
_TARGET_ACC: float = 0.30     # desired acceptance rate for secondary nudge
_ACC_WINDOW: int = 25         # sliding window length for acceptance tracking


class AdaptiveDensityAgent(BondAgent):
    """
    The crowd-aware agent.  Step size sigma scales inversely with the number
    of close neighbours: bold exploration when alone, careful steps in a dense
    / strained environment.  A secondary sliding-window controller nudges sigma
    toward TARGET_ACC acceptance.
    """

    strategy = "adaptive_density"

    def __init__(self, model, bond, mol_template, temperature, step_size=15.0):
        super().__init__(model, bond, mol_template, temperature, step_size)
        self._recent: deque = deque(maxlen=_ACC_WINDOW)
        self._sigma_scale: float = 1.0

    def _decide(self) -> None:
        n = len(self.active_neighbors)
        sigma = (
            _SIGMA_SPARSE
            + (_SIGMA_DENSE - _SIGMA_SPARSE) * n / (n + _DENSITY_SCALE)
        ) * self._sigma_scale

        phi = self.bond.current_angle
        proposed = _wrap(phi + float(self.model.rng.normal(0.0, sigma)))
        delta_e = self.energy_at_angle(proposed) - self.energy
        accepted = self._metropolis(delta_e)

        self._recent.append(accepted)
        if accepted:
            self._apply_angle(proposed)

        # Secondary: nudge sigma_scale toward target acceptance rate
        if len(self._recent) == _ACC_WINDOW:
            acc = sum(self._recent) / _ACC_WINDOW
            if acc > _TARGET_ACC:
                self._sigma_scale = min(self._sigma_scale * 1.02, 3.0)
            else:
                self._sigma_scale = max(self._sigma_scale * 0.98, 0.1)


# ---------------------------------------------------------------------------
# Strategy 5 — Gradient Exchange (share numerical gradients with neighbours)
# ---------------------------------------------------------------------------

_ALPHA_MAX: float = 0.60   # maximum weight on collective gradient signal
_ALPHA_SCALE: float = 4.0  # half-saturation constant for alpha


class GradientExchangeAgent(BondAgent):
    """
    The altruistic agent.  Blends its own local energy gradient with the mean
    gradient of close neighbours.  The blend weight alpha grows with crowd
    density: more neighbours → stronger collective signal.
    Rule: "where do we collectively want to go?"
    """

    strategy = "gradient_exchange"

    def _decide(self) -> None:
        g_self = self.gradient  # computed in step() before _decide

        if self.active_neighbors:
            n = len(self.active_neighbors)
            alpha = _ALPHA_MAX * n / (n + _ALPHA_SCALE)
            g_nbrs = float(np.mean([nb.gradient for nb in self.active_neighbors]))
            g_blended = (1.0 - alpha) * g_self + alpha * g_nbrs
        else:
            g_blended = g_self

        bias = float(np.clip(-g_blended * 3.0, -self.step_size, self.step_size))
        phi = self.bond.current_angle
        proposed = _wrap(phi + float(self.model.rng.normal(bias, self.step_size)))
        delta_e = self.energy_at_angle(proposed) - self.energy
        if self._metropolis(delta_e):
            self._apply_angle(proposed)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

AGENT_CLASSES = {
    "isolated": IsolatedAgent,
    "local_greed": LocalGreedAgent,
    "consensus": SpatialConsensusAgent,
    "adaptive_density": AdaptiveDensityAgent,
    "gradient_exchange": GradientExchangeAgent,
}
