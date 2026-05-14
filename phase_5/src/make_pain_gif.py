"""
make_pain_gif.py — Animate the first N steps of the pain-signal model.

Visualises the molecule exiting steric conflicts from a zeros initialisation:
  • Atoms coloured by per-atom pain level (element colour → deep red)
  • Dashed red lines between atom pairs closer than r_pain (active clashes)
  • Yellow highlight on bonds that rotate in the current step
  • Mini energy–time chart at the bottom of each frame

Output:  results/pain_anim_{molecule}.gif  (one per molecule)

Usage:
    uv run python src/make_pain_gif.py
"""

from __future__ import annotations

import io
import os
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from pain_model import PainModel, PainParams

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MOLECULES = [
    ("aspirin",    "CC(=O)Oc1ccccc1C(=O)O"),
    ("lidocaine",  "CCN(CC)CC(=O)Nc1c(C)cccc1C"),
    ("fluoxetine", "CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1"),
]

# Best params from Stage IV optimisation
BEST_PARAMS = PainParams(r_pain=5.06, pain_decay=0.62, step_size=2.91, vote_threshold=0.006)

N_FRAMES  = 65     # steps to animate (action mostly in first ~50)
HOLD_FIRST = 4     # repeat first frame this many times (let viewer read initial state)
HOLD_LAST  = 6     # repeat last frame
MS_PER_FRAME = 120 # ms per frame → ~8 fps

BG = "#0d1117"     # dark background

# CPK-ish element colours — dark style
_EC = {
    "C": "#555566", "H": "#aaaaaa", "O": "#dd4444",
    "N": "#4466dd", "F": "#33cc88", "Cl": "#55bb33",
    "S": "#ddcc22", "default": "#aaaaaa",
}
# Scatter marker sizes (area in pts²)
_ES = {
    "C": 160, "H": 55,  "O": 140, "N": 150,
    "F": 120, "Cl": 200, "S": 200, "default": 130,
}

# CPK colours optimised for white background
_EC_W = {
    "C": "#404040", "H": "#cccccc", "O": "#cc2222",
    "N": "#2244cc", "F": "#1a9966", "Cl": "#33aa22",
    "S": "#bb9900", "default": "#888888",
}
_ES_W = {
    "C": 200, "H": 60,  "O": 180, "N": 190,
    "F": 160, "Cl": 240, "S": 240, "default": 170,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bonds(mol) -> List[Tuple[int, int]]:
    return [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]


def _symbols(mol) -> List[str]:
    return [a.GetSymbol() for a in mol.GetAtoms()]


def _pain_per_atom(pos: np.ndarray, r: float) -> np.ndarray:
    diff = pos[:, None, :] - pos[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    return np.maximum(0.0, r - d).sum(axis=1)


def _pca2d(all_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """PCA projection plane from (T, N, 3) array of positions."""
    flat = all_pos.reshape(-1, 3)
    mean = flat.mean(axis=0)
    _, _, Vt = np.linalg.svd(flat - mean, full_matrices=False)
    return mean, Vt[:2]   # (3,), (2, 3)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_snapshots(
    smiles: str,
    params: PainParams,
    n_frames: int,
    seed: int = 42,
) -> Tuple[PainModel, List[dict]]:
    """Run model step-by-step, recording positions + pain + rotations."""
    m = PainModel(smiles, params, init="zeros", n_steps=0, seed=seed)

    def snap(rotated_pairs):
        pos = m.mol.GetConformer(0).GetPositions().copy()
        return {
            "pos":        pos,
            "pain":       _pain_per_atom(pos, params.r_pain),
            "rotated":    rotated_pairs,
            "energy":     m.energy_snapshots[-1],
            "total_pain": m.pain_snapshots[-1],
        }

    snapshots = [snap([])]  # step 0

    for _ in range(n_frames):
        prev = {b.bond_idx: b.current_angle for b in m.bonds}
        m.step()
        rotated = [
            (b.atom_i, b.atom_j)
            for b in m.bonds
            if abs(b.current_angle - prev[b.bond_idx]) > 0.01
        ]
        snapshots.append(snap(rotated))

    return m, snapshots


# ---------------------------------------------------------------------------
# Single-frame renderer
# ---------------------------------------------------------------------------

def _render_molecule(
    ax,
    snap: dict,
    xy: np.ndarray,
    symbols: List[str],
    bonds: List[Tuple[int, int]],
    bonds_set: set,
    max_pain: float,
    r_pain: float,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    step: int,
    n_frames: int,
) -> None:
    ax.cla()
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    pos3 = snap["pos"]
    pain = snap["pain"]
    n = len(pos3)

    # --- clash lines (non-bonded pairs within r_pain) ---
    clash_segs = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(pos3[i] - pos3[j]))
            if d < r_pain and (i, j) not in bonds_set and (j, i) not in bonds_set:
                clash_segs.append([xy[i], xy[j]])
    if clash_segs:
        ax.add_collection(LineCollection(
            clash_segs, colors="#ff3333", alpha=0.22,
            linewidths=0.9, linestyles="--", zorder=1,
        ))

    # --- all bonds ---
    ax.add_collection(LineCollection(
        [[xy[i], xy[j]] for i, j in bonds],
        colors="#445566", linewidths=1.6, zorder=2,
    ))

    # --- rotated bonds (yellow highlight) ---
    if snap["rotated"]:
        ax.add_collection(LineCollection(
            [[xy[i], xy[j]] for i, j in snap["rotated"]],
            colors="#ffe033", linewidths=5.0, alpha=0.9, zorder=3,
        ))

    # --- atoms ---
    pain_norm = np.clip(pain / max_pain, 0.0, 1.0) if max_pain > 1e-6 else np.zeros(n)
    for k in range(n):
        sym = symbols[k]
        base = mcolors.to_rgb(_EC.get(sym, _EC["default"]))
        red  = (0.95, 0.12, 0.12)
        f    = pain_norm[k] ** 0.6          # gamma compress so subtle pain is visible
        colour = tuple(b * (1 - f) + r * f for b, r in zip(base, red))

        sz = _ES.get(sym, _ES["default"])
        ax.scatter(xy[k, 0], xy[k, 1], s=sz, c=[colour],
                   edgecolors="none", zorder=4)
        if sym != "H":
            ax.text(xy[k, 0], xy[k, 1], sym, ha="center", va="center",
                    fontsize=5.0, color="white", fontweight="bold", zorder=5)

    # --- annotations ---
    e = snap["energy"]
    p = snap["total_pain"]
    n_rot = len(snap["rotated"])

    ax.set_title(
        f"step {step:3d}/{n_frames}   "
        f"E = {e:,.0f} kcal/mol   pain = {p:.0f}",
        color="#ccd0dd", fontsize=8.5, pad=5, loc="center",
    )
    if n_rot:
        ax.annotate(
            f"↺  {n_rot} rotation{'s' if n_rot > 1 else ''}",
            xy=(0.03, 0.04), xycoords="axes fraction",
            color="#ffe033", fontsize=8, fontweight="bold",
        )


def _render_energy_bar(
    ax,
    energies_so_far: List[float],
    e0: float,
    n_frames: int,
) -> None:
    ax.cla()
    ax.set_facecolor(BG)
    n = len(energies_so_far)

    # Log scale to handle the huge dynamic range (zeros init)
    log_e = [np.log10(max(e, 1.0)) for e in energies_so_far]
    log_e0 = np.log10(max(e0, 1.0))
    log_ef = np.log10(max(energies_so_far[-1], 1.0))

    ax.fill_between(range(n), log_e, alpha=0.55, color="#4488ff", linewidth=0)
    ax.plot(range(n), log_e, color="#88bbff", linewidth=1.1)
    ax.axvline(n - 1, color="white", linewidth=0.7, alpha=0.4)

    ax.set_xlim(0, n_frames)
    ax.set_ylim(0, log_e0 * 1.05)
    ax.set_ylabel("log E", color="#8899aa", fontsize=6.5, rotation=0, labelpad=20)
    ax.set_xlabel("step", color="#8899aa", fontsize=6.5)
    ax.tick_params(colors="#8899aa", labelsize=6)
    for spine in ax.spines.values():
        spine.set_color("#2a2a3a")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# GIF builder
# ---------------------------------------------------------------------------

def make_gif(mol_name: str, smiles: str, params: PainParams, seed: int = 42) -> str:
    print(f"  [{mol_name}] collecting {N_FRAMES} snapshots …")
    m, snaps = collect_snapshots(smiles, params, N_FRAMES, seed=seed)

    syms   = _symbols(m.mol)
    bl     = _bonds(m.mol)
    bs     = {(i, j) for i, j in bl} | {(j, i) for i, j in bl}

    # PCA projection consistent across all frames
    all_pos = np.stack([s["pos"] for s in snaps])   # (T+1, N, 3)
    mean3, axes2 = _pca2d(all_pos)
    proj = [(s["pos"] - mean3) @ axes2.T for s in snaps]   # list of (N, 2)

    all_xy = np.vstack(proj)
    pad = 1.3
    xlim = (all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
    ylim = (all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad)

    # Pain scale: max over first 3 frames (initial clashes), so later frames
    # can go to 0 rather than scaling to any residual pain.
    max_pain = max(snaps[k]["pain"].max() for k in range(min(3, len(snaps))))
    max_pain = max(max_pain, 1.0)

    e0 = snaps[0]["energy"]
    ef = snaps[-1]["energy"]
    print(f"  [{mol_name}] E: {e0:,.0f} → {ef:,.0f}  |  rendering …")

    fig, (ax_mol, ax_bar) = plt.subplots(
        2, 1, figsize=(5.2, 6.2), dpi=90,
        gridspec_kw={"height_ratios": [5, 1], "hspace": 0.10},
    )
    fig.patch.set_facecolor(BG)

    # Add molecule name as a supertitle
    fig.text(
        0.5, 0.99, mol_name.capitalize(),
        ha="center", va="top", color="#eeeeff",
        fontsize=12, fontweight="bold",
    )

    pil_frames: List[Image.Image] = []

    all_energies = [s["energy"] for s in snaps]

    for fi, (snap, xy) in enumerate(zip(snaps, proj)):
        _render_molecule(
            ax_mol, snap, xy, syms, bl, bs,
            max_pain, params.r_pain, xlim, ylim,
            step=fi, n_frames=N_FRAMES,
        )
        _render_energy_bar(ax_bar, all_energies[: fi + 1], e0, N_FRAMES)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90,
                    bbox_inches="tight", facecolor=BG)
        buf.seek(0)
        pil_frames.append(Image.open(buf).copy().convert("RGB"))
        buf.close()

    plt.close(fig)

    # Build frame list with hold on first / last
    frames_out = (
        [pil_frames[0]] * HOLD_FIRST
        + pil_frames
        + [pil_frames[-1]] * HOLD_LAST
    )
    n_total = len(frames_out)
    durations = (
        [MS_PER_FRAME * 2] * HOLD_FIRST       # slow hold on first
        + [MS_PER_FRAME] * len(pil_frames)
        + [MS_PER_FRAME * 4] * HOLD_LAST       # long pause at end
    )

    out = os.path.join(RESULTS_DIR, f"pain_anim_{mol_name}.gif")
    frames_out[0].save(
        out,
        save_all=True,
        append_images=frames_out[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )
    print(f"  Saved  {out}  ({n_total} frames, {n_total * MS_PER_FRAME / 1000:.1f}s)")
    return out


# ---------------------------------------------------------------------------
# Clean (white-background) renderer
# ---------------------------------------------------------------------------

def _render_molecule_clean(
    ax,
    snap: dict,
    xy: np.ndarray,
    symbols: List[str],
    bonds: List[Tuple[int, int]],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    step: int,
    n_frames: int,
) -> None:
    ax.cla()
    ax.set_facecolor("#ffffff")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # All bonds — light gray, round caps
    ax.add_collection(LineCollection(
        [[xy[i], xy[j]] for i, j in bonds],
        colors="#c8c8c8", linewidths=2.2, zorder=2,
        capstyle="round",
    ))

    # Rotating bonds — orange highlight
    if snap["rotated"]:
        ax.add_collection(LineCollection(
            [[xy[i], xy[j]] for i, j in snap["rotated"]],
            colors="#e06010", linewidths=6.0, alpha=0.80, zorder=3,
            capstyle="round",
        ))

    # Atoms
    n = len(xy)
    for k in range(n):
        sym = symbols[k]
        colour = _EC_W.get(sym, _EC_W["default"])
        sz     = _ES_W.get(sym, _ES_W["default"])
        ax.scatter(
            xy[k, 0], xy[k, 1], s=sz, c=[colour], zorder=4,
            edgecolors="white", linewidths=1.0,
        )
        if sym != "H":
            ax.text(
                xy[k, 0], xy[k, 1], sym,
                ha="center", va="center",
                fontsize=5.5, color="white", fontweight="bold", zorder=5,
                fontfamily="sans-serif",
            )

    # Rotation label (bottom-left)
    if snap["rotated"]:
        n_rot = len(snap["rotated"])
        ax.annotate(
            f"↺  {n_rot} rotation{'s' if n_rot > 1 else ''}",
            xy=(0.03, 0.04), xycoords="axes fraction",
            color="#e06010", fontsize=8.5, fontweight="bold",
            fontfamily="sans-serif",
        )

    # Step counter (bottom-right, subtle)
    ax.annotate(
        f"{step} / {n_frames}",
        xy=(0.97, 0.04), xycoords="axes fraction",
        color="#aaaaaa", fontsize=7.5, ha="right",
        fontfamily="sans-serif",
    )


def _render_energy_bar_clean(
    ax,
    energies_so_far: List[float],
    e0: float,
    n_frames: int,
) -> None:
    ax.cla()
    ax.set_facecolor("#ffffff")
    n = len(energies_so_far)

    log_e  = [np.log10(max(e, 1.0)) for e in energies_so_far]
    log_e0 = np.log10(max(e0, 1.0))

    ax.fill_between(range(n), log_e, color="#99bbdd", alpha=0.40, linewidth=0)
    ax.plot(range(n), log_e, color="#336699", linewidth=1.4)
    ax.axvline(n - 1, color="#bbbbbb", linewidth=0.8)

    ax.set_xlim(0, n_frames)
    ax.set_ylim(0, log_e0 * 1.08)
    ax.set_ylabel("log₁₀ E", color="#888888", fontsize=6.5,
                  rotation=0, labelpad=26)
    ax.set_xlabel("step", color="#888888", fontsize=6.5)
    ax.tick_params(colors="#aaaaaa", labelsize=6)
    for spine in ax.spines.values():
        spine.set_color("#dddddd")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_gif_clean(mol_name: str, smiles: str, params: PainParams, seed: int = 42) -> str:
    """White-background, no-clash-lines version of the animation."""
    print(f"  [{mol_name}] clean — collecting {N_FRAMES} snapshots …")
    m, snaps = collect_snapshots(smiles, params, N_FRAMES, seed=seed)

    syms = _symbols(m.mol)
    bl   = _bonds(m.mol)

    all_pos = np.stack([s["pos"] for s in snaps])
    mean3, axes2 = _pca2d(all_pos)
    proj = [(s["pos"] - mean3) @ axes2.T for s in snaps]

    all_xy = np.vstack(proj)
    pad = 1.3
    xlim = (all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
    ylim = (all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad)

    e0 = snaps[0]["energy"]
    ef = snaps[-1]["energy"]
    print(f"  [{mol_name}] clean — E: {e0:,.0f} → {ef:,.0f}  |  rendering …")

    fig, (ax_mol, ax_bar) = plt.subplots(
        2, 1, figsize=(5.0, 6.0), dpi=100,
        gridspec_kw={"height_ratios": [5, 1], "hspace": 0.12},
    )
    fig.patch.set_facecolor("#ffffff")

    fig.text(
        0.5, 0.993, mol_name.capitalize(),
        ha="center", va="top",
        color="#222222", fontsize=13, fontweight="bold",
        fontfamily="sans-serif",
    )

    all_energies = [s["energy"] for s in snaps]
    pil_frames: List[Image.Image] = []

    for fi, (snap, xy) in enumerate(zip(snaps, proj)):
        _render_molecule_clean(
            ax_mol, snap, xy, syms, bl,
            xlim, ylim, step=fi, n_frames=N_FRAMES,
        )
        _render_energy_bar_clean(ax_bar, all_energies[: fi + 1], e0, N_FRAMES)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100,
                    bbox_inches="tight", facecolor="#ffffff")
        buf.seek(0)
        pil_frames.append(Image.open(buf).copy().convert("RGB"))
        buf.close()

    plt.close(fig)

    frames_out = (
        [pil_frames[0]] * HOLD_FIRST
        + pil_frames
        + [pil_frames[-1]] * HOLD_LAST
    )
    durations = (
        [MS_PER_FRAME * 2] * HOLD_FIRST
        + [MS_PER_FRAME] * len(pil_frames)
        + [MS_PER_FRAME * 4] * HOLD_LAST
    )

    out = os.path.join(RESULTS_DIR, f"pain_anim_{mol_name}_clean.gif")
    frames_out[0].save(
        out, save_all=True, append_images=frames_out[1:],
        duration=durations, loop=0, optimize=False,
    )
    print(f"  Saved  {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys as _sys
    mode = _sys.argv[1] if len(_sys.argv) > 1 else "both"

    print("=== Pain-model animation ===")
    for mol_name, smiles in MOLECULES:
        if mode in ("dark", "both"):
            make_gif(mol_name, smiles, BEST_PARAMS)
        if mode in ("clean", "both"):
            make_gif_clean(mol_name, smiles, BEST_PARAMS)
    print("Done.")
