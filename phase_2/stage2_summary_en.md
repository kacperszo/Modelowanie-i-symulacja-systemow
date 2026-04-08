# Stage 2 — Tool Analysis and Selection
## Dihedral Agents: Agent-based conformational sampling of small molecules

**Course:** Modelowanie i symulacja systemów
**Supervisor:** dr hab. Wojciech Turek
**Author:** Kacper
**Date:** 2026-03-21

---

## 1. Problem Statement

The goal of the project is to sample the **conformational space** of small organic molecules — the set of 3D shapes a molecule can adopt through rotation around its single bonds.

Each rotatable bond (C–C, C–O, C–N) has one degree of freedom: its **dihedral angle** φ ∈ [−180°, 180°]. A dihedral angle is defined by four consecutive atoms A–B–C–D as the angle between the plane containing A–B–C and the plane containing B–C–D, measured looking along the B→C axis:

```
    A                     D
     \                   /
      B ————————————— C

  Newman projection (looking along B→C):

      A           A           A
      |            \          |
 D -- B    φ=0°   D-B  φ=60° B--D    φ=180°
  eclipsed          gauche      anti (most stable)
```

The dihedral angle is computed via cross products of bond vectors:

```
b1 = B−A,  b2 = C−B,  b3 = D−C
n1 = b1 × b2,  n2 = b2 × b3
φ = atan2( (n1 × n2)·(b2/|b2|),  n1·n2 )
```

**Why this matters:** conformational space grows as ~3^N (three energy minima per bond). A typical drug-like molecule has 5–10 rotatable bonds → 243–59 049 combinations. Finding energetically favorable conformations is central to computational drug design.

---

## 2. Existing Tools — Survey

| Tool | Approach | Speed | Accuracy | CS Interest |
|------|----------|-------|----------|-------------|
| **Systematic search** | Enumerate all angle combinations on a grid | O(k^N) — infeasible for N > 7 | Complete but intractable | None |
| **ETKDG** (RDKit) | Knowledge-based: statistics from crystal structure databases | Milliseconds | Good for common topologies; fails for rare/macrocyclic structures | None |
| **OMEGA** (OpenEye) | Knowledge-based, commercial | Fast | Similar to ETKDG | None |
| **Monte Carlo** (RDKit UFF/MMFF) | Random torsion perturbations + Metropolis acceptance | Medium | Physically sound; susceptible to kinetic trapping | Metropolis criterion, MCMC |
| **Molecular Dynamics** (OpenMM, GROMACS) | Integrate Newton's equations of motion | Slow (hours/days on HPC) | Physically correct trajectories, limited timescale (~ns) | Parallelism strategies |
| **Custom ABM** | Each rotatable bond = autonomous agent; graph-coloring scheduler | Configurable | Tunable; designed to study scheduling effects | Agent-based modeling, graph algorithms |

### Notes on individual tools

**ETKDG** is the current industry standard for conformer generation and is available for free via RDKit. It is fast and produces good results for molecules well-represented in the Cambridge Structural Database (CSD). However it is a black box — it does not expose the internal sampling strategy, which makes it unsuitable for studying the effect of update ordering.

**Monte Carlo with MMFF94** (also in RDKit) is the natural baseline for our project. It uses the Metropolis-Hastings criterion to sample the Boltzmann distribution of conformations at temperature T. The key limitation is kinetic trapping: at T = 300 K barriers above ~3–4 kcal/mol are rarely crossed, leading to the sampler being stuck in a local minimum.

**Molecular Dynamics** produces the most physically faithful trajectories but is far too expensive for the scope of this project and does not offer a natural way to study discrete scheduling strategies.

---

## 3. Selected Approach and Justification

**Selected:** Custom agent-based Monte Carlo simulation implemented in Python using **RDKit** (MMFF94 force field + ETKDG as baseline) and **Mesa** (agent-based modeling framework).

### Why custom ABM over ready-made tools

The project's central research question is not just *"find good conformations"* but *"how does the update scheduling strategy affect convergence?"* This question cannot be answered with ETKDG or off-the-shelf MD — it requires explicit control over which bonds are updated, in what order, and whether updates can happen in parallel.

The agent-based model provides this control naturally:

- each rotatable bond is a **BondAgent** with state φ and a `step()` method
- the scheduler is a **GraphColoringScheduler**: bonds sharing an atom form a dependency graph; greedy graph coloring partitions bonds into independent sets (color groups) that can be updated simultaneously without geometric conflicts
- alternative scheduling strategies (sequential, random, graph-coloring) are interchangeable components

### Why RDKit for energy evaluation

RDKit provides:
1. **MMFF94** — a well-validated empirical force field covering all common organic atom types; free, Python-accessible, no external calls needed
2. **ETKDG** — used as a reference conformer generator to validate that our ABM samples the same distribution
3. **Mol parsing** — SMILES → 3D coordinates with hydrogen assignment; handles the molecular graph needed for the dependency graph

RDKit is the de facto standard library for cheminformatics in Python, which ensures reproducibility and makes the project extensible.

### Why Mesa

Mesa is a Python framework for agent-based models. It provides a clean `Agent` / `Model` / `Scheduler` architecture that maps directly onto our design. Using Mesa makes the scheduling strategy a first-class, swappable component rather than an implementation detail buried in a loop.

---

## 4. Tool Stack Summary

```
RDKit            — SMILES parsing, 3D embedding (ETKDG), MMFF94 energy evaluation
Mesa             — agent-based model framework (Agent, Model, Scheduler)
NumPy            — vector math for dihedral angle computation
Matplotlib       — visualization of dihedral distributions and energy traces
Python 3.11      — implementation language
uv               — dependency management
```

---

## 5. Research Questions (defined at this stage)

1. Does the ABM converge to the same dihedral angle distributions as RDKit ETKDG?
2. How does scheduling strategy (graph-coloring vs. sequential vs. random) affect convergence speed and conformational coverage?
3. How does temperature T affect the exploration–exploitation trade-off?

These questions will be answered with working prototype results in Stages 4–5.
