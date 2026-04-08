# Dihedral Agents — ABM of Small Molecule Conformational Sampling

Agent-based model (Mesa 3.x) where each rotatable bond in a small organic
molecule is an autonomous agent. Agents use local MMFF94 energy information
and proximity-gated communication to decide whether and how to rotate.

## Research question

How does the agent decision strategy and spatial communication range affect
the speed and quality of conformational sampling?

## Strategies

| Strategy | Description |
|---|---|
| `isolated` | Pure Metropolis; no social influence (baseline) |
| `local_greed` | Imitates the lowest-energy nearby neighbour |
| `consensus` | Vetoes moves when the neighbourhood is stressed |
| `adaptive_density` | Step size scales with neighbourhood crowd density |
| `gradient_exchange` | Blends own and neighbours' MMFF94 gradients |

## Quick start

```bash
uv sync
uv run python src/tests.py   # run unit tests
uv run python src/run.py     # 60-experiment grid → results/
```

## File structure

```
src/
├── molecule.py   RDKit utilities, bond detection, dependency graph, coloring
├── agents.py     BondAgent base class + 5 strategy subclasses
├── model.py      MoleculeModel, ProximityScheduler, init functions
├── run.py        Experiment grid + 5 plots + CSV
└── tests.py      Unit tests (pure Python + RDKit)
pyproject.toml
README.md
```

## Dependencies

Python 3.11+, managed with `uv`. Key packages: `mesa>=3.0`, `rdkit>=2023.9`,
`numpy`, `matplotlib`, `networkx`.
