# Topology Design

### How parameter-free network connectivity shapes what structures emerge during learning

**Question.** How does fixed feed-forward connectivity, holding the learned operators, parameter count and softmax cross-entropy loss fixed, change which features get preserved, erased or constructed, which computations get shared or specialized, and which updates transfer or interfere? And through what mechanism?

**Approach: derive before measuring.** Topology fixes a function class *and* a parameterization. The parameterization's symmetries give conserved quantities, and those set an effective preconditioner on the learned function. The preconditioner sets learning order and implicit bias, and those determine emergent structure. Each link has existing theory, so each study starts from a prediction. See [docs/SPEC.md](docs/SPEC.md).

## Status (19 Oct 2026)

- **Phase 0 (linear theory):** core results proved and tested (`pytest`, 45 tests). The headline consequences:
  - Every graph in the residual parent-selection family is identical to first order at zero branch weights. Topology acts only at second order, so its effect is gated by *richness* (how far training moves the branches).
  - The parallel graph $(I+A+B)$ is a single matrix.
  - The composed graph $(I+B)(I+A)$ follows Arora et al.'s closed-form preconditioned flow. Its one-example transfer is a closed-form function of the learned map: it accelerates amplification and slows erasure.
  - Open: mixed chain-plus-shortcut graphs (an exploratory probe shows the shortcut carrying more of the erasure), imbalanced initialization, separable-data implicit bias. See [docs/phase0_linear_theory.md](docs/phase0_linear_theory.md).
- **Phases 1–5:** specified with directional predictions and kill criteria; not started.

## Study sequence

| Phase | Question |
|---|---|
| 0 | Conserved quantities, flows and implicit bias of linear residual DAGs |
| 1 | Gated deep linear networks: does routing predict which pathways win (shared vs specialized)? |
| 2 | **Flagship:** does sequential composition lower the effective leap of composite features relative to learned intermediates? |
| 3 | Reuse, interference and memorization: gradient coherence and exception localization |
| 4 | Preserve and erase: topology-dependent erasure cost; sequential vs parallel transformer blocks |
| 5 | Scale under μP / depth-μP at controlled feature load |

## Repository map

- [docs/SPEC.md](docs/SPEC.md): active research spec (question, causal chain, rules, phases, instruments).
- [docs/phase0_linear_theory.md](docs/phase0_linear_theory.md): derivations, with a test map.
- [docs/literature.md](docs/literature.md): literature grouped by the link of the chain each work informs; novelty boundary.
- [docs/history.md](docs/history.md): condensed history and decision log, including the v0 → v1 pivot.
- [docs/00_questions_in_original_words.md](docs/00_questions_in_original_words.md): the motivating questions in the researcher's words.
- [AGENTS.md](AGENTS.md): contract for coding agents.
- `src/topology_design/linear_family.py`: linear parent-selection graphs, training, closed-form predictions.
- `tests/`: Phase 0 propositions plus the v0 scalar math fixtures.
- `experiments/phase0/`: exploratory probes (labeled as such).
- `archive/v0/`: the original ChatGPT-assisted documentation, unedited, kept for provenance.

## Running

```bash
pip install -e ".[test]"   # needs torch
pytest                      # ~15 s on CPU; skip the slower run with: pytest -m "not slow"
PYTHONPATH=src python experiments/phase0/prop7_pathway_probe.py
```
