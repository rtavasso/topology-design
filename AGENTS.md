# Coding-agent contract

**Read first, in order:** [README](README.md) → [docs/SPEC.md](docs/SPEC.md) → [docs/phase0_linear_theory.md](docs/phase0_linear_theory.md). Consult the novelty boundary in [docs/literature.md](docs/literature.md) and the niche map in [docs/literature_review.md](docs/literature_review.md) before claiming anything is new. `archive/v0/` is historical: useful for task generators and caveats, **not** for priorities.

## Current priority

Finish **Phase 0** in this order (SPEC §4):
1. **Prop. 7** — analyse the coupled chain-plus-shortcut flow for mixed graphs, e.g. $k=(0,1,0,2)$ (the reduction is already proved). Explain the per-mode exceptions in `experiments/phase0/prop7_pathway_probe.py`.
2. **Imbalanced (small random) initialization.**
3. **Separable-data directional limits.**

Then **Phase 1**: replicate the neural-race result for gated deep linear networks before varying the gating topology.

## Hard rules

1. **Primary comparisons:** same learned operators, shapes and parameter count; softmax cross-entropy; single-pass DAGs; no learned gates, recurrence or per-branch losses. Diagnostic controls outside this family are labeled as such.
2. **Derive before you measure.** If a quantity has a closed form in the setting, test the closed form and measure only the deviation.
3. **Every study pre-registers a directional prediction** (which graph does what, roughly how much, why), plus a falsifier, in its report *before* the main runs.
4. **Fair comparison:**
   - tune learning rate per arm and report best-vs-best plus curves;
   - sweep richness (initialization scale, branch multiplier);
   - over-provision width so every arm fits every target, and report fit;
   - compare at matched steps *and* matched training loss;
   - seeds: run a pilot null distribution first, then ≥10 paired seeds for reported results and ≈29 per arm for headline claims, judged by run-level bootstrap $P(A>B)$;
   - richness phase diagram, where the effect must vanish as $\alpha\to0$; dose–response over one-edge mutants; random-routing control;
   - include an identical-graph null, an operator-label permutation, and mask-preserving nulls for any weight-graph statistic.
   - Full protocol: [SPEC §3](docs/SPEC.md).
5. **Label each claim** as expressivity, optimization, or mechanism. A faster loss curve is none of these by itself.
6. **Use the tiered measurement stack** ([SPEC §5](docs/SPEC.md)). Primary endpoints come from function space. Representation comparisons are made only at symmetry-matched sites, with nulls. Never use raw CKA alone. A between-topology claim needs agreement across two tiers. Do not build SAE pipelines, causal-depth scores, or architecture search.
7. **Tests first** for every new derivation or metric. The test suite must pass before any training result is reported.

## Workflow per change

1. State the hypothesis, the graph difference, the prediction and the falsifier.
2. Write the analytic fixture or test.
3. Run the smallest experiment that can falsify the prediction.
4. Write a short report under `experiments/<phase>/` with:
   - provenance (commit, seeds, configs);
   - results including nulls and failed predictions;
   - classification of the claim type.
5. Append a dated entry to [docs/history.md](docs/history.md). Never rewrite past entries. Never create a second copy of an existing derivation; link to it.

## Repository conventions

- Library code lives in `src/topology_design/`. Keep it small, float64, and CPU-friendly for Phase 0–1.
- `experiments/` holds runnable scripts plus result notes. Mark exploratory work as **EXPLORATORY** in the header.
- Large artifacts go outside git.
