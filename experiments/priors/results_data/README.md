# Priors round results kept in git

The full raw results stay outside git: on the researcher's Mac under `results/kaggle_q1u`, `kaggle_q1s` and `kaggle_q2`, and as the outputs of the Kaggle kernels `riley9/topology-fleet-q1u`, `-q1s` and `-q2`.

| file | contents |
|---|---|
| `q1_summary.md/json`, `q2_summary.md/json` | Output of `analyze.py`: the preregistered verdicts and tables, as reported. |
| `q1u_runs.json.gz`, `q1s_runs.json.gz`, `q2_runs.json.gz` | One record per run: `run_id`, full `config`, `n_params`, the run's `summary` (with `shrink` for q1u and `jump` for q2), and a thinned `trajectory`. |
| `verification/` | The independent recomputation scripts. |

**How the trajectories are thinned:**
- every eval up to step 1,000;
- then every 500 steps (q1u, q1s) or every 100 (q2);
- plus the last eval.

Kept fields: step, train/test accuracy and loss, drop accuracy and loss (q1u), `gen`, `mem`, `rule_frac_test`, `weight_norms`. Floats are rounded to 5 decimals.

The preregistered criteria read every eval, and the raw runs evaluate every 100 steps (every 25 in q2's dense window). Recomputing them from this copy is therefore approximate; event times in `summary` are exact.

Load with `json.load(gzip.open(path, "rt"))`.
