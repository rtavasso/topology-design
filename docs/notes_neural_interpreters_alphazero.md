# Notes: Neural Interpreters and AlphaZero, read for the per-token routing design

Read in full on 26 Oct 2026 for history Entry 39:
- Neural Interpreters: arXiv 2110.06399, the arXiv PDF;
- AlphaZero: arXiv 1712.01815, the ar5iv HTML.

Details marked *(AGZ)* come from the AlphaGo Zero paper (Silver et al. 2017, Nature), which the AlphaZero preprint builds on and does not repeat. They were not read for this note.

## Neural Interpreters (Rahaman, Gondal, Joshi, Gehler, Bengio, Locatello, Schölkopf; NeurIPS 2021)

### Architecture
- **Scripts.** A set-to-set model: a stack of n_s scripts, with no parameters shared across scripts.
- **Functions.** A script holds n_f functions. Each function is a pair f_u = (s_u, c_u):
  - a **signature** s_u, a unit vector in a d_type-dimensional "type space";
  - a **code** c_u, a vector that programs a *shared* interpreter.

  Because the code programs shared weights, the parameter count does not depend on the number of functions. A new function costs one signature and one code.
- **Routing.** An MLP infers each token's type t_i on the type hypersphere. Its compatibility with each function is a truncated kernel, exp(−(1 − s_u·t_i)/σ), set to zero beyond the threshold τ, then normalized over functions. The threshold controls how sparse routing is. Routing is **soft and multi-function**: a token's update is the compatibility-weighted sum, over functions, of each function's stream.
- **Interpreter.** A stack of n_l "lines of code" (LOCs). Each LOC is modulated attention followed by a modulated MLP. All linear layers are ModLin, conditioned on the function's code: y = W(x ⊙ LN(W_c c)) + b.
  - Attention between tokens i and j inside function u is scaled by C_ui·C_uj, so tokens interact only *inside a function both are routed to*. Context-reading is part of the primitive, gated by type.
  - A token routed to no function passes through unchanged (residual identity).
- **Iterations.** A script applies type-matching plus the interpreter n_i times, reusing the same weights across depth. Paths are partially ordered sets of functions, and functions may repeat. This occupies the "reuse across depth × selection" quadrant (history Entry 36), with soft selection.
- **Collapse remedy.** The main failure mode is signatures and types collapsing to one point, so that everything routes everywhere. The fix: freeze the signatures at random initialization (high entropy), which forces the type MLP to produce diverse types.

### Results
- **Fuzzy Boolean toy.** Product fuzzy logic (and, or, not), N = 5 inputs; 20 functions for pretraining, 10 held out. Fine-tuning on the held-out functions, measured by R²:

  | what was fine-tuned | R² |
  |---|---|
  | output tokens only | 0.920 |
  | **routing only** (type inference and signatures) | **0.986** |
  | everything | 0.995 |

  Re-routing a frozen library recovers almost everything: evidence that the learned functions recompose. Function–primitive correspondence was not verified directly; it was only inferred from routing plots.
- **Digits** (SVHN, MNIST-M, MNIST):
  - In-distribution, NI matches ViT within 0.2% with 0.64M parameters against ViT's 1.8M.
  - It adapts faster to K-MNIST with little data; the gap closes as data grows.
  - Adding functions at adaptation time helps.
  - Dropping functions at test time degrades accuracy gradually.
  - Keeping the pretrained functions beats re-initialized codes and signatures.
- **PGM abstract reasoning,** out-of-distribution test accuracy:

  | split | NI | ViT | best prior |
  |---|---|---|---|
  | interpolation | 70.5 | 67.7 | 65.4 |
  | attribute pairs | 36.6 | 34.1 | 36.8 (VAE-WReN) |
  | triple pairs | 45.2 | 44.1 | 43.6 |
  | held-out triples | 20.0 | 15.9 | 24.6 |
  | extrapolation | 19.4 | 16.4 | 18.9 |

  Chance is 12.5%. In-distribution, NI scores 79.9 on held-out triples and 91.8 on extrapolation. The gains are real but modest, and the hard systematic splits stay near chance.
- **Anytime behavior.** Halving the iterations at test time keeps at least 70% of accuracy.
- **Ablations** (100-run sweep):
  - 5 functions work consistently, while most 1-function runs fail;
  - frozen signatures help marginally;
  - more scripts stabilize training;
  - the authors recommend more iterations for out-of-distribution performance and fast adaptation.
- **Stated limitations:** bottom-up processing only; no higher-order functions; no lazy execution.

## AlphaZero (Silver et al., arXiv 1712.01815)

- **Network.** f_θ(s) → (p, v): a prior over next moves, and a scalar expected outcome.
- **Loss:** l = (z − v)² − πᵀ log p + c‖θ‖².
  - z is the game's final outcome.
  - π is the distribution of MCTS visit counts at the root.
- **Policy improvement by search.** MCTS turns the raw prior p into a better policy π, and the network is trained toward π: expert iteration.
  - Simulations pick moves by prior, value and visit count. *(AGZ)* PUCT: argmaxₐ Q(s,a) + c_puct·P(s,a)·√N(s) / (1 + N(s,a)).
  - Leaves are evaluated by v. Values are **averaged** over subtrees, not minimaxed.
  - 800 simulations per move in training.
  - Dirichlet noise at the root, α = 0.3 / 0.15 / 0.03 for chess / shogi / Go, scaled inversely to the number of legal moves.
  - Moves are sampled in proportion to visit counts during training, and chosen greedily at evaluation.
  - Illegal moves are masked and the prior renormalized.
- **Differences from AlphaGo Zero:**
  - it models expected outcome, draws included;
  - no symmetry augmentation;
  - one continually updated network, with no evaluation gating;
  - the same hyperparameters for every game except the Dirichlet α.
- **Scale:**
  - 700k steps at batch 4,096;
  - 44M / 24M / 21M self-play games for chess / shogi / Go;
  - 5,000 TPUs for self-play and 64 for training;
  - 9 / 12 / 34 hours.
- **Results:**
  - vs Stockfish: 28 wins, 72 draws, 0 losses;
  - vs Elmo: 90 wins, 2 draws, 8 losses;
  - vs AlphaGo Zero (3-day version): 60 wins, 40 losses.

  It evaluates about 80k positions per second against Stockfish's 70M: a learned prior and value make the search selective, about 1,000× fewer positions. It also scales better with thinking time than alpha-beta.
- **Why MCTS plus a network works** (the paper's argument): network evaluations make large, non-linear errors. Averaging over subtrees cancels them, while minimax propagates the worst error to the root.
- *(AGZ)* The raw network without search is much weaker than the network with search, a "seeing vs searching" gap.

## Related, not read for this note

AlphaDev (Mankowitz et al. 2023) applies AlphaZero to assembly programs, with correctness plus latency as reward. AlphaTensor applies it to matrix-multiplication algorithms. Both build programs from primitive instructions, with exact execution, and both search per task rather than amortizing across worlds in context.
