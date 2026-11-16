# Building one block by attention over a bank of weights — a formalisation

**Status: a formalisation, nothing trained.** Written 16 Nov 2026 for idea I38 (history Entries 110–111). Exact statements are tests in `tests/test_weight_lookup.py` (U1–U4) and `tests/test_weight_mixing.py` (W1–W6). Related work is cited from memory and has not been re-read.

## 1. The atom is a unit, not a row and not a matrix

A block V tanh(B h) is a sum over hidden units: Σ_i v_i tanh(b_i · h). Unit i is a pair — the **read row** b_i of B and the **write column** v_i of V. Three sizes of thing could be keyed and recombined:

| What is keyed | Recombination | Verdict |
|---|---|---|
| Whole matrices (one key per stream) | W̄ = Σ_k α_k W_k | Fragile inside the simplex (W1): two streams that compute the same function with their units in a different order average to something else (U2). Needs the streams' units to stay aligned. |
| Rows and columns independently | any row with any column | Meaningless: a read row without its write column is half a unit. |
| **Units** (row and column together, one key per unit) | the block's B is the retrieved rows, its V the retrieved columns scaled by their weights | **Exact and safe.** Assembling the matrices from retrieved units is the same as gating units (U1); loading two sets of units gives exactly the sum of the two blocks (U1, W4); relabelling the bank changes nothing (U2). |

The reason is the block's permutation symmetry: hidden units can be reordered without changing the function, so any operation on weights that is to be meaningful must treat (b_i, v_i) as one object. Blending two units' rows inside one slot brings the interference back (U3), so retrieval into a slot should be hard (one unit per slot) even when the weight on the slot is soft. For a purely linear map the same atom is a rank-one term v_i b_iᵀ, and the build-up is W̄ = W_0 + Σ_i w_i v_i b_iᵀ.

So: yes to a complete build-up of the block's matrices by attention, with units as the things retrieved.

## 2. The bank

N units. Unit i carries: read row b_i ∈ ℝ^d; write column v_i ∈ ℝ^d; an **address key** κ_i (what makes it retrievable); a bid weight β_i. A *primitive* is a set of units that are retrieved together; nothing has to mark the sets in advance.

## 3. Two queries, with different jobs

The worlds result (Entry 108) fixes this part: a gate that reads one pooled summary of context and state did not recombine; one retrieval per item of context, followed by a competition decided by the state, did. So the input and the state are both queries, but not concatenated into one:

| Query | Asks | Matched against |
|---|---|---|
| each **item of the input** (a token, a stated rule), separately | which units does this item make available? | address keys κ_i |
| the **state** | among what is available, what applies now? | the units' own read rows b_i — every unit already scores the state through b_i · h |
| the **pass** (a register, Entry 104), if the task needs it | which item is current? | positions of the items |

## 4. One pass

Items e_1 … e_J of the input; state h.

1. **Load.** For each item j: a_j = sparse-attention over the bank with query Q e_j and keys κ_i (top-m, or a sparse softmax). a_{ji} is how much unit i belongs to item j's operator.
2. **Read.** Loaded units read the state: z_i = tanh(b_i · h), computed only for units some item loaded.
3. **Compete.** Each item bids from its own units: bid_j = Σ_i a_{ji} β_i z_i; π = softmax_j(bid).
4. **Assemble and apply.** Unit weights w_i = Σ_j π_j a_{ji}; h ← h + ε Σ_i w_i v_i z_i, which is the block with B̄ = the loaded rows and V̄ = the loaded columns scaled by w.

At most J·m of the N units are computed. If the load step does not depend on the state, the block can be assembled once per pass and applied to every position.

With address keys shared inside fixed groups of units, this is exactly the per-token streams arm of the worlds carrier (U4) — the arm that recombined. The formalisation is that arm with the stream shrunk to a unit and the group left to be learned.

## 5. Choices left open

- **Address keys.** Free (learned per unit); shared within a group (streams); or derived from the unit's content, κ_i = P b_i, so that the address moves as the unit learns. The last is the "lookup over the weights themselves"; it closes a loop between what a unit does and when it is retrieved (competitive-learning dynamics; Entry 110, point 4).
- **How the state competes.** Bids per item as above (items compete), or a softmax directly over loaded units in place of tanh (units compete: the block becomes a modern Hopfield retrieval step, V̄ softmax(β B̄ h)).
- **Sparsity of the load, and its schedule.** A unit that is never loaded gets no gradient (Entry 103); starting soft and sharpening is the known cure (EvoMoE, Dynamic Convolution).
- **Combination across items** when several should apply at once: with units as atoms, loading both is exact (U1); no weight-space trick is needed.

## 6. What this does not settle

Where the items come from when the input does not name them (Entry 109, point 6). Whether groups of units emerge as primitives without being told (in Entry 105 a trained gated block did sort its units by token). Everything about training dynamics is expectation (Entry 110).

## 7. Relatives (from memory, not re-read)

Feed-forward layers as key–value memories (Geva et al. 2021); fast weight programmers — attention as a matrix built from rank-one key–value terms (Schlag, Irie & Schmidhuber 2021); attention as a hypernetwork (Schug et al. 2024); Neural Interpreters (Rahaman et al. 2021, read in Entry 94); modern Hopfield networks (Ramsauer et al. 2020); product-key memory layers (Lample et al. 2019); SMEAR, CondConv and Dynamic Convolution for whole-matrix mixing.
