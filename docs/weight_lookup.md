# Building one block by attention over a bank of weights — a formalisation

**Status: a formalisation, nothing trained.** Written 16 Nov 2026 for idea I38 (history Entries 110–112). Exact statements are tests in `tests/test_weight_lookup.py` (U1–U8) and `tests/test_weight_mixing.py` (W1–W6). Related work is cited from memory and has not been re-read.

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

## 8. Keys computed from the weights, one key head per site (Entry 112)

The researcher's proposal: the parallel blocks are learnable parameters, a prior of primitives; a learned **key head** works on those weights to produce the keys; a matrix and the rows and columns tied to it share one key head, and a different matrix in the same block has its own. This is the "derived from content" option of section 5, made specific. It holds up, with three sharpenings.

**8.1 "A matrix with its tied rows and columns" is a site.** A *site* is one set of hidden units together with everything that must move with them when they are reordered: here the rows of B and the columns of V. Sites are the unit of the block's symmetry, so they are the right unit for a key head, for two reasons. Inside a site the read and the write cannot be fetched separately — reads from one block with writes from another depend on how the second block's units happen to be labelled (U5) — so the tied pair needs one *key*, not only one projection. Across sites nothing is tied: two sites talk only through the state, so site 1 can come from block k and site 2 from block l, and all K² assembled blocks are well defined and distinct (U5). With S sites that is K^S blocks from K stored ones — a second axis of recombination, on top of loading several primitives into one site. The testbed block V tanh(B h) has a single site, so there it is one key head. (In a transformer block the sites would be the feed-forward hidden units and the attention heads, a head's four matrices moving together; that is the standard symmetry of those layers and is not tested here.)

**8.2 The key must be unchanged by the site's symmetries.** A tanh unit can have its row and column both negated without changing the block. A key that is linear in the row and column, κ = P [b; v], is negated with them (U6): one function, two addresses. A key built from the product of the two, κ = (U b) ⊙ (W v), is unchanged (U6). The general recipe that gets every symmetry right at once: **key a primitive by what it computes** — evaluate the block's site on a few learned probe inputs and project the outputs. That is a key head acting on the weights, it is unchanged by reordering and sign flips (U8), and it needs its own probes per site because sites take different inputs — which is the researcher's "a different matrix has its own key projection". Summed over a block's units, the product key is the linear case of the same thing: a readout wᵀ(VB)u of the block's Jacobian at the origin (U8).

**8.3 One key per block and site, not one per unit.** A key computed from a single unit's weights addresses by *similarity*; a primitive is a set of units that are *used together*, and those are different from one another by necessity. Measured on the worlds teacher (ten teachers, six primitives of six units each): with a linear key on the read row, one query can pick a primitive's six units out of the thirty-six in 0 of 60 cases; with the write column added, 5 of 60 and never all six primitives of a teacher; with the full product v bᵀ, always — but only because thirty-six units give thirty-six independent features, so the key head is then a table with one entry per unit, a free key by another name (U7). So the membership of units in a primitive should come from the block structure — which is what "parallel blocks as a prior of primitives" already provides — and the key head should compute one key per block and site from all of that block's units there. That is the arrangement of U4 (the tested per-token streams arm) with the group's key computed from the group's weights instead of stored beside them.

**8.4 What changes against stored keys (expectation, untested).** The address follows the content: there is no table to go stale, a primitive that changes is found where it now belongs, and two primitives that compute nearly the same thing at the probes get nearly the same address and cannot be told apart by an item — by construction. The coupling runs both ways: the loss can now change what a primitive computes in order to change who retrieves it, unless the gradient from the address into the weights is cut. Under soft retrieval there is a positive feedback (retrieved together → same gradient → more alike → closer keys) that merges primitives; with stored keys the addresses can separate first and the content follow ("the gate must lead", Entry 110), and here they cannot move independently.
