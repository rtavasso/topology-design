"""The rewrite carrier (experiments/rewrite/DESIGN.md, revision 3): equational rewriting over small terms as a
proof-shaped toy. Terms are nested tuples (a leaf is one of the atoms 'a' 'b' 'c' 'd' '0'; an internal node is a pair);
primitives are eight directed rewrite rules applied at a node; a world is the set of axioms in force; the executor,
the BFS oracle, the verifier and the posteriors are exact.
"""
