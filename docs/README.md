# Documentation map

Only five documents define the active ReprForge direction:

1. [`HANDOFF.md`](HANDOFF.md)
2. [`current-research-spec.md`](current-research-spec.md)
3. [`progressive-materialization-experiment-matrix.md`](progressive-materialization-experiment-matrix.md)
4. [`evidence-registry.md`](evidence-registry.md)
5. [`benchmark-landscape.md`](benchmark-landscape.md)

The remaining files record earlier hypotheses, preregistrations, negative
results, and infrastructure. They are retained for scientific auditability,
but are not instructions and must not be treated as current claims unless the
evidence registry explicitly marks them active.

`branch-ledger-2026-08-08.tsv` records the local branch heads before inactive
worktrees were removed. Removing a clean worktree did not remove its branch or
commit; recreate one with `git worktree add <path> <branch>` when an audit
actually needs it.

New documentation should be added only when it is one of:

- a frozen protocol needed to reproduce a paper table;
- a compact evidence summary that updates the registry;
- a handoff or architecture document required to operate the system.

Exploratory notes belong in an experiment branch or external research log,
not as another authoritative document in this directory.
