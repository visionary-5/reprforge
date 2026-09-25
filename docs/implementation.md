# Method and implementation

## Execution-path interface discovery

`reprforge/tracing.py:discover_interface` runs a forward pass under PyTorch
operator dispatch. The integration supplies visual, text and structural seed
tensors. The report describes the executed path, its consumed visual frontier,
and the parameter/buffer dependencies contributing to that frontier. It does
not prove completeness for unseen branches or establish equivalence of different
operator implementations or non-tensor configurations.

`reprforge/hooked.py:CutSpec.from_report` selects module outputs and output
leaves. `capture` retains them and records hashes of the visual-branch inputs.
`dependency_digest` reads the named parameters and registered buffers as loaded,
including non-persistent buffers. `resume` substitutes retained visual outputs
while executing the target forward; the target computes its own text-side
context. The integration must compare source and target dependency digests
before requesting replay. `resume` does not itself perform this model check.

Generic input validation hashes the tensors arriving at the visual-branch
boundary. With `check_inputs=False`, input equality is an explicit external
precondition. The implementation expects the supported module-output interface
and one execution of each retained module per page. It is not a universal
checkpoint converter or a proof of arbitrary graph equivalence.

## Fixed ColQwen2.5 continuation

`reprforge/integrations/colqwen.py` implements the fixed post-vision interface
used by the independent reconstruction and lifecycle experiments. It retains
visual tokens, input IDs, masks and grid metadata. The target obtains embeddings
from its own embedding table and recomputes positions before running its language
model and retrieval projection. Retained IDs still depend on compatible
tokenization and input templates.

`experiments/paper/e2e/e2e_rebuild.py` calls this fixed integration, rather than
`discover_interface` or generic `hooked.resume`. Its historical model contract
comes from `scripts/_common.py:contract`, which hashes `visual.state_dict()`;
non-persistent buffers are not included in that hash. The lifecycle comparison
uses the vision, processor, model configuration and dtype fields. Other recorded
fields should not be mistaken for fields enforced by that comparison.

This is the implementation associated with the recorded lifecycle timings.
Replacing it with a different validation path would require a separate timing
measurement; the existing measurements must not be reassigned to modified code.

## Failure handling

The generic replay primitive raises `PageLevelMismatch` on an input mismatch.
Its caller must choose the fallback policy. The released-pair audit records
rejections and separately evaluates selected forced-reuse controls. The fixed
lifecycle runner stops if its model contract fails; it does not time automatic
fallback on those failures. `scripts/replay_target.py` provides a separate
ColQwen CLI that routes invalid contracts or page artifacts to full encoding.
These are separate entry points, not interchangeable end-to-end workflows.

## Retention and approximate storage

The main lifecycle retains one fixed post-vision state per page. The NumPy
planning utilities and synthetic examples explore retention choices; their
availability does not establish runtime multi-boundary fallback in the measured
lifecycle. Compression experiments are separate approximate-replay evaluations.
BF16 exact-replay results must not be attributed to PCA/INT8 states.
