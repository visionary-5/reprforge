# Public release dependency trace

## Decision

**PROMOTE a three-stage version decision; do not promote universal replay.**
ReprForge should first decide whether an old boundary is dependency-legal, then
discharge conservative metadata changes only with an exact collection-scoped
output-equivalence certificate, and finally admit the artifact only when
measured retrieval quality, replay time, and storage justify it.

## What the public releases actually show

The frozen census contains three consecutive official transitions. ColPali
v1.1→v1.2 changes the base contract from the `mix` checkpoint to the `pt`
checkpoint, so it requires raw rebuilding even though the target adapter itself
contains no vision LoRA tensors. ColPali v1.2→v1.3 and ColQwen2.5 v0.1→v0.2 keep
their respective bases and change only decoder LoRA plus retrieval projection
tensors. These are two real releases for which a post-vision boundary can remain
valid, not a hypothetical adapter-update workload.

Static file identity is nevertheless insufficient. The ColPali v1.2 and v1.3
processor files have different SHA-256 hashes, but the already completed Energy
execution compared both processors on all 2,225 pages and found `input_ids`,
`attention_mask`, and `pixel_values` tensor-identical. Conversely, the packaged
ColQwen processor defaults differ, so its transition is legal only under the
explicitly frozen processor used by the existing prefix canary and complete
MMDocIR replay.

The two public domain adapters provide the opposite controls. The Vietnamese
ColQwen adapter has 504 decoder and two projection tensors, no vision tensors,
and already produced bitwise-equal raw/replay endpoints on 128 pages. The
Turkish ColPali adapter contains 162 vision LoRA tensors and must rebuild raw.
Thus an `adapter` label is not an invalidation class.

## Why legality is not payoff

The legal ColPali v1.2→v1.3 boundary is not a system win: exact IR was 12.564×
the terminal representation for only 18.76% measured holdout saving, while the
compact complete-Energy alternatives failed their frozen joint quality and
execution gates. The ColQwen boundary is admitted: on complete MMDocIR its
PCA-256 IR was 0.984× terminal, preserved official retrieval quality, and the
integrated replay→SQ8→validation→publication transition saved 73.397%.

This asymmetric result strengthens the compiler framing. A fixed caching rule
would either miss the useful ColQwen transition or waste storage on ColPali.
ReprForge's research object is the decision and execution contract, not the
claim that intermediate-state reuse always wins.

## New mechanism-level insight

The relevant invariant is **collection-scoped stage output**, not file or package
identity. A processor file may change without changing the tensors consumed by
the vision stage on the indexed collection. ReprForge can safely avoid this
false invalidation by storing a collection fingerprint and exact output digest,
then comparing the target processor over the same records. The proof does not
claim global processor equivalence: any record difference, scope mismatch,
upstream base change, or stale source/target fingerprint falls back to raw.

The public package now implements this certificate contract and applies it
before constructing the planner's invalidated-component set.

## Claim boundary

This is a small convenience census of already mirrored public releases, not a
sample of production update frequency. It supports the existence and diversity
of real version transitions. It cannot estimate how often a deployed user will
update an adapter, processor, base model, or index policy.
