# Processor equivalence certificate analysis

## Decision

**PROMOTE the certificate contract, but charge its scan cost.** Two processor
packages with different file fingerprints produced the same ordered output
digest over all 2,225 Energy pages. A collection-scoped proof can therefore
remove this false invalidation safely. It is not a free metadata check: target
processing and streaming hashing took 82.549 seconds on the CPU-only run.

## Result

The source and target processor configuration SHA-256 values differ. Under the
frozen batch size and call contract, both produced exactly the same SHA-256
digest for 557 ordered batches containing `input_ids`, `attention_mask`, and
`pixel_values`. The source-build output fingerprint is 263 bytes, the transition
certificate is 469 bytes, and their combined serialized payload is 732 bytes.
All frozen gates passed, including complete page coverage, no CUDA
initialization, and rejection of a deliberately changed output digest.

The source pass took 91.033 seconds and represents one-time bookkeeping during
the original index build. The target pass took 82.549 seconds at update time.
Digest comparison itself took 19 microseconds; image decoding and processor
execution dominate. This target scan is common work in a raw rebuild but is
additional work relative to blindly trusting an old IR. It must therefore be
represented explicitly in the physical plan rather than hidden in admission.

## Mechanism implication

Version maintenance now has three separate gates:

1. **Dependency legality.** A base, vision, or unknown tensor change invalidates
   post-vision IR without attempting behavioral certification.
2. **Scoped semantic validity.** An exact fingerprint match is free. If only the
   upstream file/package fingerprint changed, a stored source-output fingerprint
   can be compared with the target processor over the same ordered collection.
3. **Physical admission.** Replay time plus equivalence-validation time must beat
   raw rebuilding and satisfy storage and retrieval-quality contracts.

The third gate matters empirically. ColPali already exposes little reusable
prefix compute and its compact IR failed the retrieval/execution contract, so
this successful equivalence proof does not turn ColPali into a positive system
result. ColQwen's successful complete transition uses an explicitly pinned
processor fingerprint, so it does not pay this full-collection scan.

## Code consequence

The public planner now persists compact component-output fingerprints, creates
source/target/scope-bound equivalence certificates, removes only the proven
invalidation, charges `validation_seconds` to reuse, and compares the resulting
route against raw rebuilding. This fixes the previous behavior in which every
legal artifact was replayed even when validation made it slower than raw.

## Boundary

The certificate proves equality only for this corpus order, batch schedule, and
three output fields. It is not global processor equivalence. The timing is one
CPU-only execution on the A100 host, while earlier model timings excluded
preprocessing; no combined end-to-end ColPali speedup is inferred by adding the
separate measurements.
