# What must be true for target replay to be correct?

This note specifies the existing mechanism, its sufficient conditions and its
evidence obligations. It is not a new planner or a universal equivalence checker.

## 1. A cut is a complete resumable state, not a layer number

Let the target document encoder be decomposed as

    E_t(x) = R_t(S_t(x)).

S includes every value the target continuation needs from earlier computation:
saved activations and resume context. R must include every remaining target
operation, including terminal projection, normalization and output masking.
An activation without its required context need not satisfy this decomposition.

For the actual independent ColQwen2.5 endpoint, the saved state contains
post-merger visual tokens, input_ids, attention_mask and image_grid_thw.
Replay computes target text embeddings and target positions, executes the target
language model, projection, normalization and masking. The expensive visual
tokens and the text embeddings are not saved as one fused tensor.

## 2. Sufficiency and validity are separate obligations

If the decoded source state equals S_t(x), and R_t is a faithful target
continuation, then R_t(decoded S_s(x)) = E_t(x), by substitution. This elementary
implication is useful as a correctness contract, not as a novel theorem.

There are two ways that an implementation can fail this contract:

1. the old state differs from what the target needs (invalid state);
2. the continuation omits or misreconstructs a required value (incomplete replay).

Testing only the same suffix on the same activation does not discharge either
obligation for a real source-to-target transition. Independent native raw target
encoding is needed as the experimental reference.

## 3. Dependencies follow the values actually saved

For each saved value, consider all upstream inputs, operations, parameters and
configuration that determine it. Their union is the state's dependency set.
Unchanged or appropriately scoped equivalent dependencies are sufficient to
preserve that state under a deterministic, matching numerical execution contract.
An unresolved dependency change requires conservative fallback; it does not
logically imply the output must differ for every page.

| Target component changes | Visual tokens + processor context | Fused visual/text language input |
|---|---|---|
| Decoder / retrieval projection | Recomputed by target | Recomputed by target |
| Text embedding table only | Recomputed by target; no automatic invalidation | Embedded in saved tensor; invalidating unless equivalence established |
| Vision tower or merger | Embedded in saved visual tokens | Embedded in saved visual tokens |
| Processor | Affects visual input and/or saved context | Affects visual input and/or saved context |

These are specific representations, not a claim that all cuts at the language
entrance have identical dependencies. A changed base name is a warning to inspect
components, not itself a proof that every saved value changed. Hidden changes
to architecture, interface, numerical execution or other upstream operations
must still invalidate reuse if equivalence is unproven.

The public library consumes dependencies and fingerprints declared by the model
integration. `CutState.validate()` checks structural fields; it does not trace
model code, compute context hashes, or prove that the dependency declaration is
complete. Calling it a general automatic soundness verifier would overstate the
implementation. The integration must bind context and source identity to the
stored state and record its execution contract.

## 4. Exact, approximate, and fallback

Exact replay requires faithful state storage and target continuation in addition
to dependency validity. Dependency matching alone does not make a lossy codec
exact. Mathematical output equality also does not independently guarantee BF16
bit patterns across arbitrary kernels, batches or software versions; endpoint
checks report those execution conditions explicitly.

Approximate decoding substitutes a different state. The equality argument no
longer applies, and the existing experiments measure resulting fidelity rather
than prove a universal error or ranking bound. An invalid upstream state is not
made valid by labelling it approximate.

Fallback executes E_t(x) directly for the affected page. The page-scoped processor
extension refines when a processor mismatch can be discharged, but does not
replace the core sufficiency and target-continuation requirements.

## 5. Evidence obligations

| Obligation | Evidence |
|---|---|
| Locality occurs in real releases | Census and release dependency traces, with processor/base qualifications |
| Stored source state and full continuation recover target | Independent raw/replay endpoints, separate source/target processes |
| Target reconstruction matters | `experiments/core-mechanism`: stale text-embedding and missing-position ablations |
| Dependency boundaries predict validity | Controlled embedding, vision and merger interventions in the same experiment |
| Replay physically omits vision | Visual-module invocation counts in native versus replay |
| Skipping vision saves document computation | Existing synchronized stage measurements with stated timing boundaries |

Controlled interventions test causal conditions, not their frequency in public
upgrades. The experiments and protocol are in
[`../experiments/core-mechanism/`](../experiments/core-mechanism/).
