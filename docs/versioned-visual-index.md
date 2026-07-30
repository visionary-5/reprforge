# Versioned Visual Delta Index

## Purpose

The minimal lifecycle system keeps a cheap representation for every item and
materializes an expensive visual representation only when the policy requests
it:

```text
immutable text base
        +
active visual delta generation
        |
        v
delta scores replace base scores for cached items
```

This is the physical state required by the structural-locator cache
experiment. Locator and utility policies remain separate from the index: they
decide *which* item IDs to materialize; this module makes those decisions
durable and queryable.

## On-disk state

```text
index/
  manifest.json
  active.json
  base/
  versions/
    version-00000001/
      version.json
      index/
    version-00000002/
      version.json
      index/
```

- `base/` is compiled once and never modified.
- Each positive version is an immutable snapshot of the active visual delta.
- `active.json` is replaced atomically.
- Version zero means an empty visual delta.
- A reader may pin an explicit version while newer versions are published.

The current writer contract is single-writer. A crash while staging cannot
move `active.json`, so existing readers remain on the previous complete
generation.

## Query semantics

The runtime first scores every base item. It then scores the items in the
active visual delta and replaces the corresponding base scores.

Replacement is essential. Taking `max(base_score, visual_score)` would create
a representation that neither the planner nor the replay evaluated and would
hide cases where a visual route introduces or suppresses distractors.

## Control operations

```bash
python -m reprforge.versioned_visual_index create \
  --bank /path/to/embedding-bank \
  --output /path/to/tiered-index \
  --base-route text \
  --visual-route image

python -m reprforge.versioned_visual_index materialize \
  --index /path/to/tiered-index \
  --item-id layout:10 \
  --item-id layout:11

python -m reprforge.versioned_visual_index status \
  --index /path/to/tiered-index

python -m reprforge.versioned_visual_index rollback \
  --index /path/to/tiered-index \
  --version 1
```

Repeated materialization of already resident item IDs is a no-op cache hit and
does not create a new generation.

## Deliberate V0 limits

- NumPy reference query path only; the existing Torch token-work runtime has
  not yet been adapted to base+delta execution.
- One writer and no filesystem lock.
- Every generation snapshots all currently cached visual items. This makes
  rollback simple but duplicates storage across versions.
- No eviction, TTL, background acquisition queue, or delta compaction.
- Materialization copies already encoded vectors from an embedding bank; live
  model execution is outside this control-plane module.

The next benchmark should determine whether to replace full snapshots with an
append-only delta log plus periodic compaction. That design is justified only
if generation-copy cost becomes material relative to visual encoding.
