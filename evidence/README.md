# Evidence assets

Files ending in `.tar.gz` are Git LFS objects; Git stores only pointers. Run
`git lfs install --local` and `git lfs pull` after cloning, then use `experiments/verify_artifacts.py`.

- `historical-evidence.tar.gz`: immutable numerical output and reports for the
  experiments in `docs/provenance.json`. Excluded/contaminated runs remain marked
  as such; their inclusion is for audit, not headline evidence.
- `input-metadata.tar.gz`: the exact processor/tokenizer/config files and small
  public model projection fragments needed for the independent endpoint check.
  Full model weights and datasets are downloaded from pinned public sources.
- `independent-endpoint-evidence.tar.gz`: all four targets’ raw page comparisons,
  rankings, environment/input hashes, protocol and runtime log.
- `bundles.json`: byte sizes and SHA-256 checksums for all bundles.

Large page collections, model shards, cut states and embedding banks stay outside
Git and outside these evidence bundles. Historical reports may overstate a claim;
`docs/claim-ledger.md` records the current interpretation.
