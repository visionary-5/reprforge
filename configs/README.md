# Configurations

`colqwen.json` defines the source retriever, four target retrievers and six
evaluation collections used by the independent reconstruction runner. Paths
beginning with `inputs/` are resolved by
`experiments/reconstruction/prepare_inputs.py`.

`projections/` contains the retrieval projection extracted from each target's
documented base checkpoint. These files are model inputs, not experiment
outputs. Their upstream repository, revision or content hash, source shard and
SHA-256 are recorded in `experiments/reconstruction/sources.json`.
