# Adapter component census

Inspect public checkpoint headers and model configuration without loading a GPU
model. Install the reproduction dependencies, then run:

```bash
python experiments/adapter_census/census.py --help
```

The runner records repository revisions, tensor names and stage classifications
in the chosen output directory. Stage classification alone does not establish
replay validity: the processor and upstream parameter contract must also match.
