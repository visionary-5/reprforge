# Target reconstruction

Run independent source capture and target raw/replay processes for four ColQwen2.5
retrievers under the common base processor. See the
[reproduction guide](../../docs/reproduction.md) for input preparation and commands.

`protocol.json` specifies page selection, processor budget, BF16, batch size and
warmup/timing scope. `run.py` produces per-page tensor comparisons, target-native
rankings, retained states and raw/replay tensor banks in a new output directory.
`verify_tensor_payloads.py` checks your generated banks without assuming equality.

For a larger gallery use `full_collection/`; for repeated timing use its
`repeat_timing.py`. `official_upgrade/` contains the ColQwen2 release-pair runner,
with Energy input preparation in `data_preparation/`. `large_scale/` contains
streaming reconstruction and serving-index construction utilities.
