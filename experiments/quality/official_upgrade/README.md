# Official-upgrade quality

Use the [quality runner](../README.md) with `--family colqwen2`, this directory's
`protocol.json` and `targets.json`, and local ColQwen2 source/base/processor paths.
Replace the input-path templates with your prepared files. The source is v0.1;
the target is v1.0. Keep the protocol's processor budget and gallery definition.

Run `analyze.py --help` to aggregate your output with an explicit query mask.
The target reference shares a visual prefix; independent raw reconstruction
is evaluated by `experiments/reconstruction/official_upgrade`.
