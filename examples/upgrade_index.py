"""Runnable version-planning and immutable-publication walkthrough."""

from __future__ import annotations

import tempfile
from pathlib import Path

from reprforge import (
    MaterializationOption,
    VersionManifest,
    break_even_upgrades,
    choose_materializations,
    cut_leverage,
    publish_generation,
    resolve_active_generation,
    seal_generation,
)

active = VersionManifest(
    source="pages-sha256:v1",
    processor="processor-sha256:v1",
    vision="vision-sha256:v1",
    base_embedding="embedding-sha256:v1",
    adapter="adapter-sha256:v1",
    projection="projection-sha256:v1",
    index_policy="sq8-v1",
)
target = VersionManifest(
    **{
        **active.to_dict(),
        "adapter": "adapter-sha256:v2",
        "projection": "projection-sha256:v2",
    }
)
update = active.update_scenario(target, "adapter-v2")

# Before fitting any codec: how much of a raw encode happens before the cut?
# (ArxivQA on one A100, batch size 1.)
leverage = cut_leverage(
    preprocess_seconds=0.132, prefix_seconds=0.926, suffix_seconds=0.193
)

# Replace these illustrative measurements with one collection's calibration.
post_vision = MaterializationOption(
    name="post_vision_ir",
    depends_on=frozenset({"processor", "vision", "base_embedding"}),
    storage_bytes=386_000_000,
    replay_seconds=32.2,
    materialization_seconds=20.0,
    quality_fraction=0.995,
)
decision = choose_materializations(
    (post_vision,),
    (update,),
    raw_rebuild_seconds=217.8,
    storage_budget_bytes=400_000_000,
    minimum_quality_fraction=0.99,
)
assert decision.routes[0].source == "post_vision_ir"

with tempfile.TemporaryDirectory(prefix="reprforge-generation-") as temporary:
    deployment = Path(temporary)
    generation = deployment / "generations" / "adapter-v2"
    (generation / "terminal").mkdir(parents=True)
    (generation / "serving").mkdir()
    (generation / "terminal" / "vectors.bin").write_bytes(b"target vectors")
    (generation / "serving" / "candidate.bin").write_bytes(b"candidate index")

    seal_generation(
        deployment,
        "adapter-v2",
        ("terminal/vectors.bin", "serving/candidate.bin"),
        version=target,
    )
    publish_generation(deployment, "adapter-v2")
    active_path, manifest = resolve_active_generation(deployment)

    print("changed components:", sorted(update.changed_components))
    print("cut leverage:", f"{leverage:.3f}")
    upgrades = break_even_upgrades(
        saved_seconds_per_upgrade=217.8 - 32.2,
        gpu_usd_per_hour=1.9,
        retained_gb=0.386,
        storage_usd_per_gb_month=0.023,
        horizon_months=3,
    )
    print("break-even upgrades (A100, object storage, 3 mo):", f"{upgrades:.2f}")
    print("selected rebuild source:", decision.routes[0].source)
    print("expected saving:", f"{decision.saving_fraction:.1%}")
    print("active generation:", active_path.name)
    print("published adapter:", manifest.version.adapter)
