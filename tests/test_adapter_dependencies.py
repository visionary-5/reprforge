import pytest

from reprforge import (
    MaterializationOption,
    choose_materializations,
    inspect_adapter_tensor_keys,
)


def test_language_only_domain_adapter_preserves_post_vision_ir() -> None:
    keys = [
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
        "base_model.model.custom_text_proj.lora_A.weight",
        "base_model.model.custom_text_proj.lora_B.weight",
    ]

    scope = inspect_adapter_tensor_keys(keys)

    assert scope.post_vision_replay_valid
    assert scope.post_vision_replay_blockers == ()
    assert scope.changed_components == frozenset({"adapter", "projection"})
    assert scope.decoder_tensors == 2
    assert scope.projection_tensors == 2


def test_domain_adapter_that_touches_vision_invalidates_post_vision_ir() -> None:
    keys = [
        "base_model.model.vlm.vision_tower.vision_model.encoder.layers.0."
        "self_attn.q_proj.lora_A.weight",
        "base_model.model.vlm.language_model.model.layers.0."
        "self_attn.q_proj.lora_A.weight",
    ]
    scope = inspect_adapter_tensor_keys(keys)
    update = scope.to_update_scenario("domain_adapter")
    option = MaterializationOption(
        name="post_vision",
        depends_on=frozenset({"processor", "vision", "base_embedding"}),
        storage_bytes=100,
        replay_seconds=1,
    )

    decision = choose_materializations(
        (option,),
        (update,),
        raw_rebuild_seconds=10,
        storage_budget_bytes=100,
    )

    assert not scope.post_vision_replay_valid
    assert scope.vision_tensors == 1
    assert scope.post_vision_replay_blockers == (
        "1 adapter tensors update vision",
    )
    assert decision.selected == ()
    assert decision.routes[0].source == "raw"


def test_unknown_adapter_paths_fail_closed() -> None:
    scope = inspect_adapter_tensor_keys(["base_model.model.new_block.weight"])

    assert not scope.post_vision_replay_valid
    assert scope.post_vision_replay_blockers == (
        "1 adapter tensors have an unknown dependency",
    )
    assert scope.changed_components == frozenset(
        {"vision", "base_embedding", "adapter", "projection"}
    )


@pytest.mark.parametrize("keys", [[], [""], [None]])
def test_invalid_key_sets_are_rejected(keys: list[object]) -> None:
    with pytest.raises(ValueError):
        inspect_adapter_tensor_keys(keys)  # type: ignore[arg-type]
