import pytest

from reprforge import (
    MaterializationOption,
    choose_materializations,
    classify_tensor,
    inspect_adapter_tensor_keys,
)

# Tensor names taken from public checkpoints (safetensors headers, 2026-09-04).
COLQWEN25_ADAPTER = [
    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
    "base_model.model.model.layers.35.mlp.down_proj.lora_B.weight",
    "base_model.model.custom_text_proj.lora_A.weight",
    "base_model.model.custom_text_proj.lora_B.weight",
]
COLSMOL_ADAPTER = [
    "base_model.model.model.text_model.layers.0.mlp.down_proj.lora_A.weight",
    "base_model.model.model.text_model.layers.23.self_attn.v_proj.lora_B.weight",
]
COLMODERNVBERT_ADAPTER = [
    "model.text_model.layers.9.mlp.Wo.lora_B.weight",
    "model.custom_text_proj.lora_A.weight",
]
TURKISH_COLPALI_VISION_LORA = [
    "base_model.model.vision_tower.vision_model.encoder.layers.0."
    "self_attn.q_proj.lora_A.weight",
    "base_model.model.language_model.model.layers.0.self_attn.q_proj.lora_A.weight",
]


@pytest.mark.parametrize(
    "keys", [COLQWEN25_ADAPTER, COLSMOL_ADAPTER, COLMODERNVBERT_ADAPTER]
)
def test_public_decoder_only_adapters_keep_the_post_vision_cut(keys) -> None:
    scope = inspect_adapter_tensor_keys(keys)

    assert scope.post_vision_cut_legal
    assert scope.vision_tensors == 0 and scope.merger_tensors == 0
    assert "adapter" in scope.changed_components
    assert "vision" not in scope.changed_components


def test_vision_lora_invalidates_the_post_vision_cut_and_routes_raw() -> None:
    scope = inspect_adapter_tensor_keys(TURKISH_COLPALI_VISION_LORA)
    decision = choose_materializations(
        (
            MaterializationOption(
                name="post_vision",
                depends_on=frozenset({"processor", "vision", "base_embedding"}),
                storage_bytes=100,
                replay_seconds=1,
            ),
        ),
        (scope.to_update_scenario("domain_adapter"),),
        raw_rebuild_seconds=10,
        storage_budget_bytes=100,
    )

    assert not scope.post_vision_cut_legal
    assert scope.post_vision_cut_blockers == ("1 tensors update the vision tower",)
    assert decision.selected == ()
    assert decision.routes[0].source == "raw"


def test_merger_counts_as_visual_prefix() -> None:
    scope = inspect_adapter_tensor_keys(["visual.merger.mlp.0.weight"])

    assert classify_tensor("visual.merger.mlp.0.weight") == "merger"
    assert scope.changed_components == frozenset({"vision"})
    assert not scope.post_vision_cut_legal


def test_qwen_visual_blocks_are_vision() -> None:
    assert classify_tensor("visual.blocks.3.attn.qkv.weight") == "vision"
    assert classify_tensor("visual.patch_embed.proj.weight") == "vision"


def test_unknown_paths_fail_closed() -> None:
    scope = inspect_adapter_tensor_keys(["base_model.model.new_block.weight"])

    assert not scope.post_vision_cut_legal
    assert scope.changed_components == frozenset(
        {"vision", "base_embedding", "adapter", "projection"}
    )


@pytest.mark.parametrize("keys", [[], [""], [None]])
def test_invalid_key_sets_are_rejected(keys) -> None:
    with pytest.raises(ValueError):
        inspect_adapter_tensor_keys(keys)  # type: ignore[arg-type]
