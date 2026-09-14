"""Inspect the dependency scope of a small synthetic adapter on CPU."""
from reprforge import inspect_adapter_tensor_keys


def main():
    suffix = inspect_adapter_tensor_keys([
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
    ])
    upstream = inspect_adapter_tensor_keys([
        "base_model.model.visual.patch_embed.proj.lora_A.weight",
    ])
    assert suffix.post_vision_cut_legal
    assert not upstream.post_vision_cut_legal
    print("Decoder-only adapter: visual state remains eligible.")
    print("Visual adapter: reject replay until the upstream contract is established.")
    print("Tensor names alone do not verify processor or base-weight equality.")


if __name__ == "__main__":
    main()
