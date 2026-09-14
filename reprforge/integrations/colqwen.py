"""ColQwen2.5 BF16 post-merger capture and target continuation."""

import json
import re
from pathlib import Path


def load_model(spec, config, torch):
    from colpali_engine.models import ColQwen2_5
    from peft import PeftConfig, get_peft_model
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

    from .projection import load_projection

    base = ColQwen2_5.from_pretrained(
        config["base_model"],
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    proj_path = Path(spec["projection"])
    if proj_path.suffix == ".json":
        dtype = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
        proj = {
            k: torch.frombuffer(bytearray.fromhex(v["bytes"]), dtype=dtype[v["dtype"]])
            .reshape(v["shape"])
            .clone()
            for k, v in json.loads(proj_path.read_text())["tensors"].items()
        }
    elif proj_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        proj = load_file(str(proj_path))
    else:
        proj = load_projection(proj_path, torch)
    # Install the frozen base head BEFORE loading adapter weights, including head LoRA.
    with torch.no_grad():
        for name in ("weight", "bias"):
            value = getattr(base.custom_text_proj, name)
            value.copy_(proj["custom_text_proj." + name].to(value))
    pc = PeftConfig.from_pretrained(spec["adapter"], local_files_only=True)
    linear = [n for n, m in base.named_modules() if isinstance(m, torch.nn.Linear)]
    target = pc.target_modules
    pc.target_modules = [
        n
        for n in linear
        if (
            re.fullmatch(target, n)
            if isinstance(target, str)
            else any(n == t or n.endswith("." + t) for t in target)
        )
    ]
    if not pc.target_modules:
        raise RuntimeError("No adapter targets resolved")
    model = get_peft_model(base, pc)
    weights = load_peft_weights(spec["adapter"], device="cpu", local_files_only=True)
    old, new = (
        "base_model.model.model.layers.",
        "base_model.model.language_model.layers.",
    )
    remapped = {
        (new + k[len(old) :] if k.startswith(old) else k): v for k, v in weights.items()
    }
    loaded = set_peft_model_state_dict(model, remapped, adapter_name="default")
    missing = [k for k in loaded.missing_keys if "lora_" in k]
    if missing or loaded.unexpected_keys:
        raise RuntimeError(
            f"Incomplete adapter load: {missing}, {loaded.unexpected_keys}"
        )
    model.to("cuda:0").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return (
        model,
        base,
        {
            "checkpoint_tensors": len(weights),
            "missing_lora": missing,
            "unexpected": loaded.unexpected_keys,
            "mask_non_image_embeddings": base.mask_non_image_embeddings,
        },
    )


def capture(base, batch, torch):
    ids, grid = batch["input_ids"], batch["image_grid_thw"]
    offsets = grid[:, 1] * grid[:, 2]
    pixels = torch.cat(
        [row[: int(o)] for row, o in zip(batch["pixel_values"], offsets, strict=False)]
    )
    vision = torch.cat(base.get_image_features(pixels, grid), dim=0)
    return {
        "input_ids": ids.cpu(),
        "attention_mask": batch["attention_mask"].cpu(),
        "image_grid_thw": grid.cpu(),
        "vision": vision.cpu(),
    }


def replay(base, state, torch):
    """Recreate target text embeddings/positions; never execute target vision."""
    state = {k: v.to("cuda:0") for k, v in state.items()}
    ids, grid, mask = (
        state["input_ids"],
        state["image_grid_thw"],
        state["attention_mask"],
    )
    embeds = base.get_input_embeddings()(ids)
    image_mask = ids == base.config.image_token_id
    embeds = embeds.masked_scatter(
        image_mask.unsqueeze(-1).expand_as(embeds), state["vision"].to(embeds)
    )
    pos, _ = base.get_rope_index(ids, grid, None, attention_mask=mask)
    output = base.language_model(
        input_ids=None,
        inputs_embeds=embeds,
        attention_mask=mask,
        position_ids=pos,
        cache_position=torch.arange(ids.shape[1], device=ids.device),
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    )
    vectors = base.custom_text_proj(output.last_hidden_state)
    vectors = vectors / vectors.norm(dim=-1, keepdim=True)
    vectors = vectors * mask.unsqueeze(-1)
    if base.mask_non_image_embeddings:
        vectors = vectors * image_mask.unsqueeze(-1)
    return vectors[0][mask[0].bool()]
