"""Shared setup for the ColQwen command-line tools."""

import hashlib
import importlib.metadata
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup(config, spec, max_pixels):
    import torch
    from colpali_engine.models import ColQwen2_5_Processor

    from reprforge.integrations.colqwen import load_model

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_cudnn_sdp(False)
    processor = ColQwen2_5_Processor.from_pretrained(
        config["processor_model"], local_files_only=True
    )
    processor.image_processor.max_pixels = max_pixels
    processor.image_processor.size["longest_edge"] = max_pixels
    model, base, _ = load_model(spec, config, torch)
    return torch, processor, model, base


def contract(base, processor):
    """Conservative identity of the actual loaded visual prefix and input recipe."""
    import torch

    from reprforge.integrations import colqwen

    vision = hashlib.sha256()
    for name, tensor in sorted(base.visual.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        vision.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        vision.update(value.view(torch.uint8).numpy().tobytes())
    token = processor.tokenizer
    processing = {
        "processor": processor.to_dict(),
        "image_processor": processor.image_processor.to_dict(),
        "tokenizer": token.backend_tokenizer.to_str(),
        "special_tokens": token.special_tokens_map,
        "template": getattr(processor, "chat_template", None),
    }
    # Path labels do not identify computation. Everything else fails closed.
    model_config = base.config.to_dict()
    model_config.pop("_name_or_path", None)
    return {
        "vision": vision.hexdigest(),
        "processor": hashlib.sha256(
            json.dumps(processing, sort_keys=True, default=str).encode()
        ).hexdigest(),
        "model_config": model_config,
        "continuation_code": digest(colqwen.__file__),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "colpali-engine", "peft")
        },
        "dtype": "bfloat16",
    }


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")
