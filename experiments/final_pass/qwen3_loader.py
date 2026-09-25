"""Pinned publisher loaders for the existing Qwen3 4B pair; same adaptations as 20260918."""

from pathlib import Path
import torch
from transformers import AutoConfig, AutoProcessor, PreTrainedModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module

ROOT = None


def load(source):
    d = ROOT / (
        "TomoroAI__tomoro-colqwen3-embed-4b"
        if source
        else "OpenSearch-AI__Ops-Colqwen3-4B"
    )
    processor = AutoProcessor.from_pretrained(
        d, trust_remote_code=True, local_files_only=True
    )
    clsname = (
        "modeling_colqwen3.ColQwen3"
        if source
        else "modeling_ops_colqwen3.OpsColQwen3Model"
    )
    cls = get_class_from_dynamic_module(clsname, str(d), local_files_only=True)
    config = AutoConfig.from_pretrained(
        d, trust_remote_code=True, local_files_only=True
    )
    model, info = PreTrainedModel.from_pretrained.__func__(
        cls,
        d,
        config=config,
        key_mapping=cls._checkpoint_conversion_mapping,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    if source:

        def forbid(m, a):
            raise RuntimeError("Uninitialized generation head executed")

        model.vlm.lm_head.register_forward_pre_hook(forbid)
    else:
        model.dims = config.dims
    model.eval().cuda()
    backbone = model.vlm.model if source else model.qwen3vl
    if not source:

        def hook(m, a, kw):
            if (
                kw.get("image_grid_thw") is not None
                and kw.get("mm_token_type_ids") is None
            ):
                kw["mm_token_type_ids"] = (
                    kw["input_ids"] == model.config.image_token_id
                ).to(torch.int32)
            return a, kw

        backbone.register_forward_pre_hook(hook, with_kwargs=True)
    return model, processor, backbone
