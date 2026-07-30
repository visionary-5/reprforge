"""ColPali architecture used by the official MMDocIR 2025 artifact.

The current ``colpali_engine`` class has a different PaliGemma module nesting,
so loading MMDocIR's published PEFT adapter into it silently leaves every LoRA
weight unmatched.  This small compatibility class reproduces the architecture
defined in MMDocIR commit 394d84b, without importing the artifact's unrelated
retriever wrappers.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers.models.paligemma.modeling_paligemma import (
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaPreTrainedModel,
)


class MMDocIRColPali(PaliGemmaPreTrainedModel):
    """Exact projection wrapper expected by the published MMDocIR adapter."""

    def __init__(self, config: PaliGemmaConfig):
        super().__init__(config=config)
        # Transformers 4.53 wraps the multimodal body in an additional
        # ``PaliGemmaForConditionalGeneration.model`` level.  MMDocIR's 4.42
        # checkpoint stores keys as ``model.language_model...``.  Retaining
        # only the inner body restores that published state-dict contract and
        # also omits the unused language-model head.
        model = PaliGemmaForConditionalGeneration(config=config).model
        if model.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [
                f"model.language_model.{key}"
                for key in model.language_model._tied_weights_keys
            ]
        self.model = model
        self.dim = 128
        self.custom_text_proj = nn.Linear(
            self.model.config.text_config.hidden_size,
            self.dim,
        )
        self.post_init()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        kwargs.pop("output_hidden_states", None)
        outputs = self.model(*args, output_hidden_states=True, **kwargs)
        last_hidden_states = outputs.hidden_states[-1]
        projection = self.custom_text_proj(last_hidden_states)
        projection = projection / projection.norm(dim=-1, keepdim=True)
        return projection * kwargs["attention_mask"].unsqueeze(-1)
