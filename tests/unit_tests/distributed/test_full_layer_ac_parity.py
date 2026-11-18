# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Whole-layer activation checkpointing must be numerically identical to no / submodule checkpointing.

Covers every dense model that declares ``supports_full_layer_activation_checkpointing``.
"""

import pytest
import torch
from transformers import LlamaConfig, Qwen2Config, Qwen3Config

from nemo_automodel._transformers.model_capabilities import query_capabilities
from nemo_automodel.components.distributed.activation_checkpointing import (
    apply_full_layer_checkpointing_to_layers,
    apply_submodule_checkpointing,
)
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.llama.model import LlamaForCausalLM
from nemo_automodel.components.models.qwen2.model import Qwen2ForCausalLM
from nemo_automodel.components.models.qwen3.model import Qwen3ForCausalLM

_TINY = dict(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,
)
_MODELS = {
    "llama": (LlamaForCausalLM, LlamaConfig(**_TINY, head_dim=8)),
    "qwen2": (Qwen2ForCausalLM, Qwen2Config(**_TINY)),
    "qwen3": (Qwen3ForCausalLM, Qwen3Config(**_TINY, head_dim=8)),
}


def _build(name: str, ac: str | None) -> torch.nn.Module:
    model_cls, config = _MODELS[name]
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = model_cls(config, backend=BackendConfig(attn="sdpa", linear="torch", rms_norm="torch_fp32")).double()
    layers = list(model.model.layers)
    if ac == "full_layer":
        apply_full_layer_checkpointing_to_layers(model, layers)
    elif ac == "submodule":
        apply_submodule_checkpointing(layers, has_kv_sharing=False)
    return model


def _loss_and_grads(model: torch.nn.Module) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    torch.manual_seed(1)
    input_ids = torch.randint(0, _TINY["vocab_size"], (2, 12))
    logits = model(input_ids=input_ids).logits
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), input_ids[:, 1:].reshape(-1))
    loss.backward()
    grads = {
        name.replace("._checkpoint_wrapped_module", ""): p.grad.detach().clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    return loss.detach(), grads


@pytest.mark.parametrize("name", sorted(_MODELS))
def test_model_declares_full_layer_activation_checkpointing(name):
    assert query_capabilities(_MODELS[name][0]).supports_full_layer_activation_checkpointing is True


@pytest.mark.parametrize("name", sorted(_MODELS))
def test_full_layer_checkpointing_matches_submodule_and_no_checkpointing(name):
    ref_loss, ref_grads = _loss_and_grads(_build(name, None))
    for ac in ("submodule", "full_layer"):
        loss, grads = _loss_and_grads(_build(name, ac))
        torch.testing.assert_close(loss, ref_loss, rtol=1e-12, atol=1e-12)
        assert grads.keys() == ref_grads.keys(), f"{ac}: gradient set differs"
        for key, grad in grads.items():
            torch.testing.assert_close(grad, ref_grads[key], rtol=1e-10, atol=1e-12, msg=f"{ac}: {key}")
