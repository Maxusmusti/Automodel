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

"""ModelState contracts for OSFT layers: base-load template, resumable save, and exact resume."""

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components._peft.osft import OSFTConfig, apply_osft, initialize_osft
from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState

pytestmark = pytest.mark.cpu_only


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(16, 24, bias=True)
        self.norm = nn.LayerNorm(24)


def _osft_model(materialize: bool = True) -> tuple[nn.Module, dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    model = _Tiny()
    base = {k: v.detach().clone() for k, v in model.state_dict().items()}
    apply_osft(model, OSFTConfig(target_modules=["q_proj"]))
    if materialize:
        initialize_osft(model)
    return model, base


def test_init_step_template_has_dense_weight_and_no_factors():
    model, base = _osft_model(materialize=False)
    template = ModelState(model, is_init_step=True).state_dict()

    assert not any(".osft_" in key for key in template)
    assert template["q_proj.weight"].shape == base["q_proj.weight"].shape
    assert {"q_proj.weight", "q_proj.bias", "norm.weight", "norm.bias"} <= set(template)


def test_resumable_save_carries_factors_and_merged_weight():
    model, base = _osft_model()
    saved = ModelState(model).state_dict()

    assert {"q_proj.osft_U_high", "q_proj.osft_U_low", "q_proj.osft_S_low", "q_proj.osft_V_low"} <= set(saved)
    torch.testing.assert_close(saved["q_proj.weight"], base["q_proj.weight"], rtol=1e-5, atol=1e-6)


def test_resume_restores_factors_exactly():
    model, _ = _osft_model()
    with torch.no_grad():
        model.q_proj.osft_U_low.add_(0.01)
    saved = {k: v.clone() for k, v in ModelState(model).state_dict().items()}

    fresh, _ = _osft_model(materialize=False)
    ModelState(fresh).load_state_dict(saved)

    for name, value in model.q_proj.named_parameters():
        torch.testing.assert_close(getattr(fresh.q_proj, name), value, rtol=0, atol=0)
