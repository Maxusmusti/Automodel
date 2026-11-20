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

"""Linear-projection losses must reproduce the model's post-projection logit transform (or refuse to run)."""

import pytest
import torch
import torch.nn.functional as F
from transformers import Gemma2Config, GraniteConfig

from nemo_automodel.components.loss.chunked_ce import ChunkedCrossEntropy
from nemo_automodel.components.loss.utils import calculate_loss

pytestmark = pytest.mark.cpu_only


def _model(config):
    model = torch.nn.Module()
    model.config = config
    model.lm_head = torch.nn.Linear(16, 32, bias=False)
    return model


def test_granite_logits_scaling_matches_scaled_logits():
    """Granite divides logits by ``logits_scaling``; the loss must see the same logits as the model."""
    torch.manual_seed(0)
    model = _model(GraniteConfig(hidden_size=16, vocab_size=32, logits_scaling=16.0))
    hidden = torch.randn(2, 5, 16, requires_grad=True)
    labels = torch.randint(0, 32, (2, 5))

    actual = calculate_loss(ChunkedCrossEntropy(4, compile=False), model=model, hidden_states=hidden, labels=labels)
    logits = model.lm_head(hidden) / 16.0
    expected = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="sum")

    torch.testing.assert_close(actual, expected)
    unscaled = F.cross_entropy(model.lm_head(hidden).flatten(0, 1), labels.flatten(), reduction="sum")
    assert not torch.allclose(actual, unscaled)


def test_final_logit_softcapping_is_rejected():
    model = _model(Gemma2Config(hidden_size=16, vocab_size=32, final_logit_softcapping=30.0))
    with pytest.raises(ValueError, match="final_logit_softcapping"):
        calculate_loss(
            ChunkedCrossEntropy(4, compile=False),
            model=model,
            hidden_states=torch.randn(1, 3, 16),
            labels=torch.zeros(1, 3, dtype=torch.long),
        )


def test_fused_softcapping_must_match_model():
    """FusedLinearCrossEntropy applies softcapping itself, so it is accepted only with the model's value."""
    from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
    from nemo_automodel.components.loss.utils import _apply_output_logit_scaling

    model = _model(Gemma2Config(hidden_size=16, vocab_size=32, final_logit_softcapping=30.0))
    hidden = torch.randn(1, 3, 16)
    matching = FusedLinearCrossEntropy.__new__(FusedLinearCrossEntropy)
    matching.logit_softcapping = 30.0
    assert _apply_output_logit_scaling(matching, model, hidden) is hidden
    missing = FusedLinearCrossEntropy.__new__(FusedLinearCrossEntropy)
    missing.logit_softcapping = 0
    with pytest.raises(ValueError, match="set loss_fn.logit_softcapping=30.0"):
        _apply_output_logit_scaling(missing, model, hidden)
