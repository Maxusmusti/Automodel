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

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components._peft.osft import (
    OSFTConfig,
    OSFTLinear,
    OSFTProjector,
    _choose_path,
    apply_osft,
    initialize_osft,
)


def _make_layer(n=48, m=32, urr=0.25, mode="auto", bias=False, seed=0):
    gen = torch.Generator().manual_seed(seed)
    weight = torch.randn(n, m, generator=gen, dtype=torch.float64)
    cfg = OSFTConfig(unfreeze_rank_ratio=urr, forward_mode=mode)
    layer = OSFTLinear(m, n, cfg.rank_high(n, m), bias=bias, forward_mode=mode, dtype=torch.float64)
    layer._base_weight = weight.clone()
    if bias:
        with torch.no_grad():
            layer.bias.copy_(torch.randn(n, generator=gen, dtype=torch.float64))
    initialize_osft(layer)
    return layer, weight


@pytest.mark.parametrize(
    ("n", "m", "urr", "expected"),
    [(4096, 4096, 0.25, 3072), (1024, 4096, 0.25, 768), (12288, 4096, 0.5, 2048), (8, 8, 0.01, 7)],
)
def test_rank_high_matches_mini_trainer_rule(n, m, urr, expected):
    assert OSFTConfig(unfreeze_rank_ratio=urr).rank_high(n, m) == expected


@pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1, 1.5])
def test_invalid_ratio_rejected(ratio):
    with pytest.raises(ValueError):
        OSFTConfig(unfreeze_rank_ratio=ratio)


@pytest.mark.parametrize("mode", ["reconstruct", "factored"])
@pytest.mark.parametrize(("n", "m"), [(48, 32), (32, 48), (40, 40)])
def test_forward_equals_dense_linear_at_init(mode, n, m):
    layer, weight = _make_layer(n, m, mode=mode, bias=True)
    x = torch.randn(3, 5, m, dtype=torch.float64)
    torch.testing.assert_close(layer(x), F.linear(x, weight, layer.bias), rtol=1e-10, atol=1e-10)


def test_reconstruct_backward_matches_factored_autograd():
    recon, _ = _make_layer(mode="reconstruct")
    fact, _ = _make_layer(mode="factored")
    x = torch.randn(2, 7, 32, dtype=torch.float64)
    grad_out = torch.randn(2, 7, 48, dtype=torch.float64)
    recon(x).backward(grad_out)
    fact(x).backward(grad_out)
    for name in ("osft_U_low", "osft_S_low", "osft_V_low"):
        torch.testing.assert_close(getattr(recon, name).grad, getattr(fact, name).grad, rtol=1e-9, atol=1e-9)
    for name in ("osft_U_high", "osft_S_high", "osft_V_high"):
        assert getattr(recon, name).grad is None


def test_gradient_projection_removes_frozen_components():
    layer, _ = _make_layer()
    layer(torch.randn(4, 32, dtype=torch.float64)).sum().backward()
    OSFTProjector(layer).project_gradients()
    assert (layer.osft_U_high.t() @ layer.osft_U_low.grad).abs().max() < 1e-10
    assert (layer.osft_V_high.t() @ layer.osft_V_low.grad).abs().max() < 1e-10


def test_parameter_projection_removes_leak_and_is_idempotent():
    layer, _ = _make_layer()
    proj = OSFTProjector(layer)
    with torch.no_grad():
        layer.osft_U_low.add_(0.1 * torch.randn_like(layer.osft_U_low))
        layer.osft_V_low.add_(0.1 * torch.randn_like(layer.osft_V_low))
    proj.project_parameters()
    assert (layer.osft_U_high.t() @ layer.osft_U_low).abs().max() < 1e-10
    assert (layer.osft_V_high.t() @ layer.osft_V_low).abs().max() < 1e-10
    u_low = layer.osft_U_low.detach().clone()
    proj.project_parameters()
    torch.testing.assert_close(layer.osft_U_low.detach(), u_low, rtol=0, atol=1e-12)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(16, 16, bias=True)
        self.down_proj = nn.Linear(24, 16, bias=False)
        self.norm = nn.LayerNorm(16)
        self.lm_head = nn.Linear(16, 30, bias=False)


def test_apply_osft_replaces_only_targets():
    model = _Block()
    assert apply_osft(model, OSFTConfig(target_modules=["q_proj", "down_proj"])) == 2
    assert isinstance(model.q_proj, OSFTLinear) and model.q_proj.bias is not None
    assert isinstance(model.down_proj, OSFTLinear) and model.down_proj.bias is None
    assert type(model.lm_head) is nn.Linear and type(model.norm) is nn.LayerNorm
    # non-target parameters stay trainable; frozen factors do not
    assert model.lm_head.weight.requires_grad
    assert not model.q_proj.osft_U_high.requires_grad and model.q_proj.osft_U_low.requires_grad


def test_apply_osft_rejects_non_linear_target_and_empty_match():
    with pytest.raises(TypeError):
        apply_osft(_Block(), OSFTConfig(target_modules=["norm"]))
    with pytest.raises(ValueError):
        apply_osft(_Block(), OSFTConfig(target_modules=["does_not_exist"]))


def test_state_dict_base_load_then_merged_export():
    base = nn.Linear(32, 48, bias=True).double()
    layer = OSFTLinear(32, 48, OSFTConfig().rank_high(48, 32), bias=True, dtype=torch.float64)
    # before initialization the dense key is a template a checkpoint loader can fill
    assert layer.state_dict()["weight"].shape == (48, 32)
    layer.load_state_dict(base.state_dict(), strict=False)
    assert layer.weight is not None
    initialize_osft(layer)
    assert layer.weight is None
    exported = layer.state_dict()
    torch.testing.assert_close(exported["weight"], base.weight.detach(), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(exported["bias"], base.bias.detach())


def test_state_dict_with_factors_resumes_exactly():
    src, _ = _make_layer(bias=True)
    with torch.no_grad():
        src.osft_U_low.add_(0.01)
    dst = OSFTLinear(32, 48, src.rank_high, bias=True, dtype=torch.float64)
    dst.load_state_dict(src.state_dict())
    assert dst._osft_initialized and dst.weight is None
    for name, value in src.named_parameters():
        torch.testing.assert_close(getattr(dst, name), value)


@pytest.mark.parametrize(
    ("tokens", "n", "m", "rank_high", "rank_low", "expected"),
    [
        (16384, 4096, 4096, 3072, 1024, "reconstruct"),
        (4096, 4096, 4096, 3072, 1024, "factored"),
        (16384, 12288, 4096, 3072, 1024, "factored"),
        (16384, 1024, 4096, 768, 256, "reconstruct"),
    ],
)
def test_choose_path_matches_measured_winner(tokens, n, m, rank_high, rank_low, expected):
    assert _choose_path(tokens, n, m, rank_high, rank_low) == expected


def test_apply_osft_carries_over_materialized_weights():
    """Weights loaded before the swap (e.g. weights_already_loaded) must survive into the factors."""
    model = _Block().double()
    ref_q = (model.q_proj.weight.detach().clone(), model.q_proj.bias.detach().clone())
    apply_osft(model, OSFTConfig(target_modules=["q_proj"]))
    torch.testing.assert_close(model.q_proj.bias.detach(), ref_q[1])
    initialize_osft(model)
    x = torch.randn(5, 16, dtype=torch.float64)
    torch.testing.assert_close(model.q_proj(x), F.linear(x, *ref_q), rtol=1e-10, atol=1e-10)


def test_strict_load_still_reports_unexpected_keys():
    layer, _ = _make_layer(bias=True)
    state = layer.state_dict()
    state["not_a_param"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="not_a_param"):
        layer.load_state_dict(state, strict=True)


def test_missing_base_weight_explains_from_config():
    layer = OSFTLinear(32, 48, 24, bias=False)
    with pytest.raises(RuntimeError, match="from_pretrained"):
        initialize_osft(layer)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU autocast")
def test_reconstruct_path_works_under_autocast():
    """fp32 factors + bf16 autocast: backward must run under the forward's autocast state."""
    torch.manual_seed(0)
    weight = torch.randn(48, 32, device="cuda")
    layer = OSFTLinear(32, 48, 24, bias=False, forward_mode="reconstruct", device="cuda", dtype=torch.float32)
    layer._base_weight = weight.clone()
    initialize_osft(layer)
    x = torch.randn(4, 32, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = layer(x)
    y.float().sum().backward()
    assert y.dtype == torch.bfloat16
    assert layer.osft_U_low.grad is not None and torch.isfinite(layer.osft_U_low.grad).all()
