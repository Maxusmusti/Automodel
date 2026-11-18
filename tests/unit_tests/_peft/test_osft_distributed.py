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

"""OSFT under real FSDP2 (2 gloo ranks on CPU) matches a single-process fp32 reference.

Exercises the collective paths: round-robin SVD with factor broadcast, the batched coefficient all-reduce of
gradient and parameter projection, and the row-sharded merged weight, including uneven Shard(0) splits.
"""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, init_device_mesh

from nemo_automodel.components._peft.osft import OSFTConfig, OSFTProjector, apply_osft, initialize_osft

pytestmark = pytest.mark.cpu_only

_WORLD_SIZE = 2
_BATCH = 4


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = nn.Linear(6, 11, bias=True)  # 11 rows: uneven split over 2 ranks
        self.down_proj = nn.Linear(11, 6, bias=False)
        self.head = nn.Linear(6, 3, bias=False)  # non-target, stays trainable

    def forward(self, x):
        return self.head(self.down_proj(torch.relu(self.up_proj(x))))


def _model() -> nn.Module:
    torch.manual_seed(0)
    model = _Tiny()
    apply_osft(model, OSFTConfig(unfreeze_rank_ratio=0.5, target_modules=["up_proj", "down_proj"]))
    return model


def _data():
    gen = torch.Generator().manual_seed(1)
    return torch.randn(_WORLD_SIZE * _BATCH, 6, generator=gen), torch.randn(_WORLD_SIZE * _BATCH, 3, generator=gen)


def _step(model, x, y):
    """One OSFT step; returns full (unsharded) copies of projected grads and post-step params."""
    full = lambda t: (t.full_tensor() if isinstance(t, DTensor) else t).detach().clone()  # noqa: E731
    factors = {name: full(p) for name, p in model.named_parameters()}
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    projector = OSFTProjector(model)
    projector.project_gradients()
    grads = {name: full(p.grad) for name, p in model.named_parameters() if p.grad is not None}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    optimizer.step()
    projector.project_parameters()
    params = {name: full(p) for name, p in model.named_parameters()}
    merged = {name: full(m.merged_weight()) for name, m in model.named_modules() if hasattr(m, "merged_weight")}
    return {"factors": factors, "grads": grads, "params": params, "merged": merged}


def _worker(rank: int, init_file: str, output_file: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=_WORLD_SIZE)
    try:
        mesh = init_device_mesh("cpu", (_WORLD_SIZE,), mesh_dim_names=("dp",))
        model = _model()
        fully_shard(model, mesh=mesh)
        initialize_osft(model)
        # Guard against a vacuous pass: the factors are sharded and the projector takes the collective path.
        assert isinstance(model.up_proj.osft_U_low, DTensor)
        assert OSFTProjector(model).distributed
        x, y = _data()
        local = slice(rank * _BATCH, (rank + 1) * _BATCH)
        result = _step(model, x[local], y[local])
        if rank == 0:
            torch.save(result, output_file)
    finally:
        dist.destroy_process_group()


def _reference():
    model = _model()
    initialize_osft(model)
    x, y = _data()
    return _step(model, x, y)


def test_osft_fsdp2_two_ranks_matches_single_process(tmp_path):
    init_file, output_file = tmp_path / "pg", tmp_path / "result.pt"
    mp.spawn(_worker, args=(str(init_file), str(output_file)), nprocs=_WORLD_SIZE, join=True)
    dist_result = torch.load(output_file)
    ref = _reference()

    for section in ("factors", "grads", "params", "merged"):
        assert dist_result[section].keys() == ref[section].keys(), section
        for name, value in ref[section].items():
            torch.testing.assert_close(dist_result[section][name], value, rtol=1e-4, atol=1e-5, msg=f"{section}:{name}")
    # Projected low-rank updates stay orthogonal to the frozen subspaces after the step.
    for prefix in ("up_proj", "down_proj"):
        p = dist_result["params"]
        for side in ("U", "V"):
            overlap = p[f"{prefix}.osft_{side}_high"].t() @ p[f"{prefix}.osft_{side}_low"]
            assert overlap.abs().max() < 1e-5
