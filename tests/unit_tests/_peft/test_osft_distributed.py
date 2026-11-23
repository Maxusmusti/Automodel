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

Every configuration runs two optimizer steps with the recipe's ordering (accumulate, project, clip with
``scale_grads_and_clip_grad_norm``, step, re-project parameters) and compares the pre-clip global gradient norm,
the projected (and clipped) gradients, and the post-step parameters against the same steps on one process with
the full batch. The grid covers:

- gradient accumulation 1 and 2, with an uneven final window (3 local samples split 2 + 1, loss normalized by
  the global sample count as the recipe normalizes by label tokens);
- no clipping, projection before clipping, and projection after clipping (with a threshold the norm exceeds);
- activation checkpointing off and on (the OSFT layers inside a checkpointed block).

It also exercises the collective paths: round-robin SVD with factor broadcast, the batched coefficient
all-reduce of both projections, and the row-sharded merged weight, including uneven ``Shard(0)`` splits.
"""

import itertools

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, init_device_mesh

from nemo_automodel.components._peft.osft import OSFTConfig, OSFTProjector, apply_osft, initialize_osft
from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm

pytestmark = pytest.mark.cpu_only

_WORLD_SIZE = 2
_LOCAL_BATCH = 3
_STEPS = 2
_MAX_NORM = 0.05  # below the measured pre-clip norms, so clipping binds (asserted)
_CONFIGS = [
    {"accum": accum, "clip": clip, "ac": ac}
    for accum, clip, ac in itertools.product((1, 2), (None, "before", "after"), (False, True))
]


def _cfg_id(cfg):
    return f"accum{cfg['accum']}-clip_{cfg['clip']}-ac_{cfg['ac']}"


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = nn.Linear(6, 11, bias=True)  # 11 rows: uneven split over 2 ranks
        self.down_proj = nn.Linear(11, 6, bias=False)

    def forward(self, x):
        return x + self.down_proj(torch.relu(self.up_proj(x)))


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = _Block()
        self.head = nn.Linear(6, 3, bias=False)  # non-target, stays trainable

    def forward(self, x):
        return self.head(self.block(x))


def _model(ac: bool) -> nn.Module:
    torch.manual_seed(0)
    model = _Tiny()
    apply_osft(model, OSFTConfig(unfreeze_rank_ratio=0.5, target_modules=["*.up_proj", "*.down_proj"]))
    if ac:
        model.block = checkpoint_wrapper(model.block)
    return model


def _data(step: int):
    gen = torch.Generator().manual_seed(100 + step)
    n = _WORLD_SIZE * _LOCAL_BATCH
    return torch.randn(n, 6, generator=gen), torch.randn(n, 3, generator=gen)


def _full(t):
    return (t.full_tensor() if isinstance(t, DTensor) else t).detach().clone()


def _key(name: str) -> str:
    return name.replace("._checkpoint_wrapped_module", "")


def _train(model, cfg, batches, device_mesh=None, dp_size=1):
    """Runs the recipe's optimizer-step ordering; returns per-step norms, gradients and parameters.

    FSDP2 averages gradients over the data-parallel group, so, as in the recipe, a loss normalized by the global
    count is multiplied by the DP size to make the reduced gradient the global-batch gradient.
    """
    projector = OSFTProjector(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2, weight_decay=0.01)
    history = []
    for x, y, total in batches:
        for mx, my in zip(x, y):  # microbatches; loss normalized by the global sample count
            loss = torch.nn.functional.mse_loss(model(mx), my, reduction="sum") * dp_size / (total * my.shape[-1])
            loss.backward()
        if cfg["clip"] == "before":
            projector.project_gradients()
        norm = scale_grads_and_clip_grad_norm(
            _MAX_NORM if cfg["clip"] else None, [model], device_mesh=device_mesh, foreach=True
        )
        if cfg["clip"] != "before":
            projector.project_gradients()
        grads = {_key(n): _full(p.grad) for n, p in model.named_parameters() if p.grad is not None}
        optimizer.step()
        optimizer.zero_grad()
        projector.project_parameters()
        params = {_key(n): _full(p) for n, p in model.named_parameters()}
        history.append({"norm": float(norm), "grads": grads, "params": params})
    merged = {_key(n): _full(m.merged_weight()) for n, m in model.named_modules() if hasattr(m, "merged_weight")}
    return {"history": history, "merged": merged}


def _microbatches(x, y, accum):
    if accum == 1:
        return [x], [y]
    return [x[:2], x[2:]], [y[:2], y[2:]]  # uneven final window: 2 + 1


def _worker(rank: int, init_file: str, output_file: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=_WORLD_SIZE)
    try:
        mesh = init_device_mesh("cpu", (_WORLD_SIZE,), mesh_dim_names=("dp",))
        results = {}
        for cfg in _CONFIGS:
            model = _model(cfg["ac"])
            fully_shard(model, mesh=mesh)
            initialize_osft(model)
            # Guard against a vacuous pass: the factors are sharded and the projector takes the collective path.
            assert isinstance(model.block.up_proj.osft_U_low, DTensor)
            assert OSFTProjector(model).distributed
            local = slice(rank * _LOCAL_BATCH, (rank + 1) * _LOCAL_BATCH)
            batches = []
            for step in range(_STEPS):
                x, y = _data(step)
                xs, ys = _microbatches(x[local], y[local], cfg["accum"])
                batches.append((xs, ys, x.shape[0]))
            results[_cfg_id(cfg)] = _train(model, cfg, batches, device_mesh=mesh, dp_size=_WORLD_SIZE)
        if rank == 0:
            torch.save(results, output_file)
    finally:
        dist.destroy_process_group()


def _reference(cfg):
    model = _model(cfg["ac"])
    initialize_osft(model)
    batches = []
    for step in range(_STEPS):
        x, y = _data(step)
        batches.append(([x], [y], x.shape[0]))
    return _train(model, cfg, batches)


@pytest.fixture(scope="module")
def distributed_results(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("osft_fsdp2")
    init_file, output_file = tmp / "pg", tmp / "result.pt"
    mp.spawn(_worker, args=(str(init_file), str(output_file)), nprocs=_WORLD_SIZE, join=True)
    return torch.load(output_file)


@pytest.mark.parametrize("cfg", _CONFIGS, ids=_cfg_id)
def test_osft_fsdp2_two_ranks_matches_single_process(distributed_results, cfg):
    dist_result, ref = distributed_results[_cfg_id(cfg)], _reference(cfg)
    for step, (d, r) in enumerate(zip(dist_result["history"], ref["history"])):
        torch.testing.assert_close(d["norm"], r["norm"], rtol=1e-5, atol=1e-7, msg=f"step {step}: grad norm")
        if cfg["clip"]:
            assert r["norm"] > _MAX_NORM, f"step {step}: clipping did not bind"
        for section in ("grads", "params"):
            assert d[section].keys() == r[section].keys(), f"step {step}: {section} keys"
            for name, value in r[section].items():
                torch.testing.assert_close(
                    d[section][name], value, rtol=1e-4, atol=1e-6, msg=f"step {step}: {section}:{name}"
                )
    for name, value in ref["merged"].items():
        torch.testing.assert_close(dist_result["merged"][name], value, rtol=1e-4, atol=1e-6, msg=f"merged:{name}")
    # Projected low-rank updates stay orthogonal to the frozen subspaces after every step.
    params = dist_result["history"][-1]["params"]
    for prefix in ("block.up_proj", "block.down_proj"):
        for side in ("U", "V"):
            overlap = params[f"{prefix}.osft_{side}_high"].t() @ params[f"{prefix}.osft_{side}_low"]
            assert overlap.abs().max() < 1e-5


def test_clip_order_changes_the_update(distributed_results):
    """Both clip orders are exercised and are genuinely different code paths (not a vacuous parametrization)."""
    before = distributed_results[_cfg_id({"accum": 1, "clip": "before", "ac": False})]["history"][0]["norm"]
    after = distributed_results[_cfg_id({"accum": 1, "clip": "after", "ac": False})]["history"][0]["norm"]
    assert before < after  # projection removes the frozen-subspace component, so the projected norm is smaller
