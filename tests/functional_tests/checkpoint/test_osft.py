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

"""OSFT end to end under FSDP2: training, resumable checkpoint, bitwise resume, and HF export.

Launched by tests/functional_tests/osft/L2_OSFT_FSDP2_Checkpoint.sh with torchrun (2 ranks) and recipe overrides.
"""

import copy
import time
from pathlib import Path

import datasets
import torch
import torch.distributed as dist
from safetensors import safe_open
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM

from nemo_automodel.components._peft.osft import OSFTLinear
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

datasets.disable_caching()


def _full(t: torch.Tensor) -> torch.Tensor:
    return (t.full_tensor() if isinstance(t, DTensor) else t).detach().float().cpu()


def _osft_modules(model: torch.nn.Module) -> dict[str, OSFTLinear]:
    return {name: m for name, m in model.named_modules() if isinstance(m, OSFTLinear)}


def _read_safetensors_dir(path: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    for file in sorted(path.glob("*.safetensors")):
        with safe_open(str(file), "pt") as f:
            tensors.update({k: f.get_tensor(k) for k in f.keys()})
    return tensors


def test_osft_checkpoint():
    # Bitwise resume is only meaningful with deterministic kernels: flash-attn's backward and the MoE token
    # dispatch otherwise differ in the last bits between two identical steps (CUBLAS_WORKSPACE_CONFIG and
    # FLASH_ATTENTION_DETERMINISTIC are exported by L2_OSFT_FSDP2_Checkpoint.sh).
    torch.use_deterministic_algorithms(True, warn_only=True)
    script_path = Path(__file__).parents[3] / "examples" / "llm_finetune"
    cfg = parse_args_and_load_config(script_path / "llama3_2" / "llama3_2_1b_squad.yaml")
    trainer = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    trainer.setup()
    rank = dist.get_rank()

    # Before training, every OSFT layer reproduces its pretrained weight: the base checkpoint reached the layers
    # and was decomposed (regression check for loaders that echo the model's own factor tensors back).
    base = _read_safetensors_dir(Path(cfg.model.pretrained_model_name_or_path))
    initial = {name: _full(m.merged_weight()) for name, m in _osft_modules(trainer.model_parts[0]).items()}
    assert initial
    for name, weight in initial.items():
        torch.testing.assert_close(weight, base[f"{name}.weight"].float(), rtol=1e-4, atol=1e-5, msg=name)

    trainer.run_train_validation_loop()

    model = trainer.model_parts[0]
    modules = _osft_modules(model)
    assert modules, "OSFT replaced no modules"
    trained = {name: {p: _full(v) for p, v in m.named_parameters()} for name, m in modules.items()}
    merged = {name: _full(m.merged_weight()) for name, m in modules.items()}

    # Updates stayed orthogonal to the frozen subspaces through training.
    for name, factors in trained.items():
        for side in ("U", "V"):
            overlap = factors[f"osft_{side}_high"].t() @ factors[f"osft_{side}_low"]
            assert overlap.abs().max() < 1e-4, f"{name}: {side}_low drifted into the frozen subspace"

    ckpt_root = Path(cfg.checkpoint.checkpoint_dir)
    ckpt = sorted(ckpt_root.glob("epoch_*_step_*"), key=lambda p: int(p.name.rsplit("_", 1)[-1]))[-1]

    if rank == 0:
        # Resumable checkpoint: factors and dense weights.
        saved = _read_safetensors_dir(ckpt / "model")
        for name in modules:
            assert f"{name}.osft_U_low" in saved and f"{name}.osft_V_high" in saved, name
        # HF export: the resumable layout minus the OSFT factors, merged weights, loadable by transformers.
        export = _read_safetensors_dir(ckpt / "model" / "consolidated")
        assert not any(".osft_" in key for key in export)
        expected_keys = {key for key in saved if ".osft_" not in key}
        assert set(export) == expected_keys, set(export) ^ expected_keys
        for name, weight in merged.items():
            torch.testing.assert_close(export[f"{name}.weight"].float(), weight, rtol=1e-2, atol=1e-2)
        hf_model = AutoModelForCausalLM.from_pretrained(ckpt / "model" / "consolidated", dtype=torch.float32)
        assert all(not isinstance(m, OSFTLinear) for m in hf_model.modules())
    dist.barrier()

    # Exact resume: a fresh trainer restores the factors from the checkpoint.
    resumed = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    resumed.setup()
    for name, m in _osft_modules(resumed.model_parts[0]).items():
        for p, value in m.named_parameters():
            torch.testing.assert_close(_full(value), trained[name][p], rtol=0, atol=0, msg=f"{name}.{p}")

    # Optimizer state (AdamW moments and step count of every trainable factor) round-trips bitwise.
    def _optim_state(trainer):
        state = {}
        for opt in trainer.optimizer:
            names = {id(p): n for n, p in trainer.model_parts[0].named_parameters()}
            for group in opt.param_groups:
                for p in group["params"]:
                    for key, value in opt.state.get(p, {}).items():
                        state[f"{names[id(p)]}.{key}"] = _full(value) if torch.is_tensor(value) else value
        return state

    ref_state, resumed_state = _optim_state(trainer), _optim_state(resumed)
    assert ref_state and ref_state.keys() == resumed_state.keys()
    assert any(".osft_U_low." in key for key in ref_state), "no optimizer state for the OSFT factors"
    for key, value in ref_state.items():
        torch.testing.assert_close(resumed_state[key], value, rtol=0, atol=0, msg=key)

    # The next update after resuming equals the uninterrupted run's next update, bitwise.
    data = iter(resumed.dataloader)
    grad_acc = cfg.step_scheduler.global_batch_size // (cfg.step_scheduler.local_batch_size * dist.get_world_size())
    batches = [next(data) for _ in range(grad_acc)]
    for t in (trainer, resumed):
        t.model_parts[0].train()
        t.timestamp = time.perf_counter()  # set by run_train_validation_loop; used for throughput logging
        t._run_train_optim_step(copy.deepcopy(batches), t.max_grad_norm)
    for name, m in _osft_modules(resumed.model_parts[0]).items():
        ref_module = _osft_modules(trainer.model_parts[0])[name]
        for p, value in m.named_parameters():
            torch.testing.assert_close(
                _full(value), _full(dict(ref_module.named_parameters())[p]), rtol=0, atol=0, msg=f"next step {name}.{p}"
            )
    moved = max(
        (_full(value) - trained[name][p]).abs().max().item()
        for name, m in _osft_modules(resumed.model_parts[0]).items()
        for p, value in m.named_parameters()
        if value.requires_grad
    )
    assert moved > 0, "the extra step did not update the factors; the bitwise comparison would be vacuous"
