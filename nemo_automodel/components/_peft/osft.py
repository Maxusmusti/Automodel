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

"""Orthogonal Subspace Fine-Tuning (OSFT).

OSFT (Nayak et al., "Sculpting Subspaces: Constrained Full Fine-Tuning in LLMs for Continual Learning",
ICLR 2026) decomposes each targeted weight ``W = U S V^T`` once, freezes the top-``k`` singular
directions, and trains only the remaining low-rank factors. Updates are kept orthogonal to the frozen
subspaces by projecting the low-rank factor gradients before every optimizer step and the factor values
after it::

    W_eff   = U_high S_high V_high^T + U_low S_low V_low^T
    dU_low <- dU_low - U_high (U_high^T dU_low)        U_low <- U_low - U_high (U_high^T U_low)
    dV_low <- dV_low - V_high (V_high^T dV_low)        V_low <- V_low - V_high (V_high^T V_low)

``V`` factors are stored as ``(in_features, rank)`` matrices (the paper's convention). FSDP2 shards every
factor along dim 0, so ``U`` factors are split along ``out_features`` and ``V`` factors along
``in_features``. Both projections then reduce to one small ``(k, r)`` coefficient all-reduce per factor,
with no gather of the frozen bases.

Algorithm and defaults follow the reference implementation in Red Hat's mini_trainer
(``mini_trainer/osft_utils.py``, Apache-2.0), including the post-step parameter projection.
"""

from __future__ import annotations

import contextlib
import logging
import math
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Literal

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, distribute_tensor

from nemo_automodel.components._peft.module_matcher import ModuleMatcher

logger = logging.getLogger(__name__)

OSFT_PARAM_PREFIX = "osft_"
"""Prefix of every OSFT factor parameter name. Base-checkpoint loading drops keys containing it."""

DEFAULT_OSFT_TARGETS = ["*.q_proj", "*.k_proj", "*.v_proj", "*.o_proj", "*.gate_proj", "*.up_proj", "*.down_proj"]

# The factored path's GEMMs have the frozen/trainable ranks as an inner dimension; below this rank
# they run far under peak, so the cost model discounts them (measured on H100, see _choose_path).
_SKINNY_RANK = 512
_SKINNY_EFFICIENCY = 0.6


@dataclass
class OSFTConfig:
    """OSFT settings.

    Attributes:
        unfreeze_rank_ratio: Fraction of each target's singular directions that stays trainable
            (the smallest singular values). ``rank_high = floor(min(N, M) * (1 - ratio))``, capped
            at ``min(N, M) - 1``.
        target_modules: Wildcard patterns of ``nn.Linear`` modules to decompose.
        exclude_modules: Wildcard patterns to exclude (mutually exclusive with ``target_modules``).
        forward_mode: ``"auto"`` picks per layer and per call between ``"reconstruct"`` (materialize
            ``W_eff`` and run one dense GEMM) and ``"factored"`` (four thin GEMMs), by estimated cost.
        project_gradients_before_clipping: Project gradients before gradient-norm clipping, so the
            clipped norm is the norm of the update actually applied. ``False`` reproduces mini_trainer,
            which clips the unprojected gradient and projects inside the optimizer step.
    """

    unfreeze_rank_ratio: float = 0.25
    target_modules: list[str] | str | None = field(default_factory=lambda: list(DEFAULT_OSFT_TARGETS))
    exclude_modules: list[str] | str | None = field(default_factory=list)
    forward_mode: Literal["auto", "reconstruct", "factored"] = "auto"
    project_gradients_before_clipping: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.unfreeze_rank_ratio < 1.0:
            raise ValueError(f"unfreeze_rank_ratio must be in (0, 1), got {self.unfreeze_rank_ratio}")
        if self.forward_mode not in ("auto", "reconstruct", "factored"):
            raise ValueError(f"forward_mode must be auto, reconstruct or factored, got {self.forward_mode!r}")

    def rank_high(self, out_features: int, in_features: int) -> int:
        """Number of frozen singular directions for a weight of the given shape (mini_trainer rule)."""
        full_rank = min(out_features, in_features)
        rank = int(math.floor(full_rank * (1.0 - self.unfreeze_rank_ratio)))
        return min(rank, full_rank - 1)


class _OSFTReconstructLinear(torch.autograd.Function):
    """``y = x W_eff^T`` with ``W_eff`` rebuilt from the factors; gradients only for the low factors.

    The backward pass recomputes ``W_eff`` instead of saving it, forms ``dW`` once, and projects it onto
    ``U_low``, ``S_low`` and ``V_low``. Saved activations are just ``x``, as for a plain linear layer.
    """

    @staticmethod
    def forward(ctx, x, u_high, s_high, v_high, u_low, s_low, v_low):  # noqa: D102
        weight = _merge(u_high, s_high, v_high, u_low, s_low, v_low)
        ctx.save_for_backward(x, u_high, s_high, v_high, u_low, s_low, v_low)
        return F.linear(x, weight)

    @staticmethod
    def backward(ctx, grad_out):  # noqa: D102
        x, u_high, s_high, v_high, u_low, s_low, v_low = ctx.saved_tensors
        weight = _merge(u_high, s_high, v_high, u_low, s_low, v_low)
        grad_x = grad_out @ weight
        grad_w = grad_out.reshape(-1, grad_out.shape[-1]).t() @ x.reshape(-1, x.shape[-1])  # (N, M)
        a = grad_w @ v_low  # (N, r)
        grad_u_low = a * s_low
        grad_s_low = (u_low * a).sum(0)
        grad_v_low = (grad_w.t() @ u_low) * s_low  # (M, r)
        return grad_x, None, None, None, grad_u_low, grad_s_low, grad_v_low


def _merge(u_high, s_high, v_high, u_low, s_low, v_low) -> torch.Tensor:
    """Return ``U_high S_high V_high^T + U_low S_low V_low^T`` as one ``(N, M)`` matrix."""
    return torch.addmm(torch.mm(u_low * s_low, v_low.t()), u_high * s_high, v_high.t())


def _choose_path(tokens: int, n: int, m: int, rank_high: int, rank_low: int) -> str:
    """Pick the cheaper forward/backward path from a FLOP model with a skinny-GEMM discount.

    Validated against H100 microbenchmarks of Qwen3-8B shapes at 4k/16k/32k tokens and
    unfreeze ratios 0.25/0.5: picks the measured-faster path in every case but near-ties.
    """
    full = rank_high + rank_low
    factored = tokens * (n + m) * (4 * full + 2 * rank_low)
    if rank_low < _SKINNY_RANK:
        factored /= _SKINNY_EFFICIENCY
    reconstruct = 6 * tokens * n * m + 4 * n * m * full + 4 * n * m * rank_low
    return "reconstruct" if reconstruct < factored else "factored"


class OSFTLinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` holding OSFT factors instead of a dense weight.

    Parameters (``N = out_features``, ``M = in_features``, ``k`` frozen rank, ``r`` trainable rank):
        ``osft_U_high (N, k)``, ``osft_S_high (k,)``, ``osft_V_high (M, k)``: frozen.
        ``osft_U_low (N, r)``, ``osft_S_low (r,)``, ``osft_V_low (M, r)``: trainable.

    The module's state dict also carries a dense ``weight`` entry so checkpoints keep the plain
    ``nn.Linear`` layout: loading a base checkpoint delivers ``weight`` (stashed until
    :func:`initialize_osft` decomposes it), and saving emits the merged ``W_eff``. Factor entries are
    authoritative whenever they are present (exact resume).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank_high: int,
        *,
        bias: bool,
        forward_mode: str = "auto",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Create an uninitialized OSFT layer; call :func:`initialize_osft` after loading weights."""
        super().__init__()
        full_rank = min(in_features, out_features)
        if not 0 < rank_high < full_rank:
            raise ValueError(f"rank_high must be in (0, {full_rank}), got {rank_high}")
        rank_low = full_rank - rank_high
        kw = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.rank_high = rank_high
        self.rank_low = rank_low
        self.forward_mode = forward_mode
        self.osft_U_high = nn.Parameter(torch.empty(out_features, rank_high, **kw), requires_grad=False)
        self.osft_S_high = nn.Parameter(torch.empty(rank_high, **kw), requires_grad=False)
        self.osft_V_high = nn.Parameter(torch.empty(in_features, rank_high, **kw), requires_grad=False)
        self.osft_U_low = nn.Parameter(torch.empty(out_features, rank_low, **kw))
        self.osft_S_low = nn.Parameter(torch.empty(rank_low, **kw))
        self.osft_V_low = nn.Parameter(torch.empty(in_features, rank_low, **kw))
        self.bias = nn.Parameter(torch.empty(out_features, **kw)) if bias else None
        self._base_weight: torch.Tensor | None = None
        self._osft_initialized = False
        self._register_state_dict_hook(OSFTLinear._add_dense_weight_to_state_dict)

    @property
    def weight(self) -> torch.Tensor | None:
        """Base weight stashed during checkpoint loading (``None`` once decomposed).

        Exists so state-dict FQN resolution of the dense ``weight`` key succeeds; the trained weight is
        :meth:`merged_weight`.
        """
        return self._base_weight

    def extra_repr(self) -> str:  # noqa: D102
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, rank_high={self.rank_high}, "
            f"rank_low={self.rank_low}, bias={self.bias is not None}, forward_mode={self.forward_mode}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        factors = (
            self.osft_U_high,
            self.osft_S_high,
            self.osft_V_high,
            self.osft_U_low,
            self.osft_S_low,
            self.osft_V_low,
        )
        path = self.forward_mode
        if path == "auto":
            tokens = x.numel() // x.shape[-1]
            path = _choose_path(tokens, self.out_features, self.in_features, self.rank_high, self.rank_low)
        if path == "reconstruct":
            y = _OSFTReconstructLinear.apply(x, *factors)
        else:
            u_high, s_high, v_high, u_low, s_low, v_low = factors
            y = ((x @ v_high) * s_high) @ u_high.t() + ((x @ v_low) * s_low) @ u_low.t()
        if self.bias is not None:
            y = y + self.bias
        return y

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """Return the dense ``W_eff`` (a row-sharded DTensor when the factors are FSDP-sharded)."""
        u_high, u_low = self.osft_U_high, self.osft_U_low
        if not isinstance(u_high, DTensor):
            return _merge(u_high, self.osft_S_high, self.osft_V_high, u_low, self.osft_S_low, self.osft_V_low)
        rows = _merge(
            u_high.to_local(),
            self.osft_S_high.full_tensor(),
            self.osft_V_high.full_tensor(),
            u_low.to_local(),
            self.osft_S_low.full_tensor(),
            self.osft_V_low.full_tensor(),
        )
        return DTensor.from_local(
            rows,
            device_mesh=u_high.device_mesh,
            placements=u_high.placements,
            shape=torch.Size((self.out_features, self.in_features)),
            stride=(self.in_features, 1),
            run_check=False,
        )

    def _dense_weight_template(self) -> torch.Tensor:
        """Empty tensor with the dense weight's global shape and the factors' sharding."""
        ref = self.osft_U_high
        shape = (self.out_features, self.in_features)
        if isinstance(ref, DTensor):
            return distribute_tensor(
                torch.empty(shape, dtype=ref.dtype, device=ref.to_local().device),
                ref.device_mesh,
                ref.placements,
                src_data_rank=None,
            )
        return torch.empty(shape, dtype=ref.dtype, device=ref.device)

    @staticmethod
    def _add_dense_weight_to_state_dict(module: "OSFTLinear", state_dict, prefix, local_metadata) -> None:
        key = prefix + "weight"
        ref = module.osft_U_high
        if ref.device.type == "meta":
            # Key discovery before materialization (e.g. the HF export layout): shape and dtype only.
            state_dict[key] = torch.empty(module.out_features, module.in_features, dtype=ref.dtype, device="meta")
        elif module._osft_initialized:
            state_dict[key] = module.merged_weight()
        else:
            state_dict[key] = module._dense_weight_template()

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ) -> None:
        dense = state_dict.pop(prefix + "weight", None)
        has_factors = any(k.startswith(prefix + OSFT_PARAM_PREFIX) for k in state_dict)
        if has_factors:
            self._osft_initialized = True
            self._base_weight = None
        elif dense is not None:
            self._base_weight = dense
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, False, missing_keys, unexpected_keys, error_msgs
        )
        if not has_factors:
            # Factors are produced by initialize_osft(); they are not expected in a base checkpoint.
            missing_keys[:] = [k for k in missing_keys if not k.startswith(prefix + OSFT_PARAM_PREFIX)]


def _iter_osft_modules(modules: Iterable[nn.Module]) -> Iterator[tuple[str, OSFTLinear]]:
    for root in modules:
        for name, module in root.named_modules():
            if isinstance(module, OSFTLinear):
                yield name, module


def _is_dense_linear(module: nn.Module) -> bool:
    """``nn.Linear``, or a TransformerEngine ``Linear`` (same ``weight``/``bias`` checkpoint keys).

    OSFT replaces the module's compute entirely, so the original kernel backend does not matter; only the
    dense 2-D weight layout does. TE is matched by name so this module does not import it.
    """
    if type(module) is nn.Linear:
        return True
    cls = type(module)
    is_te_linear = cls.__name__ == "Linear" and cls.__module__.startswith("transformer_engine.pytorch")
    weight = getattr(module, "weight", None)
    return is_te_linear and isinstance(weight, torch.Tensor) and weight.dim() == 2


def apply_osft(model: nn.Module, config: OSFTConfig) -> int:
    """Replace the targeted ``nn.Linear`` modules with uninitialized :class:`OSFTLinear` modules.

    Call on the meta-device model before sharding, like LoRA. Non-target parameters stay trainable.

    Returns:
        Number of replaced modules.
    """
    matcher = ModuleMatcher(config.target_modules, config.exclude_modules)
    replaced = 0
    for name, module in list(model.named_modules()):
        if not name or not matcher.match(module, name):
            continue
        if isinstance(module, OSFTLinear):
            continue
        if not _is_dense_linear(module):
            raise TypeError(
                f"OSFT target {name!r} is {type(module).__name__}; expected a dense linear layer "
                "(torch.nn.Linear or transformer_engine.pytorch.Linear)."
            )
        weight = module.weight
        bias = getattr(module, "bias", None)
        new = OSFTLinear(
            module.in_features,
            module.out_features,
            config.rank_high(module.out_features, module.in_features),
            bias=isinstance(bias, torch.Tensor) and bias.numel() > 0,
            forward_mode=config.forward_mode,
            device=weight.device,
            dtype=weight.dtype,
        )
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, new)
        replaced += 1
    if replaced == 0:
        raise ValueError(f"OSFT matched no modules for target_modules={config.target_modules}")
    logger.info("OSFT: replaced %d linear modules", replaced)
    return replaced


def _process_group(module: OSFTLinear) -> dist.ProcessGroup | None:
    ref = module.osft_U_high
    if isinstance(ref, DTensor):
        mesh = ref.device_mesh
        if mesh.ndim != 1:
            raise NotImplementedError("OSFT supports 1-D FSDP sharding of its factors")
        return mesh.get_group()
    return None


def _assign(param: nn.Parameter, full: torch.Tensor) -> None:
    """Copy a full tensor into a (possibly sharded) parameter."""
    full = full.to(param.dtype)
    if isinstance(param, DTensor):
        local = distribute_tensor(full, param.device_mesh, param.placements, src_data_rank=None).to_local()
        param.to_local().copy_(local)
    else:
        param.data.copy_(full)


@torch.no_grad()
def initialize_osft(model_parts: nn.Module | list[nn.Module]) -> int:
    """Decompose every stashed base weight into OSFT factors.

    The SVDs run in fp32 on GPU, distributed round-robin over the factors' process group: in each round
    every rank all-gathers the next ``world_size`` weights, decomposes the one it owns, and broadcasts
    the factors back over NCCL. Each rank then keeps its own shard.

    Returns:
        Number of initialized modules.
    """
    parts = model_parts if isinstance(model_parts, list) else [model_parts]
    modules = [m for _, m in _iter_osft_modules(parts) if not m._osft_initialized]
    if not modules:
        return 0
    missing = [m for m in modules if m._base_weight is None]
    if missing:
        raise RuntimeError(f"{len(missing)} OSFT modules have no base weight; was the base checkpoint loaded?")
    group = _process_group(modules[0])
    world = dist.get_world_size(group) if group is not None else 1
    rank = dist.get_rank(group) if group is not None else 0
    device = modules[0].osft_U_high.device if group is None else modules[0].osft_U_high.to_local().device
    max_rel_err = 0.0
    for start in range(0, len(modules), world):
        batch = modules[start : start + world]
        owned = None
        for owner, module in enumerate(batch):
            base = module._base_weight
            full = base.full_tensor() if isinstance(base, DTensor) else base
            if owner == rank:
                owned = full.to(device=device, dtype=torch.float32)
            del full
        factors = None
        if owned is not None:
            # cuSOLVER's default Jacobi driver (gesvdj) stops at ~1e-3 relative reconstruction error on
            # LLM weights; QR-based gesvd reaches ~1e-5 for ~1.3x the time. Accuracy here sets how exactly
            # W_eff equals W at step 0 and how orthonormal the frozen bases are.
            u, s, vh = torch.linalg.svd(owned, full_matrices=False, driver="gesvd" if owned.is_cuda else None)
            recon = (u * s) @ vh
            max_rel_err = max(max_rel_err, (torch.linalg.norm(recon - owned) / torch.linalg.norm(owned)).item())
            factors = (u.contiguous(), s.contiguous(), vh.t().contiguous())
            del recon, owned
        for owner, module in enumerate(batch):
            n, m = module.out_features, module.in_features
            full_rank = min(n, m)
            if owner == rank:
                u, s, v = factors
            else:
                u = torch.empty(n, full_rank, device=device, dtype=torch.float32)
                s = torch.empty(full_rank, device=device, dtype=torch.float32)
                v = torch.empty(m, full_rank, device=device, dtype=torch.float32)
            if group is not None:
                src = dist.get_global_rank(group, owner)
                for t in (u, s, v):
                    dist.broadcast(t, src=src, group=group)
            k = module.rank_high
            _assign(module.osft_U_high, u[:, :k])
            _assign(module.osft_S_high, s[:k])
            _assign(module.osft_V_high, v[:, :k])
            _assign(module.osft_U_low, u[:, k:])
            _assign(module.osft_S_low, s[k:])
            _assign(module.osft_V_low, v[:, k:])
            module._base_weight = None
            module._osft_initialized = True
    if group is not None:
        err = torch.tensor([max_rel_err], device=device)
        dist.all_reduce(err, op=dist.ReduceOp.MAX, group=group)
        max_rel_err = err.item()
    logger.info("OSFT: decomposed %d weights (max relative SVD reconstruction error %.2e)", len(modules), max_rel_err)
    return len(modules)


@contextlib.contextmanager
def _exact_fp32_matmul() -> Iterator[None]:
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


class OSFTProjector:
    """Keeps OSFT updates in the orthogonal complement of the frozen subspaces.

    Call :meth:`project_gradients` after backward (before or after gradient clipping, see
    :attr:`OSFTConfig.project_gradients_before_clipping`) and :meth:`project_parameters` after every
    optimizer step. Each call runs one fp32 coefficient all-reduce per bucket of factors.
    """

    def __init__(self, model_parts: nn.Module | list[nn.Module], bucket_bytes: int = 256 * 2**20) -> None:
        """Collect the OSFT modules of ``model_parts``."""
        parts = model_parts if isinstance(model_parts, list) else [model_parts]
        self.modules = [m for _, m in _iter_osft_modules(parts)]
        if not self.modules:
            raise ValueError("OSFTProjector found no OSFTLinear modules")
        self.group = _process_group(self.modules[0])
        self.distributed = self.group is not None and dist.get_world_size(self.group) > 1
        self.bucket_bytes = bucket_bytes

    @staticmethod
    def _local(t: torch.Tensor) -> torch.Tensor:
        return t.to_local() if isinstance(t, DTensor) else t

    def _pairs(self, use_grad: bool) -> list[tuple[torch.Tensor, torch.Tensor]]:
        pairs = []
        for m in self.modules:
            for basis, param in ((m.osft_U_high, m.osft_U_low), (m.osft_V_high, m.osft_V_low)):
                target = param.grad if use_grad else param
                if target is None:
                    continue
                pairs.append((self._local(basis), self._local(target)))
        return pairs

    @torch.no_grad()
    def _project(self, use_grad: bool) -> None:
        pairs = self._pairs(use_grad)
        with _exact_fp32_matmul():
            bucket: list[tuple[torch.Tensor, torch.Tensor]] = []
            size = 0
            for basis, target in pairs:
                bucket.append((basis, target))
                size += basis.shape[1] * target.shape[1] * 4
                if size >= self.bucket_bytes:
                    self._project_bucket(bucket)
                    bucket, size = [], 0
            if bucket:
                self._project_bucket(bucket)

    def _project_bucket(self, bucket: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        coeffs = [basis.t().float() @ target.float() for basis, target in bucket]
        if self.distributed:
            flat = torch.cat([c.reshape(-1) for c in coeffs])
            dist.all_reduce(flat, group=self.group)
            coeffs = list(torch.split(flat, [c.numel() for c in coeffs]))
        for (basis, target), coeff in zip(bucket, coeffs):
            coeff = coeff.view(basis.shape[1], target.shape[1])
            if target.dtype == torch.float32 and basis.dtype == torch.float32:
                target.addmm_(basis, coeff, alpha=-1.0)
            else:
                target.sub_((basis.float() @ coeff).to(target.dtype))

    def project_gradients(self) -> None:
        """Remove the frozen-subspace component of the low-rank factor gradients."""
        self._project(use_grad=True)

    def project_parameters(self) -> None:
        """Remove any frozen-subspace component the optimizer step introduced into the factors."""
        self._project(use_grad=False)
