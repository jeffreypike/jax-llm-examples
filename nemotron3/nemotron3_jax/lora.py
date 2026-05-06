# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LoRA adapters for nemotron3.

This module defines a parallel pytree (`LoRAWeights`) that lives alongside the
base `Weights` and is passed as an optional extra argument through
`forward`. The base weights are never mutated and the existing inference path
is unchanged when no LoRA tree is provided.

LoRA matrices are stored in HF-native 2D layouts (`a: (in, r)`, `b: (r, out)`)
so HF PEFT export becomes a trivial transpose + rename. At forward time we
compute `delta = (x @ a) @ b * (alpha / r)` and reshape it to match the layout
of the base projection's output before adding it.

Targets supported (matching the user's HF PEFT spec):
- Attention `q_proj`, `k_proj`, `v_proj`, `o_proj` for selected layers.
- Mamba `in_proj` (fused across the [wg, wx, wb, wc, wdt] split) and
  `out_proj` for selected layers.
- MoE `shared_experts.up_proj`, `shared_experts.down_proj` for all MoE layers.
- MoE per-routed-expert `experts.{i}.up_proj`, `experts.{i}.down_proj` for all
  MoE layers, stored stacked across the expert axis as a single tensor pair so
  the forward path can use `ragged_dot` against it inside the MoE shard_map.
"""

from __future__ import annotations

import dataclasses
import math
from functools import partial, lru_cache
from typing import Optional

import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import PartitionSpec as P

from nemotron3_jax.model import (
    ArrayInfo,
    Config,
    _Init,
    is_param,
    is_type,
    logical_to_physical,
    logical_to_sharding,
)


# ----------------------------------------------------------------------------
# LoRA configuration
# ----------------------------------------------------------------------------


# Default targets matching the user's HF PEFT spec.
DEFAULT_ATTN_LAYERS: tuple[int, ...] = (5, 12, 19, 26, 33, 42)
DEFAULT_MAMBA_LAYERS: tuple[int, ...] = (4, 11, 18, 25, 32, 41)


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class LoRAConfig:
    """Static LoRA configuration. Hashable so it can be used as a jit static arg."""

    rank: int = 16
    alpha: float = 32.0
    target_attn_layers: tuple[int, ...] = DEFAULT_ATTN_LAYERS
    target_mamba_layers: tuple[int, ...] = DEFAULT_MAMBA_LAYERS
    target_moe_shared: bool = True
    target_moe_experts: bool = False
    dtype: jax.typing.DTypeLike = jnp.bfloat16

    @property
    def scaling(self) -> float:
        return float(self.alpha) / float(self.rank)

    def adapt_attn(self, layer_idx: int, cfg: Config) -> bool:
        return cfg.layer_pattern[layer_idx] == "*" and layer_idx in set(self.target_attn_layers)

    def adapt_mamba(self, layer_idx: int, cfg: Config) -> bool:
        return cfg.layer_pattern[layer_idx] == "M" and layer_idx in set(self.target_mamba_layers)

    def adapt_moe_shared(self, layer_idx: int, cfg: Config) -> bool:
        return cfg.layer_pattern[layer_idx] == "E" and self.target_moe_shared

    def adapt_moe_experts(self, layer_idx: int, cfg: Config) -> bool:
        return cfg.layer_pattern[layer_idx] == "E" and self.target_moe_experts


# ----------------------------------------------------------------------------
# LoRA pytree dataclasses
# ----------------------------------------------------------------------------


def _lora_a_init(in_dim: int):
    """He-normal-ish init for LoRA A; matches PEFT's default kaiming_uniform_."""
    bound = math.sqrt(1.0 / max(in_dim, 1))
    def _init(key, shape, dtype):
        return random.uniform(key, shape, dtype, -bound, bound)
    return _init


def _zeros_init(key, shape, dtype):
    return jnp.zeros(shape, dtype=dtype)


def _make_lora_pair_info(in_dim: int, out_dim: int, rank: int, dtype) -> "LoRAPair":
    """Build a LoRAPair of ArrayInfos with replicated logical axes."""
    return LoRAPair(
        a=ArrayInfo((in_dim, rank), dtype, (None, None), _lora_a_init(in_dim)),
        b=ArrayInfo((rank, out_dim), dtype, (None, None), _zeros_init),
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class LoRAPair:
    a: jax.Array | ArrayInfo  # (in_dim, r)
    b: jax.Array | ArrayInfo  # (r, out_dim)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class AttnLoRA:
    q: LoRAPair
    k: LoRAPair
    v: LoRAPair
    o: LoRAPair

    @classmethod
    def abstract(cls, cfg: Config, lora_cfg: LoRAConfig) -> "AttnLoRA":
        embed = cfg.embed
        q_out = cfg.q_heads * cfg.head_dim
        kv_out = cfg.kv_heads * cfg.head_dim
        r = lora_cfg.rank
        d = lora_cfg.dtype
        return AttnLoRA(
            q=_make_lora_pair_info(embed, q_out, r, d),
            k=_make_lora_pair_info(embed, kv_out, r, d),
            v=_make_lora_pair_info(embed, kv_out, r, d),
            o=_make_lora_pair_info(q_out, embed, r, d),
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class MambaLoRA:
    in_proj: LoRAPair
    out_proj: LoRAPair

    @classmethod
    def abstract(cls, cfg: Config, lora_cfg: LoRAConfig) -> "MambaLoRA":
        x_size = cfg.mamba_num_heads * cfg.mamba_head_dim
        bc_size = cfg.mamba_n_groups * cfg.mamba_ssm_state_size
        dt_size = cfg.mamba_num_heads
        in_proj_out = 2 * x_size + 2 * bc_size + dt_size  # [wg, wx, wb, wc, wdt]
        out_proj_in = cfg.mamba_num_heads * cfg.mamba_head_dim
        r = lora_cfg.rank
        d = lora_cfg.dtype
        return MambaLoRA(
            in_proj=_make_lora_pair_info(cfg.embed, in_proj_out, r, d),
            out_proj=_make_lora_pair_info(out_proj_in, cfg.embed, r, d),
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class MoESharedLoRA:
    ws_up: LoRAPair
    ws_down: LoRAPair

    @classmethod
    def abstract(cls, cfg: Config, lora_cfg: LoRAConfig) -> "MoESharedLoRA":
        r = lora_cfg.rank
        d = lora_cfg.dtype
        return MoESharedLoRA(
            ws_up=_make_lora_pair_info(cfg.embed, cfg.moe_shared_ffw_size, r, d),
            ws_down=_make_lora_pair_info(cfg.moe_shared_ffw_size, cfg.embed, r, d),
        )


def _make_stacked_lora_pair_info(
    num_experts: int, in_dim: int, out_dim: int, rank: int, dtype,
    a_logical: tuple, b_logical: tuple,
) -> "LoRAPair":
    """Build a LoRAPair of stacked-across-experts ArrayInfos.

    Shapes: a is (num_experts, in_dim, rank); b is (num_experts, rank, out_dim).
    Logical axes for a/b are caller-supplied so the per-expert sharding can match
    whichever base weight (we_up vs we_down) this LoRA pair targets.
    """
    return LoRAPair(
        a=ArrayInfo((num_experts, in_dim, rank), dtype, a_logical, _lora_a_init(in_dim)),
        b=ArrayInfo((num_experts, rank, out_dim), dtype, b_logical, _zeros_init),
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class MoEExpertLoRA:
    """Per-routed-expert LoRA, stacked across the expert axis.

    `we_up.a`:   (E, embed,    rank)        sharded ("moe_e_experts", "moe_e_up_embed", None)
    `we_up.b`:   (E, rank,     ffw_size)    sharded ("moe_e_experts", None, "moe_e_up_ffw")
    `we_down.a`: (E, ffw_size, rank)        sharded ("moe_e_experts", "moe_e_down_ffw", None)
    `we_down.b`: (E, rank,     embed)       sharded ("moe_e_experts", None, "moe_e_down_embed")

    The sharding for each tensor matches the corresponding axis of the base
    `we_up`/`we_down` so that, inside the `_expert_fn` shard_map, every
    `ragged_dot` involving the LoRA pair has consistent layouts with the base
    GMM and no extra all-gathers are required. The `rank` axis is replicated
    everywhere (it's small, and replicating avoids communication on it).
    """
    we_up: LoRAPair
    we_down: LoRAPair

    @classmethod
    def abstract(cls, cfg: Config, lora_cfg: LoRAConfig) -> "MoEExpertLoRA":
        r = lora_cfg.rank
        d = lora_cfg.dtype
        E = cfg.moe_num_experts
        return MoEExpertLoRA(
            we_up=_make_stacked_lora_pair_info(
                E, cfg.embed, cfg.moe_ffw_size, r, d,
                a_logical=("moe_e_experts", "moe_e_up_embed", None),
                b_logical=("moe_e_experts", None, "moe_e_up_ffw"),
            ),
            we_down=_make_stacked_lora_pair_info(
                E, cfg.moe_ffw_size, cfg.embed, r, d,
                a_logical=("moe_e_experts", "moe_e_down_ffw", None),
                b_logical=("moe_e_experts", None, "moe_e_down_embed"),
            ),
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class LayerLoRA:
    attn: AttnLoRA | None
    mamba: MambaLoRA | None
    moe_shared: MoESharedLoRA | None
    moe_experts: MoEExpertLoRA | None = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class LoRAWeights(_Init):
    layers: list  # list[LayerLoRA] — one entry per base layer

    @classmethod
    def abstract(cls, cfg: Config, lora_cfg: LoRAConfig) -> "LoRAWeights":
        layers: list[LayerLoRA] = []
        for i in range(cfg.num_layers):
            attn = AttnLoRA.abstract(cfg, lora_cfg) if lora_cfg.adapt_attn(i, cfg) else None
            mamba = MambaLoRA.abstract(cfg, lora_cfg) if lora_cfg.adapt_mamba(i, cfg) else None
            moe_s = MoESharedLoRA.abstract(cfg, lora_cfg) if lora_cfg.adapt_moe_shared(i, cfg) else None
            moe_e = MoEExpertLoRA.abstract(cfg, lora_cfg) if lora_cfg.adapt_moe_experts(i, cfg) else None
            layers.append(LayerLoRA(attn=attn, mamba=mamba, moe_shared=moe_s, moe_experts=moe_e))
        return LoRAWeights(layers=layers)

    @classmethod
    def shardings(cls, cfg: Config, lora_cfg: LoRAConfig):
        abstract = cls.abstract(cfg, lora_cfg)
        return jax.tree.map(
            lambda info: logical_to_sharding(info.logical_axes, cfg.mesh, cfg.rules),
            abstract,
            is_leaf=is_param,
        )

    @classmethod
    def init(cls, key, cfg: Config, lora_cfg: LoRAConfig) -> "LoRAWeights":
        abstract = cls.abstract(cfg, lora_cfg)
        shardings = cls.shardings(cfg, lora_cfg)
        leaves, treedef = jax.tree.flatten(abstract, is_leaf=is_param)
        sharding_leaves = jax.tree.leaves(shardings, is_leaf=is_param)
        keys = list(random.split(key, len(leaves)))

        @partial(jax.jit, out_shardings=tuple(sharding_leaves))
        def _init_all(key_arr):
            ks = list(random.split(key_arr, len(leaves)))
            out = []
            for k, info in zip(ks, leaves):
                out.append(info.initializer(k, info.shape, info.dtype))
            return tuple(out)

        init_leaves = _init_all(key)
        return jax.tree.unflatten(treedef, list(init_leaves))


# ----------------------------------------------------------------------------
# Forward-time application helpers
# ----------------------------------------------------------------------------


def apply_lora_dense(
    x: jax.Array,
    pair: LoRAPair | None,
    scaling: float,
    out_dtype,
) -> jax.Array | None:
    """Compute LoRA delta `(x @ a) @ b * scaling` for a 2D LoRA pair.

    Returns None if pair is None. Caller is responsible for reshaping the
    delta to whatever layout the base output uses.
    """
    if pair is None:
        return None
    a = pair.a.astype(x.dtype)
    b = pair.b.astype(x.dtype)
    z = jnp.einsum("...i,ir->...r", x, a)
    delta = jnp.einsum("...r,ro->...o", z, b)
    return (delta * scaling).astype(out_dtype)
