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

"""HuggingFace PEFT <-> nemotron3 LoRA conversion.

PEFT key convention (what HF/PEFT writes):
    base_model.model.backbone.layers.{i}.mixer.{proj}.lora_{A,B}.weight
with shapes:
    lora_A: (r, in_features)
    lora_B: (out_features, r)

Internal `LoRAPair` storage is the *transpose* of that:
    a: (in_features, r)
    b: (r, out_features)

so import is `a = hf_A.T`, `b = hf_B.T`, and the layer index taken from the
key picks which `LayerLoRA` slot to populate. Mamba `in_proj` is a single fused
2D matrix in both representations, in `[wg, wx, wb, wc, wdt]` order, so no
splitting is required at load time.

Scaling: PEFT applies the LoRA delta as `(alpha / r) * (B @ A) @ x` at forward
time. To avoid having every caller of `forward`/`prefill`/`decode_step`
remember to pass `lora_scaling=lora_cfg.scaling` (and silently produce a 2x
over-strong delta if they forget), this loader **pre-multiplies the `b` matrix
by `alpha / r`**. The forward path can then run with the default
`lora_scaling=1.0` and still match HF semantics exactly. The original `alpha`
and `rank` are kept on the returned `LoRAConfig` so export code can reconstruct
the on-disk format.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import jax
import jax.numpy as jnp

import numpy as np

from nemotron3_jax.lora import (
    AttnLoRA,
    LayerLoRA,
    LoRAConfig,
    LoRAPair,
    LoRAWeights,
    MambaLoRA,
    MoEExpertLoRA,
    MoESharedLoRA,
)
from nemotron3_jax.model import Config


_PROJ_TO_FIELD = {
    "q_proj": ("attn", "q"),
    "k_proj": ("attn", "k"),
    "v_proj": ("attn", "v"),
    "o_proj": ("attn", "o"),
    "in_proj": ("mamba", "in_proj"),
    "out_proj": ("mamba", "out_proj"),
    "shared_experts.up_proj": ("moe_shared", "ws_up"),
    "shared_experts.down_proj": ("moe_shared", "ws_down"),
}

_KEY_RE = re.compile(
    r"^base_model\.model\.backbone\.layers\.(\d+)\.mixer\."
    r"(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|shared_experts\.up_proj|shared_experts\.down_proj)"
    r"\.lora_(A|B)\.weight$"
)

# Per-routed-expert keys — captured separately from `_KEY_RE` because they get
# stacked across the expert axis into a single LoRA tensor pair per (layer, proj),
# whereas the keys above each correspond to one (layer, proj) directly.
_EXPERT_PROJ_TO_FIELD = {
    "up_proj": "we_up",
    "down_proj": "we_down",
}

_EXPERT_KEY_RE = re.compile(
    r"^base_model\.model\.backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\."
    r"(up_proj|down_proj)\.lora_(A|B)\.weight$"
)


def _read_safetensors(path: str) -> dict[str, Any]:
    from safetensors import safe_open  # local import: optional dep
    out = {}
    with safe_open(path, framework="numpy") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def _empty_layer() -> LayerLoRA:
    return LayerLoRA(attn=None, mamba=None, moe_shared=None, moe_experts=None)


def _ensure_group(layer: LayerLoRA, group: str) -> LayerLoRA:
    if group == "attn" and layer.attn is None:
        layer.attn = AttnLoRA(q=None, k=None, v=None, o=None)
    elif group == "mamba" and layer.mamba is None:
        layer.mamba = MambaLoRA(in_proj=None, out_proj=None)
    elif group == "moe_shared" and layer.moe_shared is None:
        layer.moe_shared = MoESharedLoRA(ws_up=None, ws_down=None)
    elif group == "moe_experts" and layer.moe_experts is None:
        layer.moe_experts = MoEExpertLoRA(we_up=None, we_down=None)
    return layer


def load_hf_adapter(
    adapter_dir: str,
    cfg: Config,
    dtype=jnp.bfloat16,
) -> tuple[LoRAWeights, LoRAConfig]:
    """Load a PEFT-format LoRA directory into a `LoRAWeights` pytree.

    Returns the weights and a `LoRAConfig` reconstructed from the adapter
    metadata + observed key set (so `target_*_layers` reflects the actual
    populated layers, not whatever the user passed at training time).

    The host arrays are returned uncommitted to a mesh; call
    `jax.device_put` with the result of `LoRAWeights.shardings(cfg, lora_cfg)`
    if you need them on a specific mesh.
    """
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    st_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    with open(cfg_path) as f:
        adapter_cfg = json.load(f)
    rank = int(adapter_cfg["r"])
    alpha = float(adapter_cfg["lora_alpha"])
    scaling = alpha / rank

    tensors = _read_safetensors(st_path)

    layers: list[LayerLoRA] = [_empty_layer() for _ in range(cfg.num_layers)]
    # (layer, proj) -> {"A": ndarray, "B": ndarray}  for non-expert keys
    pairs_AB: dict[tuple[int, str], dict[str, jax.Array]] = {}
    # (layer, proj) -> {expert_id: {"A": ndarray, "B": ndarray}}  for routed-expert keys
    expert_pairs: dict[tuple[int, str], dict[int, dict[str, np.ndarray]]] = {}

    unmatched: list[str] = []
    for key, arr in tensors.items():
        m = _KEY_RE.match(key)
        if m is not None:
            idx = int(m.group(1))
            proj = m.group(2)
            ab = m.group(3)
            if idx >= cfg.num_layers:
                raise ValueError(f"adapter key references layer {idx} but cfg.num_layers={cfg.num_layers}")
            slot = pairs_AB.setdefault((idx, proj), {})
            slot[ab] = jnp.asarray(arr, dtype=dtype)
            continue
        em = _EXPERT_KEY_RE.match(key)
        if em is not None:
            idx = int(em.group(1))
            expert_id = int(em.group(2))
            proj = em.group(3)
            ab = em.group(4)
            if idx >= cfg.num_layers:
                raise ValueError(f"adapter key references layer {idx} but cfg.num_layers={cfg.num_layers}")
            if expert_id >= cfg.moe_num_experts:
                raise ValueError(
                    f"adapter key references expert {expert_id} but cfg.moe_num_experts={cfg.moe_num_experts}"
                )
            per_layer = expert_pairs.setdefault((idx, proj), {})
            per_expert = per_layer.setdefault(expert_id, {})
            per_expert[ab] = np.asarray(arr)
            continue
        unmatched.append(key)

    if unmatched:
        raise ValueError(f"unrecognized adapter keys: {unmatched[:5]} ...")

    attn_layers: set[int] = set()
    mamba_layers: set[int] = set()
    moe_shared_layers: set[int] = set()
    moe_expert_layers: set[int] = set()

    for (idx, proj), ab in pairs_AB.items():
        if "A" not in ab or "B" not in ab:
            raise ValueError(f"layer {idx} {proj}: missing A or B (got {sorted(ab)})")
        hf_A, hf_B = ab["A"], ab["B"]  # (r, in), (out, r)
        if hf_A.shape[0] != rank or hf_B.shape[1] != rank:
            raise ValueError(
                f"layer {idx} {proj}: rank mismatch — adapter_config r={rank} but A={hf_A.shape}, B={hf_B.shape}"
            )
        # Bake the PEFT scaling factor into `b` so the forward path can use the
        # default `lora_scaling=1.0` without any caller-side bookkeeping.
        pair = LoRAPair(a=hf_A.T, b=(hf_B * scaling).T)
        group, field = _PROJ_TO_FIELD[proj]
        layer = _ensure_group(layers[idx], group)
        setattr(getattr(layer, group), field, pair)
        if group == "attn":
            attn_layers.add(idx)
        elif group == "mamba":
            mamba_layers.add(idx)
        else:
            moe_shared_layers.add(idx)

    # Stack per-expert keys into (num_experts, in, r) / (num_experts, r, out) tensors.
    # PEFT writes each expert's LoRA pair separately; the JAX MoE kernel consumes the
    # whole stack via `ragged_dot`. Missing experts are filled with zeros so the stack
    # is well-defined even if the source adapter only targeted a subset of experts.
    E = cfg.moe_num_experts
    for (idx, proj), per_layer in expert_pairs.items():
        # Determine in/out from any populated expert
        first_eid = next(iter(per_layer))
        any_ab = per_layer[first_eid]
        if "A" not in any_ab or "B" not in any_ab:
            raise ValueError(
                f"layer {idx} expert {first_eid} {proj}: missing A or B (got {sorted(any_ab)})"
            )
        in_dim = int(any_ab["A"].shape[1])    # PEFT A is (r, in)
        out_dim = int(any_ab["B"].shape[0])   # PEFT B is (out, r)
        if any_ab["A"].shape[0] != rank or any_ab["B"].shape[1] != rank:
            raise ValueError(
                f"layer {idx} expert {first_eid} {proj}: rank mismatch — "
                f"adapter_config r={rank} but A={any_ab['A'].shape}, B={any_ab['B'].shape}"
            )
        # Internal layout: a (E, in, r), b (E, r, out)
        a_stack = np.zeros((E, in_dim, rank), dtype=np.float32)
        b_stack = np.zeros((E, rank, out_dim), dtype=np.float32)
        for eid, ab in per_layer.items():
            if "A" not in ab or "B" not in ab:
                raise ValueError(f"layer {idx} expert {eid} {proj}: missing A or B")
            a_stack[eid] = np.asarray(ab["A"]).T               # (in, r)
            b_stack[eid] = (np.asarray(ab["B"]) * scaling).T   # (r, out), scaling baked in
        pair = LoRAPair(a=jnp.asarray(a_stack, dtype=dtype), b=jnp.asarray(b_stack, dtype=dtype))
        layer = _ensure_group(layers[idx], "moe_experts")
        setattr(layer.moe_experts, _EXPERT_PROJ_TO_FIELD[proj], pair)
        moe_expert_layers.add(idx)

    lora_cfg = LoRAConfig(
        rank=rank,
        alpha=alpha,
        target_attn_layers=tuple(sorted(attn_layers)),
        target_mamba_layers=tuple(sorted(mamba_layers)),
        target_moe_shared=bool(moe_shared_layers),
        target_moe_experts=bool(moe_expert_layers),
        dtype=dtype,
    )
    return LoRAWeights(layers=layers), lora_cfg
