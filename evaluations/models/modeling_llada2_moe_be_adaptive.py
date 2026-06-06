# Copyright 2025 Antgroup and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""PyTorch LLaDA2MoE model."""

import math
from typing import List, Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import (
    MoeModelOutputWithPast,
    MoeCausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS,
)
from transformers.utils import (
    TransformersKwargs,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_llada2_moe_be_adaptive import LLaDA2MoeConfig
from transformers.generation.utils import GenerationMixin

# @sicheng: time recorder
from collections import defaultdict
from contextlib import contextmanager

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LLaDA2MoeConfig"

# @sicheng: time recorder
class SimpleCudaProfiler:
    """
    Lightweight CUDA profiler for per-step breakdown.
    Stores times in milliseconds.
    """
    def __init__(self, enabled=True):
        self.enabled = enabled and torch.cuda.is_available()
        self.current = defaultdict(float)

    @contextmanager
    def time(self, name: str):
        if not self.enabled:
            yield
            return

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        torch.cuda.synchronize()
        self.current[name] += start.elapsed_time(end)

    def reset(self):
        self.current.clear()

    def snapshot(self):
        return dict(self.current)

def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class LLaDA2MoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LLaDA2MoeRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(LLaDA2MoeRMSNorm)


class LLaDA2MoeRotaryEmbedding(nn.Module):
    def __init__(self, config: LLaDA2MoeConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


class LLaDA2MoeMLP(nn.Module):
    def __init__(self, config: LLaDA2MoeConfig, intermediate_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LLaDA2MoeGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok # @sicheng: change this for top-k run
        self.num_experts = config.num_experts

        self.n_group = config.n_group
        self.topk_group = config.topk_group

        # topk selection algorithm
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.num_experts, self.gating_dim)))
        self.routed_scaling_factor = config.routed_scaling_factor

        self.register_buffer("expert_bias", torch.zeros(self.num_experts))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def group_limited_topk(
        self,
        scores: torch.Tensor,
    ):
        num_tokens, _ = scores.size()
        # Organize the experts into groups
        group_scores = (
            scores.view(num_tokens, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)

        # Mask the experts based on selection groups
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, self.n_group, self.num_experts // self.n_group)
            .reshape(num_tokens, -1)
        )

        masked_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
        probs, top_indices = torch.topk(masked_scores, k=self.top_k, dim=-1)

        return probs, top_indices

    # @sicheng
    def block_experts_select(self, block_scores: torch.Tensor):
        bsz, block_len, _ = block_scores.shape

        # 1) each token first keeps its local top-k experts
        local_scores, local_idx = torch.topk(block_scores, k=self.top_k, dim=-1)

        masked_scores = torch.zeros_like(block_scores)
        masked_scores.scatter_(-1, local_idx, local_scores)

        # 2) aggregate to block-level expert scores
        votes = masked_scores.sum(dim=1)  # [bsz, num_experts]

        # 3) normalize block-level scores
        votes = votes / (votes.sum(dim=-1, keepdim=True) + 1e-20)

        # print("[sicheng] block-level expert scores (after normalization):", votes)

        # 4) dynamic coreset selection by top-p cumulative probability
        sorted_votes, sorted_idx = torch.sort(votes, dim=-1, descending=True)   # [bsz, num_experts]
        cum_probs = torch.cumsum(sorted_votes, dim=-1)                          # [bsz, num_experts]

        top_p = self.config.core_threshold

        # keep all experts whose cumulative prob is still <= top_p
        sorted_keep = cum_probs <= top_p

        # always keep the first expert
        sorted_keep[..., 0] = True

        # also keep the first expert that makes cumulative prob exceed top_p
        exceed = cum_probs > top_p
        exceed_prev = torch.cat(
            [torch.zeros_like(exceed[..., :1]), exceed[..., :-1]], dim=-1
        )
        first_exceed = exceed & (~exceed_prev)
        sorted_keep = sorted_keep | first_exceed

        # map mask back to original expert order
        coreset_mask = torch.zeros_like(votes, dtype=torch.bool)
        coreset_mask.scatter_(1, sorted_idx, sorted_keep)

        # 5) safeguard: make sure at least guard_size experts are kept
        num_selected = coreset_mask.sum(dim=-1)  # [bsz]
        guard_size = 16
        need_fix = num_selected < guard_size
        if need_fix.any():
            fallback_idx = torch.topk(votes, k=guard_size, dim=-1).indices
            fallback_mask = torch.zeros_like(coreset_mask, dtype=torch.bool)
            fallback_mask.scatter_(1, fallback_idx, True)
            coreset_mask = torch.where(need_fix.unsqueeze(-1), fallback_mask, coreset_mask)

        return coreset_mask


    # @sicheng: process sequence in 32-token blocks, matching training-time behavior.
    # During denoising steps seq_len=block_length=32 (n_blocks=1, no padding needed).
    # During prefill seq_len=prefix_end (may be large), handled correctly via multi-block.
    def forward(self, hidden_states):
        block_size = 32
        bsz, seq_len, _ = hidden_states.shape

        hidden_states_2d = hidden_states.view(-1, hidden_states.shape[-1])
        logits = F.linear(
            hidden_states_2d.type(torch.float32), self.weight.type(torch.float32)
        )  # [bsz*seq_len, num_experts]

        scores = torch.sigmoid(logits.float()).type_as(logits)
        scores = scores.view(bsz, seq_len, self.num_experts)  # [bsz, seq_len, E]
        scores_for_routing = scores + self.expert_bias.view(1, 1, -1)

        if self.config.mode != "aggregate":
            raise ValueError(f"Unsupported mode: {self.config.mode}")

        # Pad sequence length to a multiple of block_size, matching training.
        block_size = min(block_size, seq_len)
        pad_len = (block_size - seq_len % block_size) % block_size
        if pad_len > 0:
            scores_padded = torch.cat([
                scores,
                torch.zeros(bsz, pad_len, self.num_experts, dtype=scores.dtype, device=scores.device),
            ], dim=1)
            routing_padded = torch.cat([
                scores_for_routing,
                torch.full((bsz, pad_len, self.num_experts), float("-inf"),
                           dtype=scores_for_routing.dtype, device=scores_for_routing.device),
            ], dim=1)
        else:
            scores_padded = scores
            routing_padded = scores_for_routing

        padded_len = scores_padded.shape[1]
        n_blocks = padded_len // block_size

        # [bsz, n_blocks, block_size, E] — same layout as training gate
        routing_blocks = routing_padded.view(bsz, n_blocks, block_size, self.num_experts)

        # Per-block coreset selection (matches training).
        # block_experts_select expects [B, block_len, E]; merge bsz and n_blocks dims.
        coreset_mask = self.block_experts_select(
            routing_blocks.reshape(bsz * n_blocks, block_size, self.num_experts)
        ).view(bsz, n_blocks, self.num_experts)  # [bsz, n_blocks, E]

        token_coreset_mask = coreset_mask.unsqueeze(2).expand(
            -1, -1, block_size, -1
        ).reshape(bsz, padded_len, self.num_experts)

        constrained_scores = routing_padded.masked_fill(~token_coreset_mask, float("-inf"))

        _, topk_idx = self.group_limited_topk(
            constrained_scores.reshape(-1, self.num_experts)
        )
        topk_idx = topk_idx.view(bsz, padded_len, self.top_k)

        if pad_len > 0:
            topk_idx = topk_idx[:, :seq_len, :]

        selected_scores = torch.gather(scores, dim=-1, index=topk_idx).reshape(-1, self.top_k).type_as(logits)

        topk_weight = (
            selected_scores / (selected_scores.sum(dim=-1, keepdim=True) + 1e-20)
            if self.top_k > 1
            else selected_scores
        )
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx.view(-1, self.top_k), topk_weight, logits


class LLaDA2MoeSparseMoeBlock(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config: LLaDA2MoeConfig):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self._setup_experts()
        self.gate = LLaDA2MoeGate(config)

        # @sicheng: time recorder
        self.profiler = None
        self.layer_idx = -1

        if config.num_shared_experts is not None:
            self.shared_experts = LLaDA2MoeMLP(
                config=config,
                intermediate_size=config.moe_intermediate_size
                * config.num_shared_experts,
            )

    def _setup_experts(self):
        self.experts = nn.ModuleList(
            [
                LLaDA2MoeMLP(
                    config=self.config,
                    intermediate_size=self.config.moe_intermediate_size,
                )
                for _ in range(self.config.num_experts)
            ]
        )

    # @sicheng: + time recorder
    def forward(self, hidden_states):
        identity = hidden_states
        bsz, seq_len, h = hidden_states.shape
        profiler = self.profiler
        prefix = f"layer_{self.layer_idx}"

        if profiler is not None:
            with profiler.time(f"{prefix}.moe_total"):
                with profiler.time(f"{prefix}.moe_gate"):
                    topk_idx, topk_weight, router_logits = self.gate(hidden_states)

                hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
                flat_topk_idx = topk_idx.view(-1)

                if self.training:
                    hidden_states = hidden_states.repeat_interleave(
                        self.num_experts_per_tok, dim=0
                    )
                    y = torch.empty_like(hidden_states)
                    for i, expert in enumerate(self.experts):
                        y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
                    y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
                    y = y.to(hidden_states.dtype).view(bsz, seq_len, h)
                else:
                    with profiler.time(f"{prefix}.moe_infer"):
                        y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(
                            bsz, seq_len, h
                        )

                if self.config.num_shared_experts is not None:
                    with profiler.time(f"{prefix}.moe_shared"):
                        y = y + self.shared_experts(identity)
        else:
            topk_idx, topk_weight, router_logits = self.gate(hidden_states)
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            flat_topk_idx = topk_idx.view(-1)

            if self.training:
                hidden_states = hidden_states.repeat_interleave(
                    self.num_experts_per_tok, dim=0
                )
                y = torch.empty_like(hidden_states)
                for i, expert in enumerate(self.experts):
                    y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
                y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
                y = y.to(hidden_states.dtype).view(bsz, seq_len, h)
            else:
                y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(
                    bsz, seq_len, h
                )

            if self.config.num_shared_experts is not None:
                y = y + self.shared_experts(identity)

        return y, (
            router_logits.view(bsz, seq_len, -1),
            topk_idx.view(bsz, seq_len, -1),
        )


    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()
        outputs = []
        start_idx = 0
        for i, num_tokens_tensor in enumerate(tokens_per_expert):
            num_tokens = num_tokens_tensor.item()
            if num_tokens == 0:
                continue
            end_idx = start_idx + num_tokens
            expert = self.experts[i]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out.to(x.device))
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# Copied from transformers.models.llama.modeling_llama.LlamaAttention with Llama->LLaDA2Moe
class LLaDA2MoeAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LLaDA2MoeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim or self.hidden_size // self.num_heads
        partial_rotary_factor = (
            config.partial_rotary_factor
            if hasattr(config, "partial_rotary_factor")
            else 1.0
        )
        self.rope_dim = int(self.head_dim * partial_rotary_factor)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.scaling = self.head_dim**-0.5
        self.is_causal = False

        self.query_key_value = nn.Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=config.use_qkv_bias,
        )

        if self.config.use_qk_norm:
            self.query_layernorm = LLaDA2MoeRMSNorm(
                self.head_dim, eps=config.rms_norm_eps
            )
            self.key_layernorm = LLaDA2MoeRMSNorm(
                self.head_dim, eps=config.rms_norm_eps
            )
        self.dense = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.use_bias
        )
        self.sliding_window = getattr(config, "sliding_window", None)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]

        bsz, q_len, _ = hidden_states.size()

        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(
            bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim
        )

        query_states, key_states, value_states = qkv.split(
            [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if self.config.use_qk_norm:
            query_states = self.query_layernorm(query_states)
            key_states = self.key_layernorm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.dense(attn_output)

        return attn_output, attn_weights, past_key_value


class LLaDA2MoeDecoderLayer(nn.Module):
    def __init__(self, config: LLaDA2MoeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # @sicheng: time recorder
        self.layer_idx = layer_idx
        self.profiler = None

        self.attention = LLaDA2MoeAttention(config=config, layer_idx=layer_idx)

        self.mlp = (
            LLaDA2MoeSparseMoeBlock(config)
            if (
                config.num_experts is not None
                and layer_idx >= config.first_k_dense_replace
            )
            else LLaDA2MoeMLP(config=config, intermediate_size=config.intermediate_size)
        )

        # @sicheng: time recorder
        self.is_moe_layer = isinstance(self.mlp, LLaDA2MoeSparseMoeBlock)
        if self.is_moe_layer:
            self.mlp.layer_idx = layer_idx

        self.input_layernorm = LLaDA2MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = LLaDA2MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    # @sicheng: + time recorder
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        profiler = self.profiler
        prefix = f"layer_{self.layer_idx}"

        # Self Attention
        if profiler is not None:
            with profiler.time(f"{prefix}.attention"):
                hidden_states, self_attn_weights, present_key_value = self.attention(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    position_embeddings=position_embeddings,
                    use_cache=use_cache,
                )
        else:
            hidden_states, self_attn_weights, present_key_value = self.attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                position_embeddings=position_embeddings,
                use_cache=use_cache,
            )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if profiler is not None and not self.is_moe_layer:
            with profiler.time(f"{prefix}.dense_mlp"):
                hidden_states = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)

        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None

        hidden_states = residual + hidden_states.to(residual.device)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


LLADA2MOE_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LLaDA2MoeConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaDA2Moe Model outputting raw hidden-states without any specific head on top.",
    LLADA2MOE_START_DOCSTRING,
)
class LLaDA2MoePreTrainedModel(PreTrainedModel):
    config_class = LLaDA2MoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LLaDA2MoeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = False
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


LLADA2MOE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaDA2Moe Model outputting raw hidden-states without any specific head on top.",
    LLADA2MOE_START_DOCSTRING,
)
class LLaDA2MoeModel(LLaDA2MoePreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LLaDA2MoeDecoderLayer`]

    Args:
        config: LLaDA2MoeConfig
    """

    def __init__(self, config: LLaDA2MoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                LLaDA2MoeDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flex_attention = config._attn_implementation == "flex_attention"
        self.norm = LLaDA2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LLaDA2MoeRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value

    @add_start_docstrings_to_model_forward(LLADA2MOE_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.output_router_logits
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`transformers."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )

        if position_ids is None:
            position_ids = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
            position_ids = position_ids.unsqueeze(0)
        kv_length = past_seen_tokens + seq_length
        if attention_mask.size() == (batch_size, 1, seq_length, seq_length):
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_seen_tokens,
            )
        elif attention_mask.size() == (batch_size, 1, seq_length, kv_length):
            # Non-square mask: current block tokens (q) attending to prefix KV cache + itself (kv).
            # Already in correct 4D additive format; pass through unchanged.
            pass
        else:
            raise ValueError(
                f"LLaDA2.0 only support block attention mask with shape "
                f"{(batch_size, 1, seq_length, seq_length)} or "
                f"{(batch_size, 1, seq_length, kv_length)} (prefix KV cache), "
                f"got {tuple(attention_mask.size())}"
            )
        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits and layer_outputs[-1] is not None:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_router_logits,
                ]
                if v is not None
            )
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


class LLaDA2MoeModelLM(LLaDA2MoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LLaDA2MoeConfig):
        super().__init__(config)
        self.model = LLaDA2MoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # time recorder
        self.profiler = SimpleCudaProfiler(enabled=True)

        for layer in self.model.layers:
            layer.profiler = self.profiler
            if isinstance(layer.mlp, LLaDA2MoeSparseMoeBlock):
                layer.mlp.profiler = self.profiler

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.word_embeddings

    def set_input_embeddings(self, value):
        self.model.word_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLADA2MOE_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=MoeCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer

        >>> model = LLaDA2MoeForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.output_router_logits
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            **kwargs,
        )

        loss = None
        aux_loss = None
        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        if labels is not None:
            # LLaDA2.0 will use same label position logits
            shift_logits = logits
            shift_labels = labels
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        token_type_ids=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = (
                    past_key_values.get_max_length()
                    if hasattr(past_key_values, "get_max_length")
                    else past_key_values.get_max_cache_shape()
                )
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past

    @staticmethod
    def _top_k_logits(logits, k):
        if k is None or k <= 0:
            return logits
        else:
            values, _ = torch.topk(logits, k)
            min_values = values[..., -1, None]
            return torch.where(
                logits < min_values, torch.full_like(logits, float("-inf")), logits
            )

    @staticmethod
    def _top_p_logits(logits, p):
        if p is None or p >= 1.0:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        mask_indices = torch.scatter(
            torch.full_like(logits, False, dtype=torch.bool),
            -1,
            sorted_indices,
            sorted_mask,
        )
        return logits.masked_fill(mask_indices, float("-inf"))

    def _sample_with_temperature_topk_topp(
        self, logits, temperature=1.0, top_k=0, top_p=1.0
    ):
        orig_shape = logits.shape[:-1]
        vocab_size = logits.shape[-1]
        logits = logits.reshape(-1, vocab_size)
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature
        logits = self._top_k_logits(logits, top_k)
        logits = self._top_p_logits(logits, top_p)
        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        token_prob = torch.gather(probs, -1, token)
        return token.view(*orig_shape), token_prob.view(*orig_shape)

    @staticmethod
    def _get_num_transfer_tokens(block_length, steps):
        if steps == 0:
            return torch.tensor([], dtype=torch.int64)
        base = block_length // steps
        remainder = block_length % steps
        num_transfer_tokens = torch.full((steps,), base, dtype=torch.int64)
        num_transfer_tokens[:remainder] += 1
        return num_transfer_tokens

    # @sicheng: time recorder
    def _reset_step_profile(self):
        if hasattr(self, "profiler") and self.profiler is not None:
            self.profiler.reset()

    # @sicheng: time recorder
    @staticmethod
    def _summarize_step_profile(step_profile: dict, forward_ms: float):
        attention_ms = sum(v for k, v in step_profile.items() if k.endswith(".attention"))
        dense_mlp_ms = sum(v for k, v in step_profile.items() if k.endswith(".dense_mlp"))
        moe_total_ms = sum(v for k, v in step_profile.items() if k.endswith(".moe_total"))
        moe_gate_ms = sum(v for k, v in step_profile.items() if k.endswith(".moe_gate"))
        moe_infer_ms = sum(v for k, v in step_profile.items() if k.endswith(".moe_infer"))
        moe_shared_ms = sum(v for k, v in step_profile.items() if k.endswith(".moe_shared"))

        profiled_ms = attention_ms + dense_mlp_ms + moe_total_ms
        others_ms = max(0.0, forward_ms - profiled_ms)

        return {
            "forward_ms": float(forward_ms),
            "attention_ms": float(attention_ms),
            "dense_mlp_ms": float(dense_mlp_ms),
            "moe_total_ms": float(moe_total_ms),
            "moe_gate_ms": float(moe_gate_ms),
            "moe_infer_ms": float(moe_infer_ms),
            "moe_shared_ms": float(moe_shared_ms),
            "others_ms": float(others_ms),
        }

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        temperature: int = 0.0,
        block_length: int = 32,
        steps: int = 32,
        gen_length: int = 2048,
        top_p: Optional[int] = None,
        top_k: Optional[int] = None,
        eos_early_stop: bool = False,
        minimal_topk: int = 1,
        threshold: float = 0.95,
        eos_id: int = 156892,
        mask_id: int = 156895,
    ):
        r"""
        Generates tokens using a block-wise, iterative refinement strategy.

        This method operates differently from standard autoregressive generation. It first creates a template of the
        full desired length, filled with a special `mask_id`. It then processes this template in segments (`blocks`)
        and iteratively "denoises" or "refines" the `mask_id` tokens into actual tokens over a series of `steps` for
        each block. A custom block-diagonal causal attention mask ensures that generation within a block can attend to
        all previous blocks but not future ones.

        <Tip warning={true}>

        This is a specialized generation method. The quality and speed of the output are highly dependent on the interplay
        between `block_length`, `steps`, and `threshold`. It aims to achieve faster generation through parallel
        decoding within blocks, which is a departure from the token-by-token generation of standard `.generate()` methods.

        </Tip>

        Parameters:
            inputs (`torch.Tensor`):
                The token sequence used as a prompt for the generation.
            temperature (`float`, *optional*, defaults to 0.0):
                The value used to module the next token probabilities. A value of 0.0 corresponds to greedy decoding.
            block_length (`int`, *optional*, defaults to 32):
                The size of each generation block. The model generates text in parallel within these blocks. This is a
                key parameter for controlling the granularity of the generation process.
            steps (`int`, *optional*, defaults to 32):
                The number of iterative refinement (or "denoising") steps to perform for each block. Within each block,
                the model will try to replace `mask_id` tokens with real tokens for this many iterations.
            gen_length (`int`, *optional*, defaults to 2048):
                The maximum number of tokens to generate, excluding the prompt.
            top_p (`float`, *optional*):
                If set to a float value between 0 and 1, only the most probable tokens with probabilities that add up to
                `top_p` or higher are kept for generation (nucleus sampling).
            top_k (`int`, *optional*):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            eos_early_stop (`bool`, *optional*, defaults to `False`):
                If `True`, generation will stop as soon as a valid End-Of-Sequence token is generated and confirmed,
                even if `gen_length` has not been reached.
            minimal_topk (`int`, *optional*, defaults to 1):
                A parameter used to dynamically adjust the number of refinement `steps`. The effective number of steps
                is capped at `gen_length // minimal_topk`.
            threshold (`float`, *optional*, defaults to 0.95):
                The confidence probability threshold for accepting a sampled token. During each refinement step, a
                sampled token is only kept if its probability is above this threshold. If not enough tokens meet the
                threshold, the ones with the highest confidence are chosen.
            eos_id (`int`, *optional*, defaults to 156892):
                The token ID for the end-of-sequence token. Used for `eos_early_stop`.
            mask_id (`int`, *optional*, defaults to 156895):
                The token ID used as a placeholder for tokens that are yet to be generated. This is central to the
                iterative refinement algorithm.

        Return:
            `torch.Tensor`: A string containing the generated token IDs, starting
            after the prompt and stopping at the first `eos_id` or `gen_length`.
        """
        steps = min(steps, gen_length // minimal_topk)
        input_ids = inputs.to(self.device)

        prompt_length = input_ids.shape[1]
        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length

        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=self.device))
        block_diffusion_attention_mask = (
            (
                block_mask.repeat_interleave(block_length, dim=0)
                .repeat_interleave(block_length, dim=1)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            .log()
            .to(torch.bfloat16)
        )

        position_ids = torch.arange(total_length, device=self.device).unsqueeze(0)
        x = torch.full((1, total_length), mask_id, dtype=torch.long, device=self.device)
        x[:, :prompt_length] = input_ids.clone()

        prompt_index_full = torch.zeros_like(x, dtype=torch.bool)
        prompt_index_full[:, :prompt_length] = True

        # @sicheng
        total_avg_unique_experts = 0.0
        total_denoise_forwards = 0

        prefill_blocks = prompt_length // block_length

        denoising_steps_per_block = steps
        num_transfer_tokens_schedule = self._get_num_transfer_tokens(
            block_length, denoising_steps_per_block
        )
        # accumulated_kv: grows incrementally block by block (O(N) total cost).
        # None means no prefix tokens yet (first block with empty prompt).
        accumulated_kv = None

        for num_block in range(prefill_blocks, num_blocks):
            prefix_end = num_block * block_length
            current_window_end = (num_block + 1) * block_length

            # --- Build/extend the accumulated KV for the prefix ---
            if prefix_end > 0:
                if accumulated_kv is None:
                    # First block: full prefill of the prompt (only time we process all tokens).
                    prefix_out = self.forward(
                        x[:, :prefix_end],
                        attention_mask=block_diffusion_attention_mask[
                            :, :, :prefix_end, :prefix_end
                        ],
                        position_ids=position_ids[:, :prefix_end],
                        use_cache=True,
                        return_dict=True,
                    )
                    accumulated_kv = prefix_out.past_key_values
                    del prefix_out
                else:
                    # Subsequent blocks: forward only the 32 newly committed tokens,
                    # extending the accumulated KV by one block at O(32 × prefix_end) cost.
                    prev_prefix_end = prefix_end - block_length
                    incr_out = self.forward(
                        x[:, prev_prefix_end:prefix_end],
                        attention_mask=block_diffusion_attention_mask[
                            :, :, prev_prefix_end:prefix_end, :prefix_end
                        ],
                        position_ids=position_ids[:, prev_prefix_end:prefix_end],
                        past_key_values=accumulated_kv,
                        use_cache=True,
                        return_dict=True,
                    )
                    accumulated_kv = incr_out.past_key_values
                    del incr_out

                # Extract tensor references (no copy). DynamicLayer.update() creates new
                # tensors via torch.cat, so the originals here are never mutated by denoising.
                prefix_kv_pairs = [(layer.keys, layer.values) for layer in accumulated_kv.layers]
            else:
                prefix_kv_pairs = []

            # Attention mask for block tokens attending to [prefix KV cache + block itself]
            block_attn_mask = block_diffusion_attention_mask[
                :, :, prefix_end:current_window_end, :current_window_end
            ]
            block_position_ids = position_ids[:, prefix_end:current_window_end]
            # View into x: mutations here propagate to x automatically.
            cur_block_x = x[:, prefix_end:current_window_end]

            for step in range(denoising_steps_per_block):
                active_block_mask = cur_block_x == mask_id
                if active_block_mask.sum() == 0:
                    break

                # Build a fresh DynamicCache from static prefix references each step.
                # DynamicLayer.update() does torch.cat → new tensor stored back into
                # layer.keys/values, leaving the original prefix tensors untouched.
                if prefix_kv_pairs:
                    fresh_cache = DynamicCache()
                    for (pk, pv) in prefix_kv_pairs:
                        layer = DynamicLayer()
                        layer.keys = pk
                        layer.values = pv
                        layer.is_initialized = True
                        layer.dtype = pk.dtype
                        layer.device = pk.device
                        fresh_cache.layers.append(layer)
                else:
                    fresh_cache = None

                outputs = self.forward(
                    cur_block_x,
                    attention_mask=block_attn_mask,
                    position_ids=block_position_ids,
                    past_key_values=fresh_cache,
                    use_cache=False,
                    output_router_logits=True,
                    return_dict=True,
                )
                # logits is already block-only: [bsz, block_length, vocab_size]
                logits = outputs.logits

                # @sicheng
                step_avg_unique, step_per_layer_unique = self._compute_avg_unique_experts_per_layer(
                    outputs.router_logits,
                    active_block_mask=active_block_mask,
                    block_length=block_length,
                )
                total_avg_unique_experts += step_avg_unique
                total_denoise_forwards += 1

                x0, x0_p = self._sample_with_temperature_topk_topp(
                    logits, temperature=temperature, top_k=top_k, top_p=top_p
                )

                num_to_transfer = num_transfer_tokens_schedule[step].item()
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)

                confidence = torch.where(active_block_mask, x0_p, -torch.inf)
                high_conf_mask = confidence[0] > threshold
                num_high_confidence = high_conf_mask.sum().item()

                if num_high_confidence >= num_to_transfer:
                    transfer_index[0] = high_conf_mask
                else:
                    _, idx = torch.topk(
                        confidence[0],
                        k=min(num_to_transfer, active_block_mask.sum().item()),
                    )
                    transfer_index[0, idx] = True

                if transfer_index.any():
                    # cur_block_x is a view of x, so x is updated in-place.
                    cur_block_x[transfer_index] = x0[transfer_index]
                if eos_early_stop and (x0[transfer_index] == eos_id).any():
                    eos_pos_in_x = (x[0] == eos_id).nonzero(as_tuple=True)
                    if len(eos_pos_in_x[0]) > 0:
                        eos_pos = eos_pos_in_x[0][0].item()
                        if (x[0, prompt_length:eos_pos] != mask_id).all():
                            final_x = x[:, :total_length][:, :eos_pos + 1]
                            if total_denoise_forwards > 0:
                                avg_unique_experts_per_layer = (
                                    total_avg_unique_experts / total_denoise_forwards
                                )
                            else:
                                avg_unique_experts_per_layer = 0.0
                            return final_x, avg_unique_experts_per_layer

            if (
                eos_id is not None
                and (x[0, prompt_length:current_window_end] == eos_id).any()
            ):
                break

        generated_answer = x[:, : prompt_length + gen_length]

        mask_positions = (generated_answer[0][input_ids.shape[1] :] == eos_id).nonzero(
            as_tuple=True
        )[0]
        if len(mask_positions) > 0:
            first_mask_position = mask_positions[0].item()
        else:
            first_mask_position = gen_length

        # @sicheng
        # return generated_answer[
        #     :, input_ids.shape[1] : input_ids.shape[1] + first_mask_position + 1
        # ]
        if total_denoise_forwards > 0:
            avg_unique_experts_per_layer = (
                total_avg_unique_experts / total_denoise_forwards
            )
        else:
            avg_unique_experts_per_layer = 0.0

        return generated_answer[
            :, input_ids.shape[1] : input_ids.shape[1] + first_mask_position + 1
        ], avg_unique_experts_per_layer

    # @sicheng: not update, old version
    @torch.no_grad()
    def generate_timer(
        self,
        inputs: Optional[torch.Tensor] = None,
        temperature: int = 0.0,
        block_length: int = 32,
        steps: int = 32,
        gen_length: int = 2048,
        top_p: Optional[int] = None,
        top_k: Optional[int] = None,
        eos_early_stop: bool = False,
        minimal_topk: int = 1,
        threshold: float = 0.95,
        eos_id: int = 156892,
        mask_id: int = 156895,
    ):
        steps = min(steps, gen_length // minimal_topk)
        input_ids = inputs.to(self.device)

        prompt_length = input_ids.shape[1]
        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length

        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=self.device))
        block_diffusion_attention_mask = (
            (
                block_mask.repeat_interleave(block_length, dim=0)
                .repeat_interleave(block_length, dim=1)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            .log()
            .to(torch.bfloat16)
        )

        position_ids = torch.arange(total_length, device=self.device).unsqueeze(0)
        x = torch.full((1, total_length), mask_id, dtype=torch.long, device=self.device)
        x[:, :prompt_length] = input_ids.clone()

        total_avg_unique_experts = 0.0
        total_denoise_forwards = 0

        # time records
        step_records = []

        generate_start = torch.cuda.Event(enable_timing=True)
        generate_end = torch.cuda.Event(enable_timing=True)
        generate_start.record()

        prefill_blocks = prompt_length // block_length

        denoising_steps_per_block = steps
        num_transfer_tokens_schedule = self._get_num_transfer_tokens(
            block_length, denoising_steps_per_block
        )

        for num_block in range(prefill_blocks, num_blocks):
            current_window_end = (num_block + 1) * block_length
            cur_x = x[:, :current_window_end]
            cur_attn_mask = block_diffusion_attention_mask[
                :, :, :current_window_end, :current_window_end
            ]
            cur_position_ids = position_ids[:, :current_window_end]

            for step in range(denoising_steps_per_block):
                active_block_mask = cur_x[:, -block_length:] == mask_id
                active_tokens = int(active_block_mask.sum().item())
                if active_tokens == 0:
                    break

                self._reset_step_profile()

                step_start = torch.cuda.Event(enable_timing=True)
                step_end = torch.cuda.Event(enable_timing=True)

                step_start.record()
                outputs = self.forward(
                    cur_x,
                    attention_mask=cur_attn_mask,
                    position_ids=cur_position_ids,
                    output_router_logits=True,
                    return_dict=True,
                )
                step_end.record()
                torch.cuda.synchronize()
                step_forward_ms = step_start.elapsed_time(step_end)

                logits = outputs.logits

                step_avg_unique, step_per_layer_unique = self._compute_avg_unique_experts_per_layer(
                    outputs.router_logits,
                    active_block_mask=active_block_mask,
                    block_length=block_length,
                )

                total_avg_unique_experts += step_avg_unique
                total_denoise_forwards += 1

                active_logits = logits[:, -block_length:, :]
                x0, x0_p = self._sample_with_temperature_topk_topp(
                    active_logits, temperature=temperature, top_k=top_k, top_p=top_p
                )

                num_to_transfer = num_transfer_tokens_schedule[step].item()
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)

                confidence = torch.where(active_block_mask, x0_p, -torch.inf)
                high_conf_mask = confidence[0] > threshold
                num_high_confidence = high_conf_mask.sum().item()

                if num_high_confidence >= num_to_transfer:
                    transfer_index[0] = high_conf_mask
                else:
                    _, idx = torch.topk(
                        confidence[0],
                        k=min(num_to_transfer, active_block_mask.sum().item()),
                    )
                    transfer_index[0, idx] = True

                accepted_tokens = int(transfer_index.sum().item())

                if transfer_index.any():
                    cur_x[:, -block_length:][transfer_index] = x0[transfer_index]

                step_profile = self.profiler.snapshot() if self.profiler is not None else {}
                step_time_summary = self._summarize_step_profile(step_profile, step_forward_ms)

                step_records.append(
                    {
                        "block_idx": int(num_block),
                        "step_idx": int(step),
                        "current_window_end": int(current_window_end),
                        "active_tokens": active_tokens,
                        "accepted_tokens": accepted_tokens,
                        "avg_unique_experts": float(step_avg_unique),
                        "per_layer_unique_experts": [float(v) for v in step_per_layer_unique],
                        "layer_time_breakdown_ms": {k: float(v) for k, v in step_profile.items()},
                        **step_time_summary,
                    }
                )

                if eos_early_stop and (x0[transfer_index] == eos_id).any():
                    eos_pos_in_x = (cur_x[0] == eos_id).nonzero(as_tuple=True)
                    if len(eos_pos_in_x[0]) > 0:
                        eos_pos = eos_pos_in_x[0][0].item()
                        if (cur_x[0, prompt_length:eos_pos] != mask_id).all():
                            final_x = x[:, :total_length][:, : eos_pos + 1]

                            if total_denoise_forwards > 0:
                                avg_unique_experts_per_layer = (
                                    total_avg_unique_experts / total_denoise_forwards
                                )
                            else:
                                avg_unique_experts_per_layer = 0.0

                            generate_end.record()
                            torch.cuda.synchronize()
                            generate_total_ms = generate_start.elapsed_time(generate_end)

                            time_records = {
                                "generate_total_ms": float(generate_total_ms),
                                "num_denoise_forwards": int(total_denoise_forwards),
                                "step_records": step_records,
                            }

                            return final_x, avg_unique_experts_per_layer, time_records

            x[:, :current_window_end] = cur_x
            if (
                eos_id is not None
                and (x[0, prompt_length:current_window_end] == eos_id).any()
            ):
                break

        generated_answer = x[:, : prompt_length + gen_length]

        mask_positions = (generated_answer[0][input_ids.shape[1] :] == eos_id).nonzero(
            as_tuple=True
        )[0]
        if len(mask_positions) > 0:
            first_mask_position = mask_positions[0].item()
        else:
            first_mask_position = gen_length

        if total_denoise_forwards > 0:
            avg_unique_experts_per_layer = (
                total_avg_unique_experts / total_denoise_forwards
            )
        else:
            avg_unique_experts_per_layer = 0.0

        generate_end.record()
        torch.cuda.synchronize()
        generate_total_ms = generate_start.elapsed_time(generate_end)

        time_records = {
            "generate_total_ms": float(generate_total_ms),
            "num_denoise_forwards": int(total_denoise_forwards),
            "step_records": step_records,
        }

        return (
            generated_answer[
                :, input_ids.shape[1] : input_ids.shape[1] + first_mask_position + 1
            ],
            avg_unique_experts_per_layer,
            time_records,
        )

    # @sicheng
    @staticmethod
    def _compute_avg_unique_experts_per_layer(
        router_logits,
        active_block_mask=None,
        block_length=32,
    ):
        """
        Returns:
            avg_unique: float
                average unique activated experts across MoE layers for this step
            per_layer_unique: List[float]
                unique activated experts for each MoE layer
        """
        if router_logits is None or len(router_logits) == 0:
            return 0.0, []

        per_layer_unique = []

        for layer_out in router_logits:
            # layer_out = (router_logits, topk_idx)
            _, topk_idx = layer_out   # shape: [bsz, seq_len, top_k]

            # only keep current block
            if block_length is not None:
                topk_idx = topk_idx[:, -block_length:, :]

            # only keep active positions
            # if active_block_mask is not None:
            #     # active_block_mask shape: [1, block_length]
            #     # after boolean indexing: [num_active_tokens, top_k]
            #     topk_idx = topk_idx[active_block_mask]

            if topk_idx.numel() == 0:
                per_layer_unique.append(0.0)
            else:
                unique_num = torch.unique(topk_idx.reshape(-1)).numel()
                per_layer_unique.append(float(unique_num))

        avg_unique = sum(per_layer_unique) / len(per_layer_unique)
        return avg_unique, per_layer_unique