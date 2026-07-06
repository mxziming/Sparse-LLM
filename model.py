from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
from typing import Optional, Union, List
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.utils import TransformersKwargs
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2MLP, Qwen2RMSNorm, Qwen2PreTrainedModel, Qwen2RotaryEmbedding
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.generation import GenerationMixin
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class BaseModelOutputWithPast(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[List[tuple[torch.FloatTensor, ...]]] = None


@dataclass
class CausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[List[tuple[torch.FloatTensor, ...]]] = None


class Indexer(nn.Module):
    """
    Lightning Indexer — the DSA scoring module. Adds self.bypass (bool):
        True  -> skip sparse selection, behave like dense attention (for Break-even benchmark)
        False -> normal top-k sparse selection (default)
    """

    def __init__(self, config):
        super().__init__()

        # Store the model's hidden dimension
        self.hidden_size: int = config.hidden_size
        # Number of attention heads in the main model
        self.n_heads: int = config.num_attention_heads
        # Number of key/value heads
        self.key_value_heads = config.num_key_value_heads   
        # Dimension of each attention head, derived from hidden_size / num_heads.
        self.head_dim: int = config.hidden_size // config.num_attention_heads

        # Fixed top-k size — how many tokens the Indexer selects per query position.
        self.index_topk: int = 128

        # Linear projection from hidden_size down to a SINGLE shared key dimension (not per-head)
        # 从 hidden_size 投影到一个单一的、所有头共享的 key 维度（不分头）
        self.wk = nn.Linear(self.hidden_size, self.head_dim)

        # Linear projection producing one importance weight per head, per token.
        # 线性投影，为每个 token 在每个头上产生一个重要性权重。
        self.weights_proj = nn.Linear(self.hidden_size, self.n_heads)

        # Register a non-persistent buffer to hold cached keys during incremental decoding.
        # 注册一个非持久化的 buffer，用于在增量解码时缓存 key。
        self.register_buffer("k_cache", None, persistent=False)

        # Break-even switch / Break-even 开关
        # bypass=True -> 返回全部 token 的索引，退化为 dense attention。
        # bypass=False -> 正常的 top-k 稀疏选择（默认）。
        self.bypass: bool = False

    def forward(
        self,
        hidden_states: torch.Tensor,   # [B,S,hidden], 输入激活值
        query_states: torch.Tensor,    # [B,H,S,head_dim], 来自主 attention 模块的 query
        cos: torch.Tensor,              # 旋转位置编码的 cosine 分量
        sin: torch.Tensor,              # 旋转位置编码的 sine 分量
        mask=None,                      # 可选的加法式注意力掩码
    ):
        # Read batch size and sequence length from the input shape.
        bsz, seqlen, _ = hidden_states.size()   # [B, S, hidden_size]

        # Project hidden states down to the single shared key vector per token.
        # [B, S, hidden_size] -> [B, S, head_dim] where head_dim = hidden_size / heads
        key_states = self.wk(hidden_states)

        # Compute per-head importance weights, scaled by 1/sqrt(n_heads) for numerical stability.
        # [B, S, hidden_size] -> [B, S, heads]
        weights = self.weights_proj(hidden_states) * self.n_heads ** -0.5

        # During prefill (multi-token input), overwrite the cache with the freshly computed keys.
        # 在 prefill 阶段（多 token 输入），用新算出的 key 覆盖缓存。
        if seqlen > 1:
            self.k_cache = key_states

        # During decode (single new token), concatenate the new key onto the cached history.
        # 在 decode 阶段（单个新 token），把新 key 拼接到已缓存的历史记录后面。
        if seqlen == 1:
            key_states = torch.cat([self.k_cache, key_states], dim=1)
            # Update the cache to include this newest key, ready for the next decode step.
            # 更新缓存，包含这个最新的 key，为下一步 decode 做准备。
            self.k_cache = key_states

        # 函数期待四个维度，添加 head 维度，[B, 1, S, head_dim]
        key_states = key_states.unsqueeze(1)
        # 函数期待两个输入，分别是 query 和 key
        key_states, _ = apply_rotary_pos_emb(key_states, key_states, cos, sin)

        # Compute raw scores: query (per head) dotted with the shared key, broadcasting over heads
        # [B, heads, S, head_dim] * [B, 1, head_dim, S] -> [B, heads, S, S]
        attn_scores = query_states @ key_states.transpose(2, 3)

        # Apply ReLU — this is what makes scores sparse and non-negative (irrelevant tokens score exactly 0).
        attn_scores = F.relu(attn_scores, inplace=False)

        # Multiply by the per-head importance weights, broadcasting weights across the key dimension.
        # [B, S, heads] -> [B, heads, S, 1] * [B, heads, S, S] -> [B, heads, S, S]
        attn_scores = weights.transpose(1, 2).unsqueeze(-1) * attn_scores

        # Sum across heads, collapsing to a single combined score per (query, key) pair.
        # [B, heads, S, S] -> [B, 1, S, S]
        attn_scores = attn_scores.sum(1, keepdim=True)

        # If a mask was provided (e.g. causal mask), add it so masked positions become -inf before top-k selection.
        if mask is not None:
            attn_scores = attn_scores + mask

        # bypass switch — dense mode returns the full index set
        if self.bypass:
            # Returning every index means the downstream scatter-based mask masks nothing -> standard dense attention.
            # 返回每一个索引意味着下游基于 scatter 构建的掩码什么都不遮 —— 等价于标准 dense attention。
            # k = seq_len (S)
            k = key_states.shape[2]
            # Build an index tensor [1,1,1,k] containing 0..k-1, then broadcast it to every batch and query position.
            # 构造一个 [1,1,1,k] 的索引张量，内容是 0..k-1，再广播到每个 batch 和每个 query 位置。
            # [k,] -> [1, 1, 1, k] -> [B, 1, S, k] where k = S
            topk_indices = torch.arange(k, device=hidden_states.device).view(1, 1, 1, k).expand(bsz, 1, seqlen, k)
        else:
            # Normal path — select the index_topk highest-scoring positions (capped by however many keys actually exist).
            # [B, 1, S, S] -> [B, 1, S, topk] or [B, 1, S, k] where k = S
            topk_indices = attn_scores.topk(min(self.index_topk, key_states.shape[2]), dim=-1)[1]

        # Return both the selected indices and the raw scores (the latter used elsewhere for KL distillation).
        # 返回选出的索引和原始打分（后者在别处用于 KL 蒸馏）。
        # Shape is [B, 1, S, topk] or [B, 1, S, S] and [B, 1, S, S] respectively.
        return topk_indices, attn_scores


class Qwen2Attention(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None
        self.indexer = Indexer(config)
        self.chunked_prefill = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        bsz, seqlen, _ = hidden_states.size()   # [batch_size, seq_len, hidden_size]
        input_shape = hidden_states.shape[:-1]  # [batch_size, seq_len]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Step 1: Q/K/V Projection
        # [B, s, hidden_size] -> [B, s, heads * head_dim] -> [B, s, heads, head_dim] -> [B, heads, s, head_dim]
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # [B, s, hidden_size] -> [B, s, key_heads * head_dim] -> [B, s, key_heads, head_dim] -> [B, key_heads, s, head_dim]
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # [B, s, hidden_size] -> [B, s, value_heads * head_dim] -> [B, s, value_heads, head_dim] -> [B, value_heads, s, head_dim]
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Step 2: RoPE
        cos, sin = position_embeddings
        # query: [B, heads, s, head_dim]
        # key/value: [B, key_heads, s, head_dim], [B, value_heads, s, head_dim]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Step 3: K/V Cache
        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # [B, key_heads, S_cached, head_dim] -> [B, key_heads, S_cached + 1, head_dim]
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Step 4: Attention Score
        # Repeat K/V for heads/key_heads (heads/value_heads) times
        # [B, key_heads, s, head_dim] -> [B, heads, s, head_dim]
        # [B, value_heads, s, head_dim] -> [B, heads, s, head_dim]
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        raw_attn_weights = None
        indexer_attn_scores = None
        topk_indices = None
        
        # Step 5a: inference (chunked prefill)
        if self.chunked_prefill and seqlen > 1:
            # build Indexer.k_cache chunk by chunk, [B, chunk, head_dim]
            idx_key = self.indexer.wk(hidden_states)
            
            if self.indexer.k_cache is None:
                self.indexer.k_cache = idx_key
            else:
                self.indexer.k_cache = torch.cat([self.indexer.k_cache, idx_key], dim=1)

            S_q, S_kv = seqlen, key_states.shape[2]
            cache_start = S_kv - S_q

            causal_mask = torch.full((bsz, 1, S_q, S_kv), float('-inf'), device=hidden_states.device)
            for i in range(S_q):
                causal_mask[:, :, i, : cache_start + i + 1] = 0.0

            # [B, heads, chunk, head_dim] * [B, heads, head_dim, S] -> [B, heads, chunk, S]
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
            attn_weights = attn_weights + causal_mask
            attn_weights = attn_weights.softmax(dim=-1, dtype=attn_weights.dtype)
            attn_output  = torch.matmul(attn_weights, value_states)

            topk_indices        = None
            indexer_attn_scores = None
            raw_attn_weights    = None
        
        # Step 5b: train (warmup/train)
        elif attention_mask is not None:
            # [B, heads, s, head_dim] * [B, heads, head_dim, s] -> [B, heads, s, s]
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
            
            if attention_mask.dtype == torch.bool:
                # 反转，True原本表示 “可以看”，False表示 “被遮住”
                # 现在True的位置变成False
                attention_mask = attention_mask.logical_not()
                # 将反转后True的位置，也就是原本False的位置变成-inf
                attention_mask = attention_mask.float().masked_fill(attention_mask, float('-inf'))

            # [B, 1, s, topk] or [B, 1, s, s], [B, 1, s, s]
            topk_indices, indexer_attn_scores = self.indexer(hidden_states, query_states, cos, sin, mask=attention_mask)

            # add mask to attention weights [B, heads, s, s]
            raw_attn_weights = attn_weights + attention_mask
            
            # first we create a tensor full of -inf with the shape of [B, 1, s, s]
            # then we at the last dimension change the value at positions of topk_indices to 0
            # [B, 1, s, s]
            index_mask = torch.full((bsz, 1, seqlen, seqlen), float("-inf"), device=hidden_states.device).scatter(-1, topk_indices, 0)
            # add mask to indexer weights, which is double-check
            # we already add this mask inside indexer
            index_mask = index_mask + attention_mask

            # select the top-k elements per query token
            # [B, heads, s, s] + [B, 1, s, s] -> [B, heads, s, s]
            attn_weights = attn_weights + index_mask
            attn_weights = attn_weights.softmax(dim=-1, dtype=attn_weights.dtype)

            # [B, heads, s, s] * [B, heads, s, head_dim] = [B, heads, s, head_dim]
            attn_output = torch.matmul(attn_weights, value_states)
        
        else:
            # Step 5c: inference (prefill)
            if seqlen > 1:
                # [B, heads, s, head_dim] * [B, heads, head_dim, s] -> [B, heads, s, s]
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

                # 先创建一个全为 True 的布尔矩阵
                # 取下三角部分（包含对角线），上三角置为 False
                mask = torch.tril(torch.ones((bsz, 1, seqlen, seqlen), device=hidden_states.device, dtype=torch.bool), diagonal=0)
                # 将 False 的位置替换成-inf
                mask = mask.logical_not()
                mask = mask.float().masked_fill(mask, float('-inf'))

                topk_indices, indexer_attn_scores = self.indexer(hidden_states, query_states, cos, sin, mask=mask)

                # 添加掩码至 attention weights
                raw_attn_weights = attn_weights + mask
            
                # 添加掩码至 indexer attention weights
                index_mask = torch.full((bsz, 1, seqlen, seqlen), float("-inf"), device=hidden_states.device).scatter(-1, topk_indices, 0)
                index_mask = index_mask + mask

                attn_weights = attn_weights + index_mask
                attn_weights = attn_weights.softmax(dim=-1, dtype=attn_weights.dtype)

                # [B, heads, s, s] * [B, heads, s, head_dim] = [B, heads, s, head_dim]
                attn_output = torch.matmul(attn_weights, value_states)
            
            # Step 5d: inference (generate)
            else:
                
                if self.indexer.bypass:
                    # [B, heads, s, head_dim] * [B, heads, head_dim, S] -> [B, heads, s, S]
                    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

                    # Softmax
                    attn_weights = attn_weights.softmax(dim=-1, dtype=attn_weights.dtype)

                    # [B, heads, s, s] * [B, heads, s, head_dim] = [B, heads, s, head_dim]
                    attn_output = torch.matmul(attn_weights, value_states)

                else:
                    topk_indices, indexer_attn_scores = self.indexer(hidden_states, query_states, cos, sin, mask=None)

                    K = topk_indices.shape[-1]
                    # [B, 1, s, topk] -> [B, heads, s, topk]
                    gather_idx = topk_indices.expand(-1, self.config.num_attention_heads, -1, -1)
                    # [B, heads, s, topk] -> [B, heads, s, topk, 1] -> [B, heads, s, topk, head_dim]
                    gather_idx_d = gather_idx.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)

                    key_states_expanded = key_states.unsqueeze(2)          # [B, heads, 1, s, head_dim]
                    value_states_expanded = value_states.unsqueeze(2)      # [B, heads, 1, s, head_dim]

                    key_states_sparse = torch.gather(
                        key_states_expanded.expand(-1, -1, 1, -1, -1),
                        3,
                        gather_idx_d,
                    ).squeeze(2)    # [B, heads, topk, head_dim]
 
                    value_states_sparse = torch.gather(
                        value_states_expanded.expand(-1, -1, 1, -1, -1),
                        3,
                        gather_idx_d,
                    ).squeeze(2)    # [B, heads, topk, head_dim]

                    # [B, heads, s, head_dim] * [B, heads, head_dim, topk] -> [B, heads, s, topk]
                    attn_weights = torch.matmul(query_states, key_states_sparse.transpose(2, 3)) * self.scaling

                    # Softmax
                    attn_weights = attn_weights.softmax(dim=-1, dtype=attn_weights.dtype)

                    # [B, heads, s, s] * [B, heads, s, head_dim] = [B, heads, s, head_dim]
                    attn_output = torch.matmul(attn_weights, value_states_sparse)
        
        # [B, heads, s, head_dim] -> [B, s, heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [B, s, heads, head_dim] -> [B, s, heads * head_dim]
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        # [B, s, heads * head_dim] -> [B, s, hidden_size]
        attn_output = self.o_proj(attn_output)

        # [B, s, hidden_size]
        # ([B, 1, s, top_k], [B, heads, s, s], [B, 1, s, s])
        return attn_output, (topk_indices, raw_attn_weights, indexer_attn_scores)


class Qwen2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, attentions = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, attentions
  
    
class Qwen2Model(Qwen2PreTrainedModel):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_attentions = []
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states, attentions = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                output_attentions=output_attentions,
                **kwargs,
            )
            all_attentions.append(attentions)

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            attentions=all_attentions if output_attentions else None,
        )


class Qwen2ForCausalLM(Qwen2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/Qwen2.5-0.5B-Instruct/')
    model = AutoModelForCausalLM.from_pretrained('/home/user/Downloads/Qwen2.5-0.5B-Instruct/')

    messages = [{"role": "user", "content": "你好"}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize = False)
    # inputs = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=48)
    inputs = tokenizer(text, return_tensors="pt")['input_ids']
    print(inputs)
    
    output = model.generate(inputs, do_sample=False)
    print(tokenizer.decode(output[0]))

    # for layer in model.model.layers:
    #     old_self_attn = layer.self_attn
    #     new_self_attn = Qwen2Attention(layer.self_attn.config, layer.self_attn.layer_idx)
    #     new_self_attn.load_state_dict(old_self_attn.state_dict(), strict=False)
    #     layer.self_attn = new_self_attn

    model = Qwen2ForCausalLM.from_pretrained('/home/user/Downloads/Qwen2.5-0.5B-Instruct/')
    output = model.generate(inputs, do_sample=False)
    print(tokenizer.decode(output[0]))