import torch
import torch.nn

def scaled_dot_product_attention(query, key, value, att_mask=None, is_causal=False, scale=None, enable_gqa=False):
    # Query: B, H_q, L, D
    # Key: B, H_k, S, D
    # Value: B, H_v, S, D

    L, S = query.size(-2), key.size(-2)
    if scale is None:
        scale = query.size(-1) ** -0.5

    attn_bias = torch.zeros((L, S), dtype=query.dtype, device=query.device)
    if is_causal:
        assert att_mask is None
        att_mask = (query.new_ones(L, S).tril(diagonal=0) == 1)

    attn_bias.masked_fill_(~att_mask, float('-inf'))

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)// key.size(-3), dim=-3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), dim=-3)
    
    attn_weights = query @ key.transpose(-2, -1) * scale
    attn_weights += attn_bias
    attn_weights = torch.nn.softmax(attn_weights, dim=-1)
    attn_weights = attn_weights @ value
    return attn_weights

def group_query_attention(x, Wq, Wk, Wv, att_mask = None, is_causal = False, enable_gqa=False):
    query = torch.einsum('bld,dh->blh', x, Wq)
    key = torch.einsum('bld,dh->blh', x, Wk)
    value = torch.einsum('bld,dh->blh', x, Wv)

    query = query.view(B, Q_L, H_q, -1).transpose(1, 2)
    key = key.view(B, K_L, H_kv, -1).transpose(1, 2)
    value = value.view(B, V_L, H_vv, -1).transpose(1, 2)

    query, key = rotary_emb(query, key)

    attention = scaled_dot_product_attention(
        query, key, value,
        att_mask=att_mask,
        is_causal=is_causal,
        scale=None,
        enable_gqa=enable_gqa
    )
     



    
