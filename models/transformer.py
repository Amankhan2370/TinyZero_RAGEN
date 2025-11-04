import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange, repeat
from types import SimpleNamespace

class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings for transformer attention."""
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return rearrange(emb, "n d -> 1 1 n d")

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    q_rot = apply_rotary_pos_emb_single(q, freqs)
    k_rot = apply_rotary_pos_emb_single(k, freqs)
    return q_rot, k_rot

def apply_rotary_pos_emb_single(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    x_rot = x * freqs.cos() + rotate_half(x) * freqs.sin()
    return x_rot

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * norm

class SwiGLU(nn.Module):
    """SwiGLU activation function for MLP."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class Attention(nn.Module):
    """Multi-head attention with rotary embeddings."""
    def __init__(self, dim: int, num_heads: int, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, dim, bias=False)
        self.rotary_emb = RotaryEmbedding(head_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)
        freqs = self.rotary_emb(x, seq_len)
        q, k = apply_rotary_pos_emb(q, k, freqs)
        scores = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if attention_mask is not None:
            scores = scores + attention_mask
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.einsum("b h i j, b h j d -> b h i d", attn_weights, v)
        attn_output = rearrange(attn_output, "b h s d -> b s (h d)")
        return self.o_proj(attn_output)

class TransformerBlock(nn.Module):
    """Single transformer block with attention and MLP."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(dim, num_heads, dim // num_heads, dropout)
        self.mlp_norm = RMSNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SwiGLU(dim, mlp_hidden_dim)
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = x + self.attn(self.attn_norm(x), attention_mask)
        h = h + self.mlp(self.mlp_norm(h))
        return h

class TransformerLM(nn.Module):
    """Transformer-based language model for RL training."""
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, 
                 num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.token_embeddings.weight
        self.apply(self._init_weights)
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> dict:
        # Added input_ids bounds check to prevent out-of-vocab indices
        if (input_ids >= self.vocab_size).any() or (input_ids < 0).any():
            invalid = input_ids[(input_ids >= self.vocab_size) | (input_ids < 0)]
            raise ValueError(
                f"Input contains token indices out of embedding range (vocab_size={self.vocab_size}): {invalid}"
            )
        x = self.token_embeddings(input_ids)
        if attention_mask is None:
            seq_len = input_ids.shape[1]
            device = input_ids.device
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
            attention_mask = (1.0 - causal_mask) * -1e9
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor = None, max_length: int = 50,
                 temperature: float = 1.0, do_sample: bool = True,
                 return_dict_in_generate: bool = False, output_scores: bool = False,
                 num_return_sequences: int = 1, pad_token_id: int = None,
                 attention_mask: torch.Tensor = None, **kwargs):
        """Generate tokens autoregressively with Hugging Face compatibility."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        batch_size = input_ids.shape[0]
        if num_return_sequences > 1:
            input_ids = input_ids.repeat(num_return_sequences, 1)
        generated = input_ids
        scores = []
        for _ in range(max_length - input_ids.shape[1]):
            outputs = self.forward(generated)
            next_token_logits = outputs["logits"][:, -1, :] / temperature
            if output_scores:
                scores.append(next_token_logits)
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tokens], dim=1)
        if return_dict_in_generate:
            return SimpleNamespace(
                sequences=generated,
                scores=tuple(scores) if output_scores else None
            )
        return generated
