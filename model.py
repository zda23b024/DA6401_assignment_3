"""
model.py - Transformer architecture for DA6401 Assignment 3.

The public signatures in this file are kept stable for the autograder.
"""

import copy
import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V.

    mask must be broadcastable to (..., seq_q, seq_k). True entries are masked.
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask.to(dtype=torch.bool), torch.finfo(scores.dtype).min)
    attn_w = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_w, V)
    return output, attn_w


def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """Return encoder padding mask with shape [batch, 1, 1, src_len]."""
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """Return decoder padding + causal mask with shape [batch, 1, tgt_len, tgt_len]."""
    _, tgt_len = tgt.shape
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    causal_mask = torch.triu(
        torch.ones((tgt_len, tgt_len), device=tgt.device, dtype=torch.bool),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(1)
    return pad_mask | causal_mask


class MultiHeadAttention(nn.Module):
    """Multi-head attention implemented from first principles."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)

        def split_heads(x: torch.Tensor) -> torch.Tensor:
            return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        q = split_heads(self.w_q(query))
        k = split_heads(self.w_k(key))
        v = split_heads(self.w_v(value))

        attn_out, attn_w = scaled_dot_product_attention(q, k, v, mask)
        self.attn_weights = attn_w
        attn_out = self.dropout(attn_out)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(attn_out)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)].to(dtype=x.dtype))


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """One Transformer encoder layer."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        attn = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn))
        ff = self.ffn(x)
        return self.norm2(x + self.dropout2(ff))


class DecoderLayer(nn.Module):
    """One Transformer decoder layer."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        self_attn = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn))
        cross_attn = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn))
        ff = self.ffn(x)
        return self.norm3(x + self.dropout3(ff))


class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """Full encoder-decoder Transformer for sequence-to-sequence tasks."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        N: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        checkpoint_path: str = None,
    ) -> None:
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.N = N
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.encoder = Encoder(EncoderLayer(d_model, num_heads, d_ff, dropout), N)
        self.decoder = Decoder(DecoderLayer(d_model, num_heads, d_ff, dropout), N)
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self.model_config = {
            "src_vocab_size": src_vocab_size,
            "tgt_vocab_size": tgt_vocab_size,
            "d_model": d_model,
            "N": N,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "dropout": dropout,
        }
        self._reset_parameters()

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.load_state_dict(state_dict)

    def _reset_parameters(self) -> None:
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def encode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        src_emb = self.src_embed(src) * math.sqrt(self.d_model)
        src_emb = self.positional_encoding(src_emb)
        return self.encoder(src_emb, src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt_emb = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.positional_encoding(tgt_emb)
        decoded = self.decoder(tgt_emb, memory, src_mask, tgt_mask)
        return self.generator(decoded)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str) -> str:
        """
        Translate a German sentence to English using attached vocab/tokenizer attributes.
        """
        if not all(hasattr(self, name) for name in ("src_vocab", "tgt_vocab", "src_tokenizer")):
            raise RuntimeError("infer requires src_vocab, tgt_vocab, and src_tokenizer attributes.")

        device = next(self.parameters()).device
        src_stoi = getattr(self.src_vocab, "stoi", self.src_vocab)
        unk_idx = src_stoi.get("<unk>", 0)
        sos_idx = src_stoi.get("<sos>", 2)
        eos_idx = src_stoi.get("<eos>", 3)
        tokens = [tok.text.lower() for tok in self.src_tokenizer(src_sentence)]
        src_ids = [sos_idx] + [src_stoi.get(token, unk_idx) for token in tokens] + [eos_idx]
        src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
        src_mask = make_src_mask(src)

        ys = torch.tensor([[sos_idx]], dtype=torch.long, device=device)
        self.eval()
        with torch.no_grad():
            memory = self.encode(src, src_mask)
            for _ in range(100):
                tgt_mask = make_tgt_mask(ys)
                logits = self.decode(memory, src_mask, ys, tgt_mask)
                next_word = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                ys = torch.cat([ys, torch.tensor([[next_word]], dtype=torch.long, device=device)], dim=1)
                if next_word == eos_idx:
                    break

        out_tokens = []
        for idx in ys.squeeze(0).tolist():
            if hasattr(self.tgt_vocab, "lookup_token"):
                token = self.tgt_vocab.lookup_token(idx)
            else:
                token = self.tgt_vocab.itos[idx]
            if token in {"<sos>", "<pad>"}:
                continue
            if token == "<eos>":
                break
            out_tokens.append(token)
        return " ".join(out_tokens)
