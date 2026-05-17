"""
train.py - Training, inference, BLEU, and checkpoint utilities.

The public signatures in this file are kept stable for the autograder.
"""

import math
import os
from collections import Counter
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import Transformer, make_src_mask, make_tgt_mask


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need".
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            denom = max(self.vocab_size - 2, 1)
            true_dist.fill_(self.smoothing / denom)
            true_dist[:, self.pad_idx] = 0
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            pad_mask = target == self.pad_idx
            true_dist[pad_mask] = 0

        loss = -(true_dist * log_probs).sum(dim=1)
        non_pad = target != self.pad_idx
        if non_pad.sum() == 0:
            return loss.sum()
        return loss[non_pad].mean()


def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.
    """
    del epoch_num
    model.train(is_train)
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in data_iter:
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_y = tgt[:, 1:]

        src_mask = make_src_mask(src)
        tgt_mask = make_tgt_mask(tgt_input)

        with torch.set_grad_enabled(is_train):
            logits = model(src, tgt_input, src_mask, tgt_mask)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_y.reshape(-1))

            if is_train:
                if optimizer is None:
                    raise ValueError("optimizer is required when is_train=True")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        non_pad = (tgt_y != getattr(loss_fn, "pad_idx", 1)).sum().item()
        batch_tokens = max(non_pad, 1)
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

    return total_loss / max(total_tokens, 1)


def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.
    """
    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)
    ys = torch.full((src.size(0), 1), start_symbol, dtype=torch.long, device=device)

    with torch.no_grad():
        memory = model.encode(src, src_mask)
        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys).to(device)
            logits = model.decode(memory, src_mask, ys, tgt_mask)
            next_word = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            ys = torch.cat([ys, next_word], dim=1)
            if src.size(0) == 1 and int(next_word.item()) == end_symbol:
                break
            if src.size(0) > 1 and torch.all(next_word.squeeze(1) == end_symbol):
                break
    return ys


def _lookup_token(vocab, idx: int) -> str:
    if hasattr(vocab, "lookup_token"):
        return vocab.lookup_token(idx)
    if hasattr(vocab, "itos"):
        return vocab.itos[idx]
    raise TypeError("tgt_vocab must support lookup_token(idx) or have an itos list")


def _vocab_index(vocab, token: str, default: int) -> int:
    if hasattr(vocab, "stoi"):
        return vocab.stoi.get(token, default)
    if isinstance(vocab, dict):
        return vocab.get(token, default)
    return default


def _tokens_from_ids(ids: list[int], vocab) -> list[str]:
    tokens = []
    for idx in ids:
        token = _lookup_token(vocab, idx)
        if token in {"<sos>", "<pad>"}:
            continue
        if token == "<eos>":
            break
        tokens.append(token)
    return tokens


def _corpus_bleu(references: list[list[str]], hypotheses: list[list[str]], max_n: int = 4) -> float:
    if not hypotheses:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        matches = 0
        total = 0
        for ref, hyp in zip(references, hypotheses):
            ref_counts = Counter(tuple(ref[i : i + n]) for i in range(max(len(ref) - n + 1, 0)))
            hyp_counts = Counter(tuple(hyp[i : i + n]) for i in range(max(len(hyp) - n + 1, 0)))
            matches += sum(min(count, ref_counts[gram]) for gram, count in hyp_counts.items())
            total += sum(hyp_counts.values())
        precisions.append((matches + 1e-9) / (total + 1e-9))

    ref_len = sum(len(ref) for ref in references)
    hyp_len = sum(len(hyp) for hyp in hypotheses)
    if hyp_len == 0:
        return 0.0
    brevity_penalty = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return 100.0 * brevity_penalty * math.exp(sum(math.log(p) for p in precisions) / max_n)


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.
    """
    start_symbol = _vocab_index(tgt_vocab, "<sos>", 2)
    end_symbol = _vocab_index(tgt_vocab, "<eos>", 3)
    references = []
    hypotheses = []

    model.eval()
    for src, tgt in test_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        for i in range(src.size(0)):
            src_i = src[i : i + 1]
            tgt_i = tgt[i].tolist()
            src_mask = make_src_mask(src_i)
            pred = greedy_decode(model, src_i, src_mask, max_len, start_symbol, end_symbol, device=device)
            references.append(_tokens_from_ids(tgt_i, tgt_vocab))
            hypotheses.append(_tokens_from_ids(pred.squeeze(0).tolist(), tgt_vocab))

    return float(_corpus_bleu(references, hypotheses))


def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimizer + scheduler state to disk.
    """
    src_vocab_itos = getattr(getattr(model, "src_vocab", None), "itos", None)
    tgt_vocab_itos = getattr(getattr(model, "tgt_vocab", None), "itos", None)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "model_config": getattr(model, "model_config", {}),
            "src_vocab_itos": src_vocab_itos,
            "tgt_vocab_itos": tgt_vocab_itos,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model and optionally optimizer/scheduler state from disk.
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if checkpoint.get("src_vocab_itos") is not None:
        from model import _LoadedVocab

        model.src_vocab = _LoadedVocab(checkpoint["src_vocab_itos"])
    if checkpoint.get("tgt_vocab_itos") is not None:
        from model import _LoadedVocab

        model.tgt_vocab = _LoadedVocab(checkpoint["tgt_vocab_itos"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint.get("epoch", 0))


def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.
    """
    import wandb
    from dataset import Multi30kDataset
    from lr_scheduler import NoamScheduler

    config = {
        "batch_size": 32,
        "num_epochs": 10,
        "d_model": 512,
        "N": 6,
        "num_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1,
        "warmup_steps": 4000,
        "lr": 1.0,
        "smoothing": 0.1,
    }
    run = wandb.init(project="da6401-a3", config=config, mode=os.environ.get("WANDB_MODE", "disabled"))
    cfg = run.config
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = Multi30kDataset("train")
    val_dataset = Multi30kDataset("validation", src_vocab=train_dataset.src_vocab, tgt_vocab=train_dataset.tgt_vocab)
    test_dataset = Multi30kDataset("test", src_vocab=train_dataset.src_vocab, tgt_vocab=train_dataset.tgt_vocab)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=test_dataset.collate_fn)

    model = Transformer(
        src_vocab_size=len(train_dataset.src_vocab),
        tgt_vocab_size=len(train_dataset.tgt_vocab),
        d_model=cfg.d_model,
        N=cfg.N,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    ).to(device)
    model.src_vocab = train_dataset.src_vocab
    model.tgt_vocab = train_dataset.tgt_vocab
    model.src_tokenizer = train_dataset.src_tokenizer

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=cfg.d_model, warmup_steps=cfg.warmup_steps)
    loss_fn = LabelSmoothingLoss(len(train_dataset.tgt_vocab), train_dataset.tgt_vocab.stoi["<pad>"], cfg.smoothing)

    for epoch in range(cfg.num_epochs):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler, epoch, True, device)
        val_loss = run_epoch(val_loader, model, loss_fn, None, None, epoch, False, device)
        save_checkpoint(model, optimizer, scheduler, epoch)
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    bleu = evaluate_bleu(model, test_loader, train_dataset.tgt_vocab, device)
    wandb.log({"test_bleu": bleu})
    run.finish()


if __name__ == "__main__":
    run_training_experiment()
