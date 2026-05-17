"""
train.py - Training, inference, BLEU, and checkpoint utilities.

The public signatures in this file are kept stable for the autograder.
"""

import math
import os
import argparse
from collections import Counter
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import Transformer, make_src_mask, make_tgt_mask


EXPERIMENT_CONFIGS = {
    "baseline": {},
    "fixed_lr": {"use_noam": False, "optimizer_lr": 1e-4},
    "no_scale": {"use_attention_scaling": False},
    "learned_pos": {"positional_encoding_type": "learned"},
    "no_smoothing": {"smoothing": 0.0},
}


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
    wandb_run=None,
    log_grad_norm_steps: int = 0,
    log_prediction_confidence: bool = False,
) -> float:
    """
    Run one epoch of training or evaluation.
    """
    model.train(is_train)
    total_loss = 0.0
    total_tokens = 0
    total_correct_confidence = 0.0
    confidence_batches = 0
    correct_tokens = 0
    accuracy_tokens = 0
    global_step = int(getattr(model, "_global_step", 0))

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
            with torch.no_grad():
                non_pad_accuracy = tgt_y != getattr(loss_fn, "pad_idx", 1)
                predictions = logits.argmax(dim=-1)
                correct_tokens += ((predictions == tgt_y) & non_pad_accuracy).sum().item()
                accuracy_tokens += non_pad_accuracy.sum().item()

            if log_prediction_confidence:
                with torch.no_grad():
                    probs = F.softmax(logits, dim=-1)
                    correct_probs = probs.gather(-1, tgt_y.unsqueeze(-1)).squeeze(-1)
                    non_pad_conf = tgt_y != getattr(loss_fn, "pad_idx", 1)
                    if non_pad_conf.any():
                        total_correct_confidence += correct_probs[non_pad_conf].mean().item()
                        confidence_batches += 1

            if is_train:
                if optimizer is None:
                    raise ValueError("optimizer is required when is_train=True")
                optimizer.zero_grad()
                loss.backward()
                global_step += 1
                if wandb_run is not None and global_step <= log_grad_norm_steps:
                    wandb_run.log(
                        {
                            "step": global_step,
                            "grad_norm/q_weight": _grad_norm(model.encoder.layers[0].self_attn.w_q.weight),
                            "grad_norm/k_weight": _grad_norm(model.encoder.layers[0].self_attn.w_k.weight),
                        },
                        step=global_step,
                    )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "step": global_step,
                            "batch_train_loss": loss.item(),
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        },
                        step=global_step,
                    )

        non_pad = (tgt_y != getattr(loss_fn, "pad_idx", 1)).sum().item()
        batch_tokens = max(non_pad, 1)
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

    model._global_step = global_step
    if wandb_run is not None and log_prediction_confidence and confidence_batches:
        split = "train" if is_train else "val"
        metrics = {
            f"{split}_prediction_confidence": total_correct_confidence / confidence_batches,
            "epoch": epoch_num,
        }
        if accuracy_tokens:
            metrics[f"{split}_token_accuracy"] = correct_tokens / accuracy_tokens
        wandb_run.log(metrics)
    return total_loss / max(total_tokens, 1)


def _grad_norm(param: torch.Tensor) -> float:
    if param.grad is None:
        return 0.0
    return float(param.grad.detach().norm(2).item())


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


def beam_search_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
    beam_size: int = 8,
    length_norm_alpha: float = 0.6,
    min_len: int = 3,
    unk_symbol: Optional[int] = None,
    pad_symbol: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate a translation using beam search to improve BLEU quality.
    """
    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)
    with torch.no_grad():
        memory = model.encode(src, src_mask)

        # Derive unk/pad indices from attached vocab if not provided
        if unk_symbol is None:
            try:
                tgt_vocab = getattr(model, "tgt_vocab", None)
                if tgt_vocab is None:
                    unk_symbol = 0
                elif hasattr(tgt_vocab, "stoi"):
                    unk_symbol = tgt_vocab.stoi.get("<unk>", 0)
                elif isinstance(tgt_vocab, dict):
                    unk_symbol = tgt_vocab.get("<unk>", 0)
                else:
                    unk_symbol = 0
            except Exception:
                unk_symbol = 0

        if pad_symbol is None:
            try:
                tgt_vocab = getattr(model, "tgt_vocab", None)
                if tgt_vocab is None:
                    pad_symbol = 1
                elif hasattr(tgt_vocab, "stoi"):
                    pad_symbol = tgt_vocab.stoi.get("<pad>", 1)
                elif isinstance(tgt_vocab, dict):
                    pad_symbol = tgt_vocab.get("<pad>", 1)
                else:
                    pad_symbol = 1
            except Exception:
                pad_symbol = 1

        beam = [([start_symbol], 0.0, False)]

        def creates_repeated_ngram(seq: list[int], next_idx: int, n: int = 3) -> bool:
            if len(seq) + 1 < 2 * n:
                return False
            candidate = seq + [next_idx]
            ngram = tuple(candidate[-n:])
            history = {tuple(candidate[i : i + n]) for i in range(len(candidate) - n)}
            return ngram in history

        for _ in range(max_len - 1):
            candidates = []
            for seq, score, finished in beam:
                if finished:
                    candidates.append((seq, score, True))
                    continue

                ys = torch.tensor([seq], dtype=torch.long, device=device)
                tgt_mask = make_tgt_mask(ys).to(device)
                logits = model.decode(memory, src_mask, ys, tgt_mask)
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)
                for blocked_idx in {pad_symbol, unk_symbol, start_symbol}:
                    if 0 <= blocked_idx < log_probs.size(-1):
                        log_probs[blocked_idx] = -float("inf")
                if len(seq) - 1 < min_len and 0 <= end_symbol < log_probs.size(-1):
                    log_probs[end_symbol] = -float("inf")

                topk_log_probs, topk_indices = log_probs.topk(min(beam_size, log_probs.size(-1)))

                for log_prob, idx in zip(topk_log_probs.tolist(), topk_indices.tolist()):
                    if not math.isfinite(log_prob) or creates_repeated_ngram(seq, idx):
                        continue
                    new_seq = seq + [idx]
                    new_score = score + log_prob
                    candidates.append((new_seq, new_score, idx == end_symbol))

            if not candidates:
                break

            def score_with_length_norm(item):
                seq, score, finished = item
                length = len(seq)
                norm = ((5.0 + length) / 6.0) ** length_norm_alpha
                return score / norm

            beam = sorted(candidates, key=score_with_length_norm, reverse=True)[:beam_size]
            if all(finished for _, _, finished in beam):
                break

        completed_beams = [item for item in beam if item[2]]
        best_seq = max(completed_beams or beam, key=score_with_length_norm)[0]

    return torch.tensor([best_seq], dtype=torch.long, device=device)


def _lookup_token(vocab, idx: int) -> str:
    if hasattr(vocab, "lookup_token"):
        return vocab.lookup_token(idx)
    if hasattr(vocab, "itos"):
        return vocab.itos[idx]
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
    beam_size: int = 8,
    length_norm_alpha: float = 0.6,
    min_len: int = 3,
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
            references.append(_tokens_from_ids(tgt_i, tgt_vocab))
            src_mask = make_src_mask(src_i)
            if beam_size > 1:
                pred = beam_search_decode(
                    model,
                    src_i,
                    src_mask,
                    max_len,
                    start_symbol,
                    end_symbol,
                    device=device,
                    beam_size=beam_size,
                    length_norm_alpha=length_norm_alpha,
                    min_len=min_len,
                )
            else:
                pred = greedy_decode(model, src_i, src_mask, max_len, start_symbol, end_symbol, device=device)
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


def _build_config(args: argparse.Namespace) -> dict:
    config = {
        "experiment_name": args.experiment,
        "batch_size": 64,
        "num_epochs": args.epochs,
        "d_model": 256,
        "N": 3,
        "num_heads": 4,
        "d_ff": 1024,
        "dropout": 0.1,
        "warmup_steps": 2000,
        "optimizer_lr": 1.0,
        "smoothing": 0.1,
        "val_bleu_beam_size": 1,
        "use_noam": True,
        "use_attention_scaling": True,
        "positional_encoding_type": "sinusoidal",
        "log_grad_norm_steps": 1000,
        "log_prediction_confidence": True,
        "log_attention_maps": True,
    }
    config.update(EXPERIMENT_CONFIGS[args.experiment])
    if args.epochs is not None:
        config["num_epochs"] = args.epochs
    return config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Transformer experiments for DA6401 Assignment 3.")
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENT_CONFIGS.keys()),
        default="baseline",
        help="W&B experiment/ablation to run.",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Epochs for this experiment run.")
    parser.add_argument("--project", type=str, default="da6401-a3", help="W&B project name.")
    return parser.parse_args()


def log_attention_heatmaps(model: Transformer, dataset, device: str, wandb_run, epoch: int) -> None:
    if wandb_run is None:
        return

    try:
        import matplotlib.pyplot as plt
        import wandb
    except Exception:
        return

    model.eval()
    src, _ = dataset[0]
    src = src.unsqueeze(0).to(device)
    src_mask = make_src_mask(src)

    with torch.no_grad():
        model.encode(src, src_mask)

    attn = model.encoder.layers[-1].self_attn.attn_weights
    if attn is None:
        return

    attn = attn[0].detach().cpu()
    token_ids = src.squeeze(0).detach().cpu().tolist()
    tokens = [_lookup_token(dataset.src_vocab, idx) for idx in token_ids]
    num_heads = attn.size(0)
    cols = min(4, num_heads)
    rows = math.ceil(num_heads / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)

    for head in range(num_heads):
        ax = axes[head // cols][head % cols]
        image = ax.imshow(attn[head].numpy(), aspect="auto", cmap="viridis")
        ax.set_title(f"Head {head}")
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=7)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=7)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(num_heads, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.tight_layout()
    wandb_run.log({"attention_maps/last_encoder_layer": wandb.Image(fig), "epoch": epoch})
    plt.close(fig)


def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.
    """
    args = _parse_args()

    import wandb
    from dataset import Multi30kDataset
    from lr_scheduler import NoamScheduler

    config = _build_config(args)
    run = wandb.init(
        project=args.project,
        name=config["experiment_name"],
        config=config,
        mode=os.environ.get("WANDB_MODE", "disabled"),
    )
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
        checkpoint_path="",
        use_attention_scaling=cfg.use_attention_scaling,
        positional_encoding_type=cfg.positional_encoding_type,
    ).to(device)
    model.src_vocab = train_dataset.src_vocab
    model.tgt_vocab = train_dataset.tgt_vocab
    model.src_tokenizer = train_dataset.src_tokenizer

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer_lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = None
    if cfg.use_noam:
        scheduler = NoamScheduler(optimizer, d_model=cfg.d_model, warmup_steps=cfg.warmup_steps)
    loss_fn = LabelSmoothingLoss(len(train_dataset.tgt_vocab), train_dataset.tgt_vocab.stoi["<pad>"], cfg.smoothing)

    best_val_loss = float("inf")
    best_val_bleu = -float("inf")
    for epoch in range(cfg.num_epochs):
        train_loss = run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            epoch,
            True,
            device,
            wandb_run=run,
            log_grad_norm_steps=cfg.log_grad_norm_steps,
            log_prediction_confidence=cfg.log_prediction_confidence,
        )
        val_loss = run_epoch(
            val_loader,
            model,
            loss_fn,
            None,
            None,
            epoch,
            False,
            device,
            wandb_run=run,
            log_prediction_confidence=cfg.log_prediction_confidence,
        )
        val_bleu = evaluate_bleu(
            model,
            val_loader,
            train_dataset.tgt_vocab,
            device=device,
            beam_size=cfg.val_bleu_beam_size,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, f"{cfg.experiment_name}_best_loss_checkpoint.pt")
            if cfg.experiment_name == "baseline":
                save_checkpoint(model, optimizer, scheduler, epoch, "best_loss_checkpoint.pt")
        if val_bleu > best_val_bleu:
            best_val_bleu = val_bleu
            save_checkpoint(model, optimizer, scheduler, epoch, f"{cfg.experiment_name}_checkpoint.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, f"{cfg.experiment_name}_best_checkpoint.pt")
            if cfg.experiment_name == "baseline":
                save_checkpoint(model, optimizer, scheduler, epoch, "checkpoint.pt")
                save_checkpoint(model, optimizer, scheduler, epoch, "best_checkpoint.pt")
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_bleu": val_bleu,
                "best_val_loss": best_val_loss,
                "best_val_bleu": best_val_bleu,
            }
        )
        print(
            f"Epoch {epoch+1}/{cfg.num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_bleu={val_bleu:.2f}, best_val_bleu={best_val_bleu:.2f}",
            flush=True,
        )
        if cfg.log_attention_maps and epoch == cfg.num_epochs - 1:
            log_attention_heatmaps(model, val_dataset, device, run, epoch)

    load_checkpoint(f"{cfg.experiment_name}_best_checkpoint.pt", model)
    bleu = evaluate_bleu(model, test_loader, train_dataset.tgt_vocab, device, beam_size=cfg.val_bleu_beam_size)
    wandb.log({"test_bleu": bleu})
    print(f"Final test BLEU: {bleu:.2f}", flush=True)
    run.finish()


if __name__ == "__main__":
    run_training_experiment()
