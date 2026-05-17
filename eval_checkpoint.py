"""
Simple evaluation script: load a saved checkpoint and compute corpus BLEU with beam search.
Run:
    python eval_checkpoint.py --checkpoint best_checkpoint.pt --beam 5
"""

import argparse
import torch
from torch.utils.data import DataLoader

from model import Transformer
from dataset import Multi30kDataset
from train import evaluate_bleu


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="best_checkpoint.pt")
    p.add_argument("--beam", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.7, help="length norm alpha")
    p.add_argument("--min_len", type=int, default=2, help="minimum target length during decoding")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load model (Transformer will load checkpoint if path exists)
    model = Transformer(checkpoint_path=args.checkpoint).to(device)

    # Build test dataset using attached vocabs
    test_dataset = Multi30kDataset("test", src_vocab=getattr(model, "src_vocab", None), tgt_vocab=getattr(model, "tgt_vocab", None))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=test_dataset.collate_fn)

    bleu = evaluate_bleu(
        model,
        test_loader,
        getattr(model, "tgt_vocab", None),
        device=device,
        beam_size=args.beam,
        length_norm_alpha=args.alpha,
        min_len=args.min_len,
    )
    print(f"Checkpoint: {args.checkpoint}  Beam: {args.beam}  Corpus BLEU: {bleu:.2f}")


if __name__ == "__main__":
    main()
