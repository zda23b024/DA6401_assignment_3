# DA6401 - Assignment 3: Implementing the Transformer for Machine Translation

## Overview

This project implements the Transformer architecture from "Attention Is All You Need" from scratch in PyTorch. The goal is German-to-English neural machine translation on the Multi30k dataset.

## Project Structure

```text
assignment3/
|-- requirements.txt
|-- README.md
|-- model.py           # Transformer architecture, attention, masks, and positional encoding
|-- lr_scheduler.py    # Noam learning-rate scheduler
|-- dataset.py         # Multi30k dataset loading and spaCy tokenization
|-- train.py           # Training, label smoothing, decoding, BLEU, and checkpoint utilities
```

## Main Components

- `model.py`: scaled dot-product attention, multi-head attention, sinusoidal positional encoding, encoder/decoder stacks, masks, and inference.
- `lr_scheduler.py`: Noam warmup plus inverse-square-root decay.
- `train.py`: label smoothing, epoch loop, greedy/beam decoding, BLEU evaluation, checkpoint save/load, and a W&B-ready training entry point.
- `dataset.py`: Multi30k loading and spaCy-based tokenization/vocabulary building.

## Gradescope Inference

`Transformer()` can be constructed with no arguments and exposes `infer(german_sentence)`.
For final submission, make the trained checkpoint available during construction by either:

- placing `checkpoint.pt` beside `model.py`, or
- setting `TRANSFORMER_CHECKPOINT_URL` to a public downloadable checkpoint URL, or
- setting `TRANSFORMER_CHECKPOINT_ID` to a public Google Drive file id.

The checkpoint should include `model_state_dict`, `model_config`, `src_vocab_itos`, and
`tgt_vocab_itos`; `train.py::save_checkpoint` already saves this format.

## W&B Experiments

Log in to W&B first:

```bash
wandb login
```

Run all report experiments with the default 10 epochs:

```bash
python train.py --experiment all
```

Or run the same training entry point with individual experiment names:

```bash
python train.py --experiment baseline
python train.py --experiment fixed_lr
python train.py --experiment no_scale
python train.py --experiment learned_pos
python train.py --experiment no_smoothing
```

W&B online logging is enabled by default. To run the final baseline longer for the
best checkpoint/BLEU score, use:

```bash
python train.py --experiment baseline --epochs 15
```
## W&B Experiment Report

Detailed experiment results, plots, and analysis can be found here:

[DA6401 Assignment 3 - Transformer Machine Translation on Multi30k](https://wandb.ai/zda23m016-iit-madras-zanzibar/da6401-a3/reports/DA6401-Assignment-3-Transformer-Machine-Translation-on-Multi30k--VmlldzoxNjkxMjE4Nw?accessToken=ktyjch8p5zoliatnmssa9cf2xeomiqtis33p1h8y4j25gybjkwapsgldr5g5tu2f)
