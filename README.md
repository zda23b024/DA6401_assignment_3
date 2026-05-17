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

Run the same training entry point with different experiment names:

```bash
python train.py --experiment baseline --epochs 15
python train.py --experiment fixed_lr --epochs 15
python train.py --experiment no_scale --epochs 15
python train.py --experiment learned_pos --epochs 15
python train.py --experiment no_smoothing --epochs 15
```

Set `WANDB_MODE=online` before running to log report plots.
