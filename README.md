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
