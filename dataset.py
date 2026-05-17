from collections import Counter
from typing import Iterable

import torch
from torch.nn.utils.rnn import pad_sequence


SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]


class Vocab:
    """Small vocabulary helper with torchtext-like methods used by this assignment."""

    def __init__(self, tokens: Iterable[str], min_freq: int = 2) -> None:
        counter = Counter(tokens)
        self.itos = list(SPECIAL_TOKENS)
        for token, freq in counter.items():
            if freq >= min_freq and token not in self.itos:
                self.itos.append(token)
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.stoi["<unk>"])

    def lookup_token(self, idx: int) -> str:
        return self.itos[idx]

    def lookup_indices(self, tokens: Iterable[str]) -> list[int]:
        return [self[token] for token in tokens]


def _load_spacy_model(name: str, lang: str):
    import spacy

    try:
        return spacy.load(name)
    except OSError:
        return spacy.blank(lang)


class Multi30kDataset(torch.utils.data.Dataset):
    def __init__(self, split="train", src_vocab: Vocab | None = None, tgt_vocab: Vocab | None = None, min_freq: int = 1):
        """
        Loads the Multi30k dataset and prepares German/English tokenizers.
        """
        from datasets import load_dataset

        self.split = split
        self.src_tokenizer = _load_spacy_model("de_core_news_sm", "de")
        self.tgt_tokenizer = _load_spacy_model("en_core_web_sm", "en")
        self.raw_data = load_dataset("bentrevett/multi30k", split=split)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.min_freq = min_freq
        if self.src_vocab is None or self.tgt_vocab is None:
            self.build_vocab()
        self.data = self.process_data()

    @property
    def pad_idx(self) -> int:
        return self.src_vocab.stoi["<pad>"]

    def _get_pair(self, example) -> tuple[str, str]:
        if "de" in example and "en" in example:
            return example["de"], example["en"]
        if "translation" in example:
            translation = example["translation"]
            return translation["de"], translation["en"]
        raise KeyError("Expected Multi30k example to contain de/en text.")

    def _tokenize_src(self, text: str) -> list[str]:
        return [tok.text.lower() for tok in self.src_tokenizer(text)]

    def _tokenize_tgt(self, text: str) -> list[str]:
        return [tok.text.lower() for tok in self.tgt_tokenizer(text)]

    def build_vocab(self):
        """
        Builds vocabulary mappings for src (de) and tgt (en), including special tokens.
        """
        src_tokens = []
        tgt_tokens = []
        for example in self.raw_data:
            src_text, tgt_text = self._get_pair(example)
            src_tokens.extend(self._tokenize_src(src_text))
            tgt_tokens.extend(self._tokenize_tgt(tgt_text))
        if self.src_vocab is None:
            self.src_vocab = Vocab(src_tokens, min_freq=self.min_freq)
        if self.tgt_vocab is None:
            self.tgt_vocab = Vocab(tgt_tokens, min_freq=self.min_freq)
        return self.src_vocab, self.tgt_vocab

    def process_data(self):
        """
        Convert English and German sentences into integer token lists.
        """
        processed = []
        src_sos = self.src_vocab.stoi["<sos>"]
        src_eos = self.src_vocab.stoi["<eos>"]
        tgt_sos = self.tgt_vocab.stoi["<sos>"]
        tgt_eos = self.tgt_vocab.stoi["<eos>"]
        for example in self.raw_data:
            src_text, tgt_text = self._get_pair(example)
            src_ids = [src_sos] + self.src_vocab.lookup_indices(self._tokenize_src(src_text)) + [src_eos]
            tgt_ids = [tgt_sos] + self.tgt_vocab.lookup_indices(self._tokenize_tgt(tgt_text)) + [tgt_eos]
            processed.append((torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)))
        return processed

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    def collate_fn(self, batch):
        src_batch, tgt_batch = zip(*batch)
        src = pad_sequence(src_batch, batch_first=True, padding_value=self.src_vocab.stoi["<pad>"])
        tgt = pad_sequence(tgt_batch, batch_first=True, padding_value=self.tgt_vocab.stoi["<pad>"])
        return src, tgt
