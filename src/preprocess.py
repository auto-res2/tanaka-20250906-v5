"""
preprocess.py – data streaming + tokenisation utilities
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import sentencepiece as spm
import torch
from datasets import load_dataset, disable_caching
from torch.utils.data import IterableDataset, DataLoader
from transformers import PreTrainedTokenizerFast


disable_caching()  # no local HF cache – always streaming


class SlimPajamaStream(IterableDataset):
    """Streams raw text from SlimPajama (train split)."""

    def __init__(self, split: str = "train"):
        super().__init__()
        self.ds = load_dataset("cerebras/SlimPajama-627B", split=split, streaming=True)

    def __iter__(self):
        # HF streaming datasets are internally sharded by worker_id already
        for sample in self.ds:
            yield sample["text"]


# ------------------------------------------------------------------
# Tokeniser helpers
# ------------------------------------------------------------------

def load_tokenizer(sp_path: Path) -> PreTrainedTokenizerFast:  # noqa: D401 – util
    if not sp_path.exists():
        raise FileNotFoundError(
            f"SentencePiece model expected at {sp_path}. Provide a 32k-vocab .model file."
        )
    tok = PreTrainedTokenizerFast(
        tokenizer_object=spm.SentencePieceProcessor(model_file=str(sp_path)),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
    )
    return tok


class PackedTokenDataset(IterableDataset):
    """Packs variable-length documents into fixed-length token windows."""

    def __init__(self, raw_iter: Iterable[str], tokenizer: PreTrainedTokenizerFast, max_seq_len: int):
        super().__init__()
        self.raw_iter = raw_iter
        self.tok = tokenizer
        self.max_seq_len = max_seq_len

    def __iter__(self):
        buf = []
        for txt in self.raw_iter:
            ids = self.tok.encode(txt, add_special_tokens=False)
            buf.extend(ids + [self.tok.eos_token_id])
            while len(buf) >= self.max_seq_len:
                yield torch.tensor(buf[: self.max_seq_len], dtype=torch.long)
                buf = buf[self.max_seq_len :]


# ------------------------------------------------------------------
# Data-loader (PyTorch) wrapper
# ------------------------------------------------------------------

def build_dataloader(max_seq_len: int, batch_size_per_gpu: int, tokenizer_path: Path) -> DataLoader:  # noqa: D401
    tokenizer = load_tokenizer(tokenizer_path)
    raw_stream = SlimPajamaStream("train")
    packed = PackedTokenDataset(raw_stream, tokenizer, max_seq_len)
    return DataLoader(packed, batch_size=batch_size_per_gpu, num_workers=4, pin_memory=True)
