from __future__ import annotations

import unicodedata

import numpy as np


def _is_whitespace(ch: str) -> bool:
    if ch in (" ", "\t", "\n", "\r"):
        return True
    return unicodedata.category(ch) == "Zs"


def _is_control(ch: str) -> bool:
    if ch in ("\t", "\n", "\r"):
        return False
    return unicodedata.category(ch).startswith("C")


def _is_punctuation(ch: str) -> bool:
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


def _is_chinese_char(cp: int) -> bool:
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF
        or 0x2A700 <= cp <= 0x2B73F
        or 0x2B740 <= cp <= 0x2B81F
        or 0x2B820 <= cp <= 0x2CEAF
        or 0xF900 <= cp <= 0xFAFF
        or 0x2F800 <= cp <= 0x2FA1F
    )


class WordPieceTokenizer:
    def __init__(self, vocab_path, max_length: int = 256, max_chars_per_word: int = 100):
        with open(vocab_path, encoding="utf-8") as handle:
            self.vocab = {line.rstrip("\n"): i for i, line in enumerate(handle)}
        self.max_length = max_length
        self.max_chars_per_word = max_chars_per_word
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.unk_id = self.vocab["[UNK]"]
        self.pad_id = self.vocab["[PAD]"]

    def _clean(self, text: str) -> str:
        out = []
        for ch in text:
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD or _is_control(ch):
                continue
            out.append(" " if _is_whitespace(ch) else ch)
        return "".join(out)

    def _pad_chinese(self, text: str) -> str:
        return "".join(f" {ch} " if _is_chinese_char(ord(ch)) else ch for ch in text)

    @staticmethod
    def _strip_accents(text: str) -> str:
        return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")

    @staticmethod
    def _split_punctuation(token: str) -> list[str]:
        pieces: list[list[str]] = []
        start_new = True
        for ch in token:
            if _is_punctuation(ch):
                pieces.append([ch])
                start_new = True
            else:
                if start_new:
                    pieces.append([])
                    start_new = False
                pieces[-1].append(ch)
        return ["".join(p) for p in pieces]

    def basic_tokens(self, text: str) -> list[str]:
        text = self._pad_chinese(self._clean(text))
        tokens = []
        for token in text.strip().split():
            token = self._strip_accents(token.lower())
            tokens.extend(self._split_punctuation(token))
        return [t for t in " ".join(tokens).strip().split()]

    def wordpiece(self, token: str) -> list[int]:
        if len(token) > self.max_chars_per_word:
            return [self.unk_id]
        ids = []
        start = 0
        while start < len(token):
            end = len(token)
            current = None
            while start < end:
                piece = token[start:end]
                if start > 0:
                    piece = "##" + piece
                if piece in self.vocab:
                    current = self.vocab[piece]
                    break
                end -= 1
            if current is None:
                return [self.unk_id]
            ids.append(current)
            start = end
        return ids

    def encode(self, text: str) -> list[int]:
        ids = []
        for token in self.basic_tokens(text):
            ids.extend(self.wordpiece(token))
        ids = ids[: self.max_length - 2]
        return [self.cls_id] + ids + [self.sep_id]

    def batch(self, texts: list[str]):
        encoded = [self.encode(t) for t in texts]
        width = max(len(e) for e in encoded)
        input_ids = np.full((len(encoded), width), self.pad_id, dtype=np.int64)
        attention = np.zeros((len(encoded), width), dtype=np.int64)
        for row, ids in enumerate(encoded):
            input_ids[row, : len(ids)] = ids
            attention[row, : len(ids)] = 1
        token_types = np.zeros_like(input_ids)
        return input_ids, attention, token_types
