import pickle
from collections import defaultdict
from collections.abc import Iterable, Iterator

import regex

from .bpe_parallel import pretokenize


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab: dict[int, bytes] = vocab
        self.merges: list[tuple[bytes, bytes]] = merges
        self.inverse_vocab: defaultdict[bytes, int] = defaultdict()
        # reverse based on len: when special tokens overlap, we must prefer the longest sequence one during regex (to disambiguate)
        self.special_tokens: list[str] | None = sorted(special_tokens, key=len, reverse=True) if special_tokens else []

    @classmethod  # this is used so we can write Tokenizer.from_files
    def from_files(
        cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None
    ) -> "Tokenizer":
        vocab: dict[int, bytes] = {}
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str, pretokens_cache: dict[bytes, tuple[int, ...]] | None = None) -> list[int]:
        # build inverse vocab
        if len(self.inverse_vocab) == 0:
            for n, b in self.vocab.items():
                self.inverse_vocab[b] = n

        txt_split = [text]

        if self.special_tokens and len(self.special_tokens) != 0:
            # split in chunks based on special tokens
            delimited_special_tokens = "|".join(regex.escape(st) for st in self.special_tokens)
            capturing_group_pattern = f"({delimited_special_tokens})"
            txt_split = regex.split(capturing_group_pattern, text)

        token_ids: list[int] = []
        for txt_chunk in txt_split:
            if self.special_tokens and txt_chunk in self.special_tokens:
                token_ids.append(self.inverse_vocab[txt_chunk.encode("utf-8")])
            else:
                # gather pretokens
                for pretoken in pretokenize(txt_chunk):
                    pretoken_bytes: bytes = pretoken.encode("utf-8")
                    if pretokens_cache is not None and pretoken_bytes in pretokens_cache:
                        token_ids.extend(pretokens_cache[pretoken_bytes])
                        continue

                    p: list[bytes] = []
                    for pb in pretoken_bytes:
                        p.append(bytes([pb]))

                    # apply the pair of ordered merges to the pretokens
                    for b1, b2 in self.merges:
                        i = 0
                        new_token: list[bytes] = []
                        while i < len(p):
                            if i + 1 < len(p) and p[i] == b1 and p[i + 1] == b2:
                                new_token.append(p[i] + p[i + 1])
                                i += 2
                            else:
                                new_token.append(p[i])
                                i += 1
                        p = new_token

                    ids_for_this_pretoken = [self.inverse_vocab[b_t] for b_t in p]
                    if pretokens_cache is not None:
                        pretokens_cache[pretoken_bytes] = tuple(ids_for_this_pretoken)
                    token_ids.extend(ids_for_this_pretoken)

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for string in iterable:
            t_ids = self.encode(string)
            yield from t_ids

    def decode(self, ids: list[int]) -> str:
        bytes_list: list[bytes] = []
        for id in ids:
            if id not in self.vocab:
                bytes_list.append("�".encode())
            else:
                bytes_list.append(self.vocab[id])

        bytes_obj = b"".join(bytes_list)

        return bytes_obj.decode("utf-8", errors="replace")
