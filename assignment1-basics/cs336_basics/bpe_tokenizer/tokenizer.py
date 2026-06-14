import pickle
from collections import defaultdict
from collections.abc import Iterable, Iterator

from .bpe_parallel import pretokenize


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab: dict[int, bytes] = vocab
        self.merges: list[tuple[bytes, bytes]] = merges
        self.special_tokens: list[str] | None = special_tokens
        self.inverse_vocab: defaultdict[bytes, int] = defaultdict()

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

    def encode(self, text: str) -> list[int]:
        # gather pretokens
        pretokens: list[tuple[bytes, ...]] = []
        for pretoken in pretokenize(text):
            pretoken_bytes: bytes = pretoken.encode("utf-8")
            pretoken_tuples: list[bytes] = []

            for pb in pretoken_bytes:
                pretoken_tuples.append(bytes([pb]))

            pretokens.append(tuple(pretoken_tuples))

        # apply the pair of ordered merges to the pretokens
        for b1, b2 in self.merges:
            for pos, p in enumerate(pretokens):
                i = 0
                new_token: list[bytes] = []
                while i < len(p):
                    if i + 1 < len(p) and p[i] == b1 and p[i + 1] == b2:
                        new_token.append(p[i] + p[i + 1])
                        i += 2
                    else:
                        new_token.append(p[i])
                        i += 1
                pretokens[pos] = tuple(new_token)

        # build token ids from the pretokens tuples and indexing in inverse vocab
        if len(self.inverse_vocab) == 0:
            for n, b in self.vocab.items():
                self.inverse_vocab[b] = n

        token_ids: list[int] = []
        for pretokens_tuple in pretokens:
            for b_t in pretokens_tuple:
                token_ids.append(self.inverse_vocab[b_t])

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        text: list[str] = []
        for id in ids:
            if id not in self.vocab:
                text.append(b"U+FFFD".decode("utf-8"))
            else:
                text.append(self.vocab[id].decode("utf-8"))

        return "".join(text)
