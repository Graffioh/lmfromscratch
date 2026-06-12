import pickle
from collections import defaultdict

from .bpe_parallel import pretokenize


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab: dict[int, bytes] = vocab
        self.merges: list[tuple[bytes, bytes]] = merges
        self.special_tokens: list[str] | None = special_tokens

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
        # reverse-index optimization taken from merge
        index_tuple_to_pretokens: defaultdict[tuple[bytes, ...], set[int]] = defaultdict(set[int])

        # gather pretokens
        pretokens: list[tuple[bytes, ...]] = []
        for pos, byte_tuple in enumerate(pretokenize(text)):
            pretokens.append(byte_tuple)
            index_tuple_to_pretokens[byte_tuple].add(pos)

        # merge pretokens in the order of training merges
        for b1, b2 in self.merges:
            for pos in index_tuple_to_pretokens[(b1, b2)]:
                i = 0
                pretokens_tuple = pretokens[pos]
                pretokens_len = len(pretokens_tuple)
                new_pretoken_tuple: list[bytes] = []
                while i < pretokens_len:
                    if i + 1 < pretokens_len and pretokens_tuple[i] == b1 and pretokens_tuple[i + 1] == b2:
                        new_pretoken_tuple.append(b1 + b2)
                        i += 2
                    else:
                        new_pretoken_tuple.append(pretokens_tuple[i])
                        i += 1
                pretokens[pos] = tuple(new_pretoken_tuple)

        # build token ids from the pretokens tuples and indexing in inverse vocab
        inv_vocab: defaultdict[bytes, int] = defaultdict()
        for n, b in self.vocab.items():
            inv_vocab[b] = n

        token_ids: list[int] = []
        for pretokens_tuple in pretokens:
            for b_t in pretokens_tuple:
                token_ids.append(inv_vocab[b_t])

        return token_ids
