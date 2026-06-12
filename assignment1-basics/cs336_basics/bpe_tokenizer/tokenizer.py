import pickle
import pprint


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


tknzr = Tokenizer.from_files("./output_train_vocab.pkl", "./output_train_merges.pkl", ["<|endoftext|>"])
pprint.pprint(vars(tknzr))
