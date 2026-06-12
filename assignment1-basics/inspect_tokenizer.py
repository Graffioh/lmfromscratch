import pprint

from cs336_basics.bpe_tokenizer.tokenizer import Tokenizer

if __name__ == "__main__":
    tknzr = Tokenizer.from_files(
        "./outputs/output_train_vocab.pkl",
        "./outputs/output_train_merges.pkl",
        ["<|endoftext|>"],
    )

    # pprint.pprint(vars(tknzr))
    token_ids_from_encode = tknzr.encode("hello lower")
    pprint.pprint(token_ids_from_encode)
