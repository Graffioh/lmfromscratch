import pprint

from cs336_basics.bpe_tokenizer.tokenizer import Tokenizer

if __name__ == "__main__":
    tknzr = Tokenizer.from_files(
        "./cs336_basics/bpe_tokenizer/output_train_vocab.pkl",
        "./cs336_basics/bpe_tokenizer/output_train_merges.pkl",
        ["<|endoftext|>"],
    )
    # pprint.pprint(vars(tknzr))
    token_ids_from_encode = tknzr.encode("pizza")
    pprint.pprint(token_ids_from_encode)
