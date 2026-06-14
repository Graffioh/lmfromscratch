import pprint

from cs336_basics.bpe_tokenizer.tokenizer import Tokenizer

if __name__ == "__main__":
    tknzr = Tokenizer.from_files(
        "./outputs/output_train_vocab.pkl",
        "./outputs/output_train_merges.pkl",
        ["<|endoftext|>"],
    )

    # pprint.pprint(vars(tknzr))
    token_ids_from_encode = tknzr.encode("hello hello hello")
    print("ENCODE: ", token_ids_from_encode)

    text_from_token_ids = tknzr.decode(token_ids_from_encode)
    print("DECODE: ", text_from_token_ids)
