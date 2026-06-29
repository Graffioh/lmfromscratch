import regex

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

    # sample 10 docs from Tinystories and check compression ratio
    txt_chunks: list[str] = []
    file_path = (
        "/Users/ubreglia/Desktop/lmfromscratch/assignment1-basics/cs336_basics/data/TinyStoriesV2-GPT4-valid.txt"
    )
    with open(file_path) as f:
        # split docs by special tokens (e.g. EOT delimiter) and pretokenize
        special_token_escape_delimiter = regex.escape("<|endoftext|>")
        txt_split = regex.split(special_token_escape_delimiter, f.read())
        for txt_chunk in txt_split[:10]:
            txt_chunks.append(txt_chunk)

    encoded_tokens: list[int] = []
    for txt in txt_chunks:
        encoded_tokens += tknzr.encode(txt)

    compression_ratio = len("".join(txt_chunks)) / len(encoded_tokens)
    print("COMPRESSION RATIO 10 CHUNKS TINYSTORIES: ", compression_ratio)
