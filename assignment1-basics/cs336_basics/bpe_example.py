import re

corpus_txt = """low low low low 
lower lower widest widest widest 
newest newest newest newest newest newest cafè"""

vocab = ["<|endoftext|>".encode("utf-8")]
vocab.extend(bytes([b]) for b in range(256))


# simple, via whitespaces not gpt 2 regex
def pretokenize(text: str) -> dict[tuple[bytes, ...], int]:
    txt_split = re.split(r"\s+", text)
    freq_table: dict[tuple[bytes, ...], int] = {}

    # for each string in the corpus
    for s in txt_split:
        # convert the string into bytes
        bytes_str = s.encode("utf-8")
        bytes_list: list[bytes] = [bytes([b]) for b in bytes_str]

        # for each bytes, we need to increment the counter
        bytes_tuple: tuple[bytes, ...] = tuple(bytes_list)
        if tuple(bytes_tuple) not in freq_table:
            freq_table[bytes_tuple] = 1
        else:
            freq_table[bytes_tuple] += 1

    return freq_table


def merge(freq_table: dict[tuple[bytes, ...], int], max_merges: int = 3):
    # merge max_merges times
    for _ in range(max_merges):
        # build pair_table
        pairs_table: dict[tuple[bytes, ...], int] = {}
        for k, v in freq_table.items():
            pairs: list[tuple[bytes, bytes]] = []
            for i in range(len(k) - 1):
                pairs.append((k[i], k[i + 1]))

            for p in pairs:
                if p not in pairs_table:
                    pairs_table[p] = v
                else:
                    pairs_table[p] += v

        # take the winning pair and merge it
        max_pair = max(pairs_table.items(), key=lambda pair: (pair[1], pair[0]))
        winning_bytes = max_pair[0]

        # find the winning bytes in the freq_table keys and whenever found, merge them
        new_table: dict[tuple[bytes, ...], int] = {}
        new_key: list[tuple[bytes, bytes]] = []
        for b_t in freq_table.keys():
            for i in range(len(b_t) - 1):
                if winning_bytes[0] == b_t[i] and winning_bytes[1] == b_t[i + 1]:
                    # replace the pair with winning bytes
                    new_key.append((winning_bytes[0], winning_bytes[1]))
            new_table[new_key] = max_pair[1]
        freq_table = new_table

    # add new tokens in vocabulary
    vocab.extend()
