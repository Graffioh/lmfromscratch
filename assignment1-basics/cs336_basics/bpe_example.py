import re
from collections import Counter, defaultdict

corpus_txt = """low low low low 
lower lower widest widest widest 
newest newest newest newest newest newest cafè"""

# build base vocab of 256 bytes + EOT token
vocab: defaultdict[int, bytes] = defaultdict(bytes)
vocab[0] = "<|endoftext|>".encode("utf-8")
for b in range(256):
    vocab[b + 1] = bytes([b])


# simple, via whitespaces not gpt 2 regex
def pretokenize(text: str) -> dict[tuple[bytes, ...], int]:
    txt_split = re.split(r"\s+", text)
    freq_table: Counter[tuple[bytes, ...]] = Counter()

    # for each string in the corpus
    for s in txt_split:
        # convert the string into bytes
        bytes_str = s.encode("utf-8")
        bytes_list: list[bytes] = [bytes([b]) for b in bytes_str]

        # for each bytes, we need to increment the counter
        bytes_tuple: tuple[bytes, ...] = tuple(bytes_list)
        freq_table[bytes_tuple] += 1

    return freq_table


def merge(
    freq_table: dict[tuple[bytes, ...], int], max_merges: int = 3
) -> tuple[list[tuple[bytes, bytes]], list[bytes]]:
    new_vocab_words: list[bytes] = []
    merges: list[tuple[bytes, bytes]] = []

    # merge max_merges times
    for _ in range(max_merges):
        # build pair_table
        pairs_table: defaultdict[tuple[bytes, ...], int] = defaultdict(int)
        for k, v in freq_table.items():
            pairs: list[tuple[bytes, bytes]] = []
            for i in range(len(k) - 1):
                pairs.append((k[i], k[i + 1]))

            for p in pairs:
                pairs_table[p] += v

        # take the winning pair and merge it
        max_pair = max(pairs_table.items(), key=lambda pair: (pair[1], pair[0]))
        winning_bytes = max_pair[0]
        merged_winning_bytes = winning_bytes[0] + winning_bytes[1]

        # find the winning bytes in the freq_table keys and whenever found, merge them
        new_table: defaultdict[tuple[bytes, ...], int] = defaultdict(int)
        for b_t_key, b_t_count in freq_table.items():
            new_key: list[bytes] = []
            i = 0

            while i < len(b_t_key):
                if i + 1 < len(b_t_key) and winning_bytes[0] == b_t_key[i] and winning_bytes[1] == b_t_key[i + 1]:
                    # replace the pair with winning byte
                    new_key.append(merged_winning_bytes)
                    i += 2
                else:
                    new_key.append(b_t_key[i])
                    i += 1

            # update new_table with new {bytes key: freq count}
            new_table[tuple(new_key)] += b_t_count

        # freq_table is now new_table with new merged pairs!
        freq_table = new_table

        # add the new entry in vocab
        new_vocab_words.append(merged_winning_bytes)
        merges.append((winning_bytes[0], winning_bytes[1]))
    return merges, new_vocab_words


pretokenized_table = pretokenize(corpus_txt)
merged_tokens, new_vocab_words = merge(pretokenized_table, 6)

cur_vocab_len = len(vocab)
for nw in new_vocab_words:
    vocab[cur_vocab_len] = nw
    cur_vocab_len += 1


print("+++++++++++++++++++++++++++")
print("VOCAB: ", vocab)
print("+++++++++++++++++++++++++++")
print("VOCAB LEN CHECKS: ", min(vocab), max(vocab), len(vocab))
print("+++++++++++++++++++++++++++")
print("MERGED TOKENS: ", merged_tokens)
