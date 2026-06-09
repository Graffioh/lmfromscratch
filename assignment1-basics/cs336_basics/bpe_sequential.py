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

freq_table: Counter[tuple[bytes, ...]] = Counter()


# simple, via whitespaces not gpt 2 regex
def pretokenize(text: str):
    txt_split = re.split(r"\s+", text)

    for s in txt_split:
        # convert the string into bytes
        bytes_str = s.encode("utf-8")
        bytes_list: list[bytes] = [bytes([b]) for b in bytes_str]

        # for each bytes, we need to increment the counter
        bytes_tuple: tuple[bytes, ...] = tuple(bytes_list)
        freq_table[bytes_tuple] += 1


def merge(
    freq_table: dict[tuple[bytes, ...], int], max_merges: int = 3
) -> tuple[list[tuple[bytes, bytes]], list[bytes]]:
    new_vocab_words: list[bytes] = []
    merges: list[tuple[bytes, bytes]] = []

    # merge max_merges times
    for _ in range(max_merges):
        # gather all the pairs to merge with respective cumulative count
        pairs_to_merge: defaultdict[tuple[bytes, ...], int] = defaultdict(int)
        for bytes_key, bytes_count in freq_table.items():
            pairs: list[tuple[bytes, bytes]] = []
            for i in range(len(bytes_key) - 1):
                pairs.append((bytes_key[i], bytes_key[i + 1]))

            for p in pairs:
                pairs_to_merge[p] += bytes_count

        # we must check if there is any pair still mergeable
        # (if pairs_table == 0, then it means we don't have pairs anymore, only single tokens hence nothing to merge)
        if len(pairs_to_merge) == 0:
            break

        # take the winning pair (break ties with lexographically greater) and merge it
        max_pair = max(pairs_to_merge.items(), key=lambda pair: (pair[1], pair[0]))
        winning_bytes = max_pair[0]
        merged_winning_bytes = winning_bytes[0] + winning_bytes[1]

        # find the winning bytes in the freq_table keys and whenever found, merge them
        new_tmp_table: defaultdict[tuple[bytes, ...], int] = defaultdict(int)
        for bytes_key, bytes_count in freq_table.items():
            new_key: list[bytes] = []
            i = 0

            while i < len(bytes_key):
                if i + 1 < len(bytes_key) and winning_bytes[0] == bytes_key[i] and winning_bytes[1] == bytes_key[i + 1]:
                    new_key.append(merged_winning_bytes)
                    i += 2
                else:
                    new_key.append(bytes_key[i])
                    i += 1

            # update new_table with new {bytes key: freq count}
            new_tmp_table[tuple(new_key)] += bytes_count

        # freq_table is now the new table with new merged pairs!
        freq_table = new_tmp_table

        # add the new entry in vocab
        new_vocab_words.append(merged_winning_bytes)

        merges.append((winning_bytes[0], winning_bytes[1]))
    return merges, new_vocab_words


# split docs by EOT delimiter before pretokenize (so we avoid pretokenization across docs)
txt_docs = re.split("<\\|endoftext\\|>", corpus_txt)
for txt in txt_docs:
    pretokenize(txt)

merged_tokens, new_vocab_words = merge(freq_table, 6)

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
