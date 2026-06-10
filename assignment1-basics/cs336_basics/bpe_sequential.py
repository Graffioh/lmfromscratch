from collections import Counter, defaultdict

import regex

VOCAB_BASE_SIZE = 256

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
gpt_regex_pattern = regex.compile(GPT2_PAT)


def pretokenize(freq_table: Counter[tuple[bytes, ...]], text: str):
    txt_split = regex.finditer(gpt_regex_pattern, text)

    for s_objects in txt_split:
        # convert the string into bytes
        bytes_str = s_objects.group().encode("utf-8")
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
    for i in range(max_merges):
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

        # store the new entries to insert in vocab
        new_vocab_words.append(merged_winning_bytes)

        merges.append((winning_bytes[0], winning_bytes[1]))
    return (merges, new_vocab_words)


def train_bpe(
    input_path: str, max_vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    freq_table: Counter[tuple[bytes, ...]] = Counter()

    # build base vocab
    vocab: defaultdict[int, bytes] = defaultdict(bytes)
    for i, st in enumerate(special_tokens):
        vocab[i] = st.encode("utf-8")

    cur_vocab_len = len(vocab)
    for i in range(VOCAB_BASE_SIZE):
        vocab[i + cur_vocab_len] = bytes([i])

    # read input text
    corpus_text = ""
    with open(input_path, encoding="utf-8") as f:
        corpus_text = f.read()

    # split docs by special tokens (e.g. EOT delimiter) and pretokenize
    delimited_special_tokens = "|".join(regex.escape(st) for st in special_tokens)
    txt_docs = regex.split(delimited_special_tokens, corpus_text)
    for txt in txt_docs:
        pretokenize(freq_table, txt)

    max_merges_count = max_vocab_size - VOCAB_BASE_SIZE - len(special_tokens)
    merges, new_vocab_words = merge(freq_table, max_merges_count)

    cur_vocab_len = len(vocab)
    for nw in new_vocab_words:
        vocab[cur_vocab_len] = nw
        cur_vocab_len += 1

    print("******************************")
    print(
        f"len(vocab)={len(vocab)} max_merges={max_merges_count} n_merges={len(merges)} n_special_tokens={len(special_tokens)}"
    )
    print("******************************")

    return (vocab, merges)


vocab, merges = train_bpe("./cs336_basics/corpus_example.txt", 271, ["<|endoftext|>"])

print("---------------------")
print(f"VOCAB: {vocab}")
print("---------------------")
print(f"MERGES: {merges}")
print("---------------------")
