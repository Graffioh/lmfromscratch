import os
from collections import Counter, defaultdict
from multiprocessing import Pool
from typing import BinaryIO

import regex

VOCAB_BASE_SIZE = 256

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
gpt_regex_pattern = regex.compile(GPT2_PAT)


# taken from `pretokenization_example.py`
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    _ = file.seek(0, os.SEEK_END)
    file_size = file.tell()
    _ = file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        _ = file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize(text: str):
    txt_split = regex.finditer(gpt_regex_pattern, text)

    for s_objects in txt_split:
        # convert the string into bytes
        bytes_str = s_objects.group().encode("utf-8")
        bytes_list: list[bytes] = [bytes([b]) for b in bytes_str]

        # for each bytes, we need to increment the counter
        bytes_tuple: tuple[bytes, ...] = tuple(bytes_list)
        yield bytes_tuple


def merge(
    freq_table: dict[tuple[bytes, ...], int], max_merges: int = 3
) -> tuple[list[tuple[bytes, bytes]], list[bytes]]:
    new_vocab_words: list[bytes] = []
    merges: list[tuple[bytes, bytes]] = []

    # optimization 1 - use a reverse index with <pair>: add((<word>, <count>)) after picking the winning pair
    #   so that we don't traverse each freq_table key to find in which word, the pair appear
    words = list(freq_table.items())
    index_pair_to_word_slots: defaultdict[tuple[bytes, ...], set[int]] = defaultdict(set[int])

    # merge max_merges times
    for i in range(max_merges):
        # gather all the pairs to merge with respective cumulative count
        pairs_to_merge: defaultdict[tuple[bytes, ...], int] = defaultdict(int)
        for pos, (bytes_key, bytes_count) in enumerate(words):
            pairs: list[tuple[bytes, bytes]] = []
            for i in range(len(bytes_key) - 1):
                bytes_to_merge = (bytes_key[i], bytes_key[i + 1])
                pairs.append(bytes_to_merge)
                index_pair_to_word_slots[bytes_to_merge].add(pos)

            for p in pairs:
                pairs_to_merge[p] += bytes_count

        # we must check if there is any pair still mergeable
        # (if pairs_table == 0, then it means we don't have pairs anymore, only single tokens hence nothing to merge)
        if len(pairs_to_merge) == 0:
            break

        # take the winning pair (break ties with lexographically greater) and merge it
        max_pair = max(pairs_to_merge.items(), key=lambda pair: (pair[1], pair[0]))
        winning_pair = max_pair[0]
        merged_winning_bytes = winning_pair[0] + winning_pair[1]

        for slot in index_pair_to_word_slots[winning_pair]:
            cur_word, cur_count = words[slot]
            new_word: list[bytes] = []
            i = 0
            while i < len(cur_word):
                if i + 1 < len(cur_word) and cur_word[i] == winning_pair[0] and cur_word[i + 1] == winning_pair[1]:
                    new_word.append(cur_word[i] + cur_word[i + 1])
                    i += 2
                else:
                    new_word.append(cur_word[i])
                    i += 1

            words[slot] = (tuple(new_word), cur_count)

        # store the new entries to insert in vocab
        new_vocab_words.append(merged_winning_bytes)

        merges.append((winning_pair[0], winning_pair[1]))
    return (merges, new_vocab_words)


def pretokenize_worker(input_path: str, start: int, end: int, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    tmp_freq_table: Counter[tuple[bytes, ...]] = Counter()

    with open(input_path, "rb") as f:
        _ = f.seek(start)
        corpus_text_chunk = f.read(end - start).decode("utf-8", errors="ignore")

        # split docs by special tokens (e.g. EOT delimiter) and pretokenize
        delimited_special_tokens = "|".join(regex.escape(st) for st in special_tokens)
        txt_docs = regex.split(delimited_special_tokens, corpus_text_chunk)
        for txt in txt_docs:
            for bytes_tuple in pretokenize(txt):
                tmp_freq_table[bytes_tuple] += 1

    return tmp_freq_table


def train_bpe(
    input_path: str | os.PathLike[str], max_vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    freq_table: Counter[tuple[bytes, ...]] = Counter()

    # build base vocab
    vocab: defaultdict[int, bytes] = defaultdict(bytes)
    for i, st in enumerate(special_tokens):
        vocab[i] = st.encode("utf-8")

    cur_vocab_len = len(vocab)
    for i in range(VOCAB_BASE_SIZE):
        vocab[i + cur_vocab_len] = bytes([i])

    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # optimization 2 - sending chunks to different processes
        pretokenize_args: list[tuple[str, int, int, list[str]]] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            pretokenize_args.append((str(input_path), start, end, special_tokens))

        with Pool(
            num_processes,
        ) as pool:
            freq_table_parallel = pool.starmap(pretokenize_worker, pretokenize_args)

    for freq_table_chunk in freq_table_parallel:
        freq_table.update(freq_table_chunk)

    max_merges_count = max_vocab_size - VOCAB_BASE_SIZE - len(special_tokens)
    merges, new_vocab_words = merge(freq_table, max_merges_count)

    cur_vocab_len = len(vocab)
    for nw in new_vocab_words:
        vocab[cur_vocab_len] = nw
        cur_vocab_len += 1

    return (vocab, merges)
