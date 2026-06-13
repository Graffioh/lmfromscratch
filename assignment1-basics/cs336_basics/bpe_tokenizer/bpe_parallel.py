import os
from collections import Counter, defaultdict
from multiprocessing import Pool
from typing import BinaryIO

import regex
from tqdm import tqdm

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
    txt_split = gpt_regex_pattern.finditer(text)
    word_count: Counter[str] = Counter()

    for s_objects in txt_split:
        # count the words occurrences
        word = s_objects.group()
        word_count[word] += 1

    return word_count


def pretokenize_worker_star(args: tuple[str, int, int, list[str]]) -> Counter[str]:
    # unpacking wrapper so imap_unordered (single-arg) can call the worker
    return pretokenize_worker(*args)


def pretokenize_worker(input_path: str, start: int, end: int, special_tokens: list[str]) -> Counter[str]:
    tmp_pretoken_freq_table: Counter[str] = Counter()

    with open(input_path, "rb") as f:
        _ = f.seek(start)
        corpus_text_chunk = f.read(end - start).decode("utf-8", errors="ignore")

        # split docs by special tokens (e.g. EOT delimiter) and pretokenize
        delimited_special_tokens = "|".join(regex.escape(st) for st in special_tokens)
        txt_docs = regex.split(delimited_special_tokens, corpus_text_chunk)
        for txt in txt_docs:
            words_with_count = pretokenize(txt)
            for word, cnt in words_with_count.items():
                tmp_pretoken_freq_table[word] += cnt

    return tmp_pretoken_freq_table


def merge(
    freq_table: dict[tuple[bytes, ...], int], max_merges: int = 3
) -> tuple[list[tuple[bytes, bytes]], list[bytes]]:
    new_vocab_words: list[bytes] = []
    merges: list[tuple[bytes, bytes]] = []

    # optimization - use a reverse index with <pair>: add((<word>, <count>)) after picking the winning pair
    #   so that we don't traverse each freq_table key to find in which word, the pair appear
    # IMPORTANT: edit them in-place during merge
    words = list(freq_table.items())
    index_pair_to_count: Counter[tuple[bytes, bytes]] = Counter()
    index_pair_to_word_slots: defaultdict[tuple[bytes, bytes], set[int]] = defaultdict(set[int])

    # gather all the pairs to merge with respective cumulative count
    for pos, (word, count) in enumerate(words):
        for w_pos in range(len(word) - 1):
            byte_pair = (word[w_pos], word[w_pos + 1])
            index_pair_to_word_slots[byte_pair].add(pos)
            index_pair_to_count[byte_pair] += count

    # merge max_merges times
    for i in tqdm(range(max_merges), desc="bpe merges"):
        # we must check if there is any pair still mergeable
        # (if == 0, then it means we don't have pairs anymore, only single tokens hence nothing to merge)
        if len(index_pair_to_word_slots) == 0:
            break

        # take the winning pair, first by count then by lexographically greater bytes and merge it
        max_pair_and_count = max(index_pair_to_count.items(), key=lambda pair: (pair[1], pair[0]))
        max_pair = max_pair_and_count[0]
        byte1: bytes = max_pair[0]
        byte2: bytes = max_pair[1]

        for slot in index_pair_to_word_slots[max_pair]:
            cur_word, cur_count = words[slot]

            # decrement the count since we're gonna merge bytes
            for cur_w_pos in range(len(cur_word) - 1):
                byte_pair = (cur_word[cur_w_pos], cur_word[cur_w_pos + 1])
                index_pair_to_count[byte_pair] -= cur_count

            new_word: list[bytes] = []
            i = 0
            while i < len(cur_word):
                if i + 1 < len(cur_word) and cur_word[i] == max_pair[0] and cur_word[i + 1] == max_pair[1]:
                    new_word.append(cur_word[i] + cur_word[i + 1])
                    i += 2
                else:
                    new_word.append(cur_word[i])
                    i += 1

            # restore the decremented count on the bytes not merged, and add new one on the bytes merged
            for new_w_pos in range(len(new_word) - 1):
                byte_pair = (new_word[new_w_pos], new_word[new_w_pos + 1])
                index_pair_to_count[byte_pair] += cur_count
                index_pair_to_word_slots[byte_pair].add(slot)

            words[slot] = (tuple(new_word), cur_count)

        _ = index_pair_to_count.pop((byte1, byte2), None)
        _ = index_pair_to_word_slots.pop((byte1, byte2), None)

        # store the new entries to insert in vocab and also update merges with the inplace pairs
        new_vocab_words.append(max_pair[0] + max_pair[1])
        merges.append((byte1, byte2))
    return (merges, new_vocab_words)


def train_bpe(
    input_path: str | os.PathLike[str], max_vocab_size: int, special_tokens: list[str], n_proc: int = 1
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
        num_processes = n_proc
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # optimization - sending chunks to different processes (parallel)
        pretokenize_args: list[tuple[str, int, int, list[str]]] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            pretokenize_args.append((str(input_path), start, end, special_tokens))

        with Pool(
            num_processes,
        ) as pool:
            pretoken_freq_table_from_parallel = list(
                tqdm(
                    # results arrive as each chunk completes, with yield, so we can use tqdm
                    pool.imap_unordered(pretokenize_worker_star, pretokenize_args),
                    total=len(pretokenize_args),
                    desc="pretokenize chunks",
                )
            )

        for pretoken_freq_table_chunk in pretoken_freq_table_from_parallel:
            freq_table_chunk: Counter[tuple[bytes, ...]] = Counter()
            for pretoken, count in pretoken_freq_table_chunk.items():
                bytes_from_pretoken = pretoken.encode("utf-8")
                bytes_split: list[bytes] = []
                for b in bytes_from_pretoken:
                    bytes_split.append(bytes([b]))

                freq_table_chunk[tuple(bytes_split)] = count
            freq_table.update(freq_table_chunk)

        max_merges_count = max_vocab_size - VOCAB_BASE_SIZE - len(special_tokens)
        merges, new_vocab_words = merge(freq_table, max_merges_count)

        cur_vocab_len = len(vocab)
        for nw in new_vocab_words:
            vocab[cur_vocab_len] = nw
            cur_vocab_len += 1

    return (vocab, merges)
