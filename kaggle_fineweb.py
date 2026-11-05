"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.

Run as a script:   $ python fineweb.py
Run in a notebook: main()

Output dir is chosen automatically:
  - env var FINEWEB_OUT_DIR if set
  - /kaggle/working/edu_fineweb10B on Kaggle
  - <script folder>/edu_fineweb10B when run as a script
  - ./edu_fineweb10B otherwise (e.g. notebooks elsewhere)
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset  # pip install datasets
from tqdm import tqdm              # pip install tqdm

# ------------------------------------------
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8)   # 100M tokens per shard, total of ~100 shards
max_shards = 10       # e.g. 20 to stop early (useful on Kaggle's ~20GB disk limit)


def get_data_cache_dir():
    # 1) explicit override
    override = os.environ.get("FINEWEB_OUT_DIR")
    if override:
        return override
    # 2) Kaggle
    if os.path.exists("/kaggle/working"):
        return os.path.join("/kaggle/working", local_dir)
        # for more disk space (not persisted after session), use:
        # return os.path.join("/kaggle/temp", local_dir)
    # 3) normal script
    try:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), local_dir)
    except NameError:
        # 4) notebook / interactive
        return os.path.join(os.getcwd(), local_dir)


DATA_CACHE_DIR = get_data_cache_dir()
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# init the tokenizer (global so worker processes can use it)
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']  # end of text token


def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot]  # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


def shard_path(shard_index):
    split = "val" if shard_index == 0 else "train"
    return os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")


def main():
    print(f"Saving shards to: {DATA_CACHE_DIR}")

    # download the dataset
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

    # tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)  # buffer for current shard
        token_count = 0
        progress_bar = None
        stopped_early = False

        for tokens in pool.imap(tokenize, fw, chunksize=16):

            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")

            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)
                progress_bar.update(len(tokens))
            else:
                # fill the current shard, write it, start a new one with the leftovers
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                progress_bar.close()
                all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
                write_datafile(shard_path(shard_index), all_tokens_np)
                shard_index += 1
                progress_bar = None

                if max_shards is not None and shard_index >= max_shards:
                    stopped_early = True
                    break

                leftover = len(tokens) - remainder
                all_tokens_np[0:leftover] = tokens[remainder:]
                token_count = leftover

        # write any remaining tokens as the last shard
        if not stopped_early and token_count != 0:
            write_datafile(shard_path(shard_index), all_tokens_np[:token_count])

    print("Done.")


if __name__ == "__main__":
    main()