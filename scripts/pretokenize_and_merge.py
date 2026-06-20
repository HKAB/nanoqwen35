"""
Pretokenize and merge: raw text parquets → merged flat dataset.

Walks --source-root recursively, collects all parquet files with "train" in
the filename for the train split (and "val" for the val split), tokenizes them
all, globally shuffles, and writes flat shards to --output-root.

No domain concept — all files are treated as one flat pool.
Parallelism: files are chunked across --workers processes.

Output is read directly by base_train.py (looks for merged_metadata.json).

Usage:
    python -m scripts.pretokenize_and_merge \\
        --source-root /path/to/vi_en_parquet_v1 \\
        --output-root /path/to/vi_en_merged_v1 \\
        --tokenizer   /path/to/Qwen3.5-0.8B-Base \\
        --T 4096 \\
        --num-shards 256 \\
        --num-val-shards 32 \\
        --workers 16

Output:
    {output_root}/merged_metadata.json
    {output_root}/train_0000.parquet  …  train_NNNN.parquet
    {output_root}/val_0000.parquet    …  val_MMMM.parquet
"""

import os
import sys
import json
import math
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Worker — top-level so ProcessPoolExecutor can pickle it
# ---------------------------------------------------------------------------

def _tokenize_files(task: dict) -> "np.ndarray | None":
    """
    Tokenize a flat list of parquet files.

    Streams all text rows through the HF Rust tokenizer, appending EOS after
    every document, then slices the token stream into T+1 packed rows.
    Any sub-row tail at the end of the last file is discarded.

    Returns an np.int32 array of shape (n_rows, T+1), or None if no rows produced.
    """
    from tokenizers import Tokenizer as HFTokenizer

    files    = task["files"]
    tok_path = task["tokenizer_path"]
    T        = task["T"]
    T1       = T + 1

    if not files:
        return None

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    tok    = HFTokenizer.from_pretrained(tok_path)
    eos_id = tok.token_to_id("<|endoftext|>")
    assert eos_id is not None, "EOS token <|endoftext|> not found in tokenizer vocab"

    overflow:   np.ndarray       = np.empty(0, dtype=np.int32)
    row_chunks: list[np.ndarray] = []

    for filepath in files:
        pf = pq.ParquetFile(filepath)
        for rg_i in range(pf.num_row_groups):
            rg    = pf.read_row_group(rg_i, columns=["text"])
            texts = rg.column("text").to_pylist()
            encs  = tok.encode_batch(texts, add_special_tokens=False)

            pieces: list[np.ndarray] = [overflow]
            for enc in encs:
                if enc.ids:
                    pieces.append(np.array(enc.ids, dtype=np.int32))
                pieces.append(np.array([eos_id], dtype=np.int32))
            stream = np.concatenate(pieces) if len(pieces) > 1 else overflow.copy()

            n_complete = len(stream) // T1
            if n_complete > 0:
                row_chunks.append(stream[:n_complete * T1].reshape(n_complete, T1).copy())
                overflow = stream[n_complete * T1:].copy()
            else:
                overflow = stream.copy()

    if not row_chunks:
        return None

    return np.concatenate(row_chunks, axis=0)


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------

def _write_shard(path: str, rows_np: np.ndarray, T_plus_1: int, rows_per_rg: int) -> None:
    """Atomically write rows_np with schema: input_ids fixed_size_list<int32>[T+1]."""
    tmp    = path + ".tmp"
    schema = pa.schema([("input_ids", pa.list_(pa.int32(), T_plus_1))])
    with pq.ParquetWriter(tmp, schema, compression="zstd", compression_level=3) as writer:
        for start in range(0, len(rows_np), rows_per_rg):
            chunk = rows_np[start:start + rows_per_rg]
            flat  = pa.array(chunk.ravel(), type=pa.int32())
            fsl   = pa.FixedSizeListArray.from_arrays(flat, T_plus_1)
            writer.write_table(pa.table({"input_ids": fsl}, schema=schema))
    os.replace(tmp, path)


def _write_split(
    split:       str,
    all_rows:    np.ndarray,
    output_root: str,
    num_shards:  int,
    T_plus_1:    int,
    rows_per_rg: int,
    rng:         np.random.Generator,
) -> int:
    """Shuffle all_rows in-place and write to flat parquet shards. Returns actual row count."""
    if len(all_rows) == 0:
        print(f"[{split}] WARNING: 0 rows — no shards written", flush=True)
        return 0

    print(f"[{split}] shuffling {len(all_rows):,} rows...", flush=True)
    rng.shuffle(all_rows)

    actual_shards  = min(num_shards, len(all_rows))
    rows_per_shard = math.ceil(len(all_rows) / actual_shards)
    if actual_shards < num_shards:
        print(f"[{split}] WARNING: only {len(all_rows):,} rows — writing {actual_shards} shards (< requested {num_shards})", flush=True)

    print(f"[{split}] writing {actual_shards} shards (~{rows_per_shard:,} rows each)...", flush=True)
    for i in range(actual_shards):
        start = i * rows_per_shard
        _write_shard(
            os.path.join(output_root, f"{split}_{i:04d}.parquet"),
            all_rows[start:start + rows_per_shard],
            T_plus_1,
            rows_per_rg,
        )

    print(f"[{split}] done — {len(all_rows):,} rows → {actual_shards} shards", flush=True)
    return len(all_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _chunk(files: list, n: int) -> list:
    """Split files into n roughly equal chunks."""
    n = min(n, len(files))
    size = math.ceil(len(files) / n)
    return [files[i:i + size] for i in range(0, len(files), size)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretokenize and merge raw text parquets into flat tokenized shards"
    )
    parser.add_argument("--source-root",    required=True, help="root directory containing raw text parquet files (searched recursively)")
    parser.add_argument("--output-root",    required=True, help="output directory for merged flat shards")
    parser.add_argument("--tokenizer",      default="Qwen/Qwen3.5-0.8B-Base", help="HF tokenizer path or model ID")
    parser.add_argument("--T",              type=int, default=4096, help="sequence length (each output row = T+1 tokens)")
    parser.add_argument("--num-shards",     type=int, default=256,  help="number of train output shards (use a multiple of max world_size)")
    parser.add_argument("--num-val-shards", type=int, default=32,   help="number of val output shards")
    parser.add_argument("--workers",        type=int, default=16,   help="parallel worker processes (files are chunked across workers)")
    parser.add_argument("--seed",           type=int, default=42,   help="shuffle seed")
    parser.add_argument("--rows-per-rg",    type=int, default=1024, help="parquet row-group size in output shards")
    parser.add_argument("--force",          action="store_true",    help="overwrite existing output")
    args = parser.parse_args()

    from nanoqwen35.dataset import list_all_parquet_files

    # ------------------------------------------------------------------
    # Discover files
    all_files   = list_all_parquet_files(args.source_root)
    train_files = [f for f in all_files if "train" in os.path.basename(f)]
    val_files   = [f for f in all_files if "val"   in os.path.basename(f)]

    if not train_files:
        print(f"ERROR: no train parquet files found under {args.source_root}", file=sys.stderr)
        print("Files must contain 'train' in the filename.", file=sys.stderr)
        sys.exit(1)

    T        = args.T
    T_plus_1 = T + 1

    os.makedirs(args.output_root, exist_ok=True)
    out_meta_path = os.path.join(args.output_root, "merged_metadata.json")
    if not args.force and os.path.exists(out_meta_path):
        print(f"Output already exists: {out_meta_path}\nUse --force to overwrite.")
        sys.exit(0)

    print(f"{'='*60}")
    print(f"pretokenize_and_merge")
    print(f"  source    : {args.source_root}")
    print(f"  output    : {args.output_root}")
    print(f"  tokenizer : {args.tokenizer}")
    print(f"  T         : {T}")
    print(f"  workers   : {args.workers}")
    print(f"  seed      : {args.seed}")
    print(f"  train files: {len(train_files)}")
    print(f"  val files  : {len(val_files)}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Chunk files and submit tasks
    train_chunks = _chunk(train_files, args.workers)
    val_chunks   = _chunk(val_files,   args.workers) if val_files else []

    all_tasks = [("train", c) for c in train_chunks] + [("val", c) for c in val_chunks]
    n_workers = min(args.workers, len(all_tasks))

    print(f"Submitting {len(train_chunks)} train + {len(val_chunks)} val chunks ({n_workers} workers)...\n")

    train_arrays: list[np.ndarray] = []
    val_arrays:   list[np.ndarray] = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {
            pool.submit(_tokenize_files, {"files": chunk, "tokenizer_path": args.tokenizer, "T": T}): (split, i)
            for i, (split, chunk) in enumerate(all_tasks)
        }
        for fut in as_completed(futs):
            split, i = futs[fut]
            try:
                rows = fut.result()
            except Exception as exc:
                print(f"FAILED chunk {i} ({split}): {exc}", file=sys.stderr)
                raise
            if rows is not None:
                print(f"  chunk {i:03d} [{split}]: {len(rows):,} rows", flush=True)
                if split == "train":
                    train_arrays.append(rows)
                else:
                    val_arrays.append(rows)

    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # Merge, shuffle, write — train
    print("\n[train] concatenating chunks...")
    all_train = np.concatenate(train_arrays, axis=0)
    del train_arrays
    num_train = _write_split("train", all_train, args.output_root, args.num_shards, T_plus_1, args.rows_per_rg, rng)
    del all_train

    # ------------------------------------------------------------------
    # Merge, shuffle, write — val
    if val_arrays:
        print("\n[val] concatenating chunks...")
        all_val = np.concatenate(val_arrays, axis=0)
        del val_arrays
        num_val = _write_split("val", all_val, args.output_root, args.num_val_shards, T_plus_1, args.rows_per_rg,
                               np.random.default_rng(args.seed + 1))
        del all_val
    else:
        print("\n[val] WARNING: no val files found — skipping")
        num_val = 0

    # ------------------------------------------------------------------
    # Write metadata
    meta = {
        "T":                T,
        "tokenizer":        args.tokenizer,
        "source_root":      args.source_root,
        "num_train_rows":   num_train,
        "num_train_shards": args.num_shards,
        "num_val_rows":     num_val,
        "num_val_shards":   args.num_val_shards,
        "seed":             args.seed,
    }
    with open(out_meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done.")
    print(f"  Train : {num_train:,} rows  →  {args.num_shards} shards")
    print(f"  Val   : {num_val:,} rows  →  {args.num_val_shards} shards")
    print(f"  Meta  : {out_meta_path}")
    print(f"\nAt world_size=8: {args.num_shards // 8} train shards per rank (file-sharding guaranteed)")


if __name__ == "__main__":
    main()
