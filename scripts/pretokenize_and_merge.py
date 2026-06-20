"""
Pretokenize and merge: raw text parquets → merged flat dataset.

Walks --source-root recursively, collects all parquet files with "train" in
the filename for the train split (and "val" for the val split), tokenizes them
all, globally shuffles, and writes flat shards to --output-root.

No domain concept — all files are treated as one flat pool.
Parallelism: files are chunked across --workers processes.

Memory model:
  Each worker tokenizes its chunk and writes rows to a temp .npy file on disk
  (never held in the main process RAM).  After all workers finish, main loads
  each temp file one-at-a-time into a memory-mapped scratch file (disk-backed),
  generates a global shuffle index (tiny: n_rows × 4 bytes), then writes output
  shards by reading from the scratch file in sorted-index order (sequential I/O).

  Peak RAM in main: max(one_chunk_rows) × (T+1) × 4 bytes ≈ hundreds of MB.
  Scratch disk: total_rows × (T+1) × 4 bytes ≈ same size as output.

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
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Worker — top-level so ProcessPoolExecutor can pickle it
# ---------------------------------------------------------------------------

def _tokenize_files(task: dict) -> dict:
    """
    Tokenize a flat list of parquet files and write rows to a temp .npy file.

    Returns {"tmp_path": str, "n_rows": int, "elapsed": float}.
    The caller is responsible for deleting tmp_path after use.
    """
    import time as _time
    from tokenizers import Tokenizer as HFTokenizer

    files       = task["files"]
    tok_path    = task["tokenizer_path"]
    T           = task["T"]
    T1          = T + 1
    chunk_id    = task["chunk_id"]
    split       = task["split"]
    output_root = task["output_root"]
    label       = f"[{split}|W{chunk_id:02d}]"
    n_files     = len(files)
    tmp_path    = os.path.join(output_root, f".tmp_{split}_w{chunk_id:03d}.npy")

    if not files:
        return {"tmp_path": None, "n_rows": 0, "elapsed": 0.0}

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    tok    = HFTokenizer.from_pretrained(tok_path)
    eos_id = tok.token_to_id("<|endoftext|>")
    assert eos_id is not None, "EOS token <|endoftext|> not found in tokenizer vocab"

    overflow:   np.ndarray       = np.empty(0, dtype=np.int32)
    row_chunks: list[np.ndarray] = []
    total_rows  = 0
    t_start     = _time.time()

    for f_idx, filepath in enumerate(files):
        fname  = os.path.basename(filepath)
        pf     = pq.ParquetFile(filepath)
        n_rgs  = pf.num_row_groups
        print(f"{label} {f_idx+1:3d}/{n_files} start  {fname}  ({n_rgs} rg)", flush=True)
        t_file      = _time.time()
        rows_before = total_rows

        for rg_i in range(n_rgs):
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
                overflow    = stream[n_complete * T1:].copy()
                total_rows += n_complete
            else:
                overflow = stream.copy()

        new_rows = total_rows - rows_before
        dt_file  = _time.time() - t_file
        print(
            f"{label} {f_idx+1:3d}/{n_files} done   {fname}  "
            f"+{new_rows:>8,} rows  total {total_rows:>10,}  ({dt_file:.1f}s)",
            flush=True,
        )

    elapsed = _time.time() - t_start

    if not row_chunks:
        print(f"{label} WARNING: 0 rows produced from {n_files} files", flush=True)
        return {"tmp_path": None, "n_rows": 0, "elapsed": elapsed}

    rows = np.concatenate(row_chunks, axis=0)
    del row_chunks

    print(f"{label} saving {len(rows):,} rows to temp file...", flush=True)
    np.save(tmp_path, rows)
    del rows

    elapsed = _time.time() - t_start
    print(
        f"{label} COMPLETE — {n_files} files  {total_rows:,} rows  {elapsed:.1f}s",
        flush=True,
    )
    return {"tmp_path": tmp_path, "n_rows": total_rows, "elapsed": elapsed}


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
    split:        str,
    chunk_results: list,          # list of dicts from _tokenize_files
    output_root:  str,
    num_shards:   int,
    T_plus_1:     int,
    rows_per_rg:  int,
    seed:         int,
) -> int:
    """
    Merge temp chunk files → memory-mapped scratch → globally shuffled output shards.

    Steps:
      1. Load each chunk's .npy file into a memmap (one chunk at a time → low peak RAM)
      2. Build a global shuffle index (just integers, tiny)
      3. Write each output shard by reading shuffled rows from the memmap
      4. Delete scratch file and temp .npy files
    """
    valid = [(r["tmp_path"], r["n_rows"]) for r in chunk_results if r["n_rows"] > 0]
    if not valid:
        print(f"[{split}] WARNING: 0 rows — no shards written", flush=True)
        return 0

    total_rows = sum(n for _, n in valid)
    scratch    = os.path.join(output_root, f".{split}_scratch.bin")

    # ------------------------------------------------------------------
    # Phase 1: load temp files into memmap (one at a time)
    print(f"\n[{split}] creating scratch memmap ({total_rows:,} × {T_plus_1} int32 = "
          f"{total_rows * T_plus_1 * 4 / 1e9:.2f} GB)...", flush=True)
    mm     = np.memmap(scratch, dtype=np.int32, mode="w+", shape=(total_rows, T_plus_1))
    offset = 0
    t_load = time.time()

    for tmp_path, n_rows in valid:
        chunk = np.load(tmp_path, mmap_mode="r")   # mmap avoids double-buffering
        mm[offset:offset + n_rows] = chunk
        offset += n_rows
        os.remove(tmp_path)
        print(
            f"[{split}]   loaded → {offset:>10,}/{total_rows:,} rows  "
            f"({100*offset//total_rows:3d}%)  {time.time()-t_load:.1f}s",
            flush=True,
        )
        del chunk

    mm.flush()
    print(f"[{split}] memmap ready ({time.time()-t_load:.1f}s)", flush=True)

    # ------------------------------------------------------------------
    # Phase 2: global shuffle index (fits in RAM: total_rows × 8 bytes)
    rng = np.random.default_rng(seed)
    print(f"[{split}] generating shuffle index...", flush=True)
    idx = rng.permutation(total_rows)

    # ------------------------------------------------------------------
    # Phase 3: write output shards
    actual_shards  = min(num_shards, total_rows)
    rows_per_shard = math.ceil(total_rows / actual_shards)
    if actual_shards < num_shards:
        print(f"[{split}] WARNING: only {total_rows:,} rows — writing {actual_shards} shards", flush=True)

    print(
        f"[{split}] writing {actual_shards} shards (~{rows_per_shard:,} rows each)...",
        flush=True,
    )
    log_every = max(1, actual_shards // 10)
    t_write   = time.time()

    for shard_i in range(actual_shards):
        # Sort the shard's indices so memmap reads are sequential (much faster I/O)
        shard_idx  = np.sort(idx[shard_i * rows_per_shard : (shard_i + 1) * rows_per_shard])
        shard_rows = np.array(mm[shard_idx])   # copy into RAM — one shard at a time
        _write_shard(
            os.path.join(output_root, f"{split}_{shard_i:04d}.parquet"),
            shard_rows,
            T_plus_1,
            rows_per_rg,
        )
        del shard_rows

        if (shard_i + 1) % log_every == 0 or (shard_i + 1) == actual_shards:
            elapsed = time.time() - t_write
            rate    = (shard_i + 1) / elapsed if elapsed > 0 else 0
            eta     = (actual_shards - shard_i - 1) / rate if rate > 0 else 0
            print(
                f"[{split}] shards {shard_i+1:4d}/{actual_shards}  "
                f"({100*(shard_i+1)//actual_shards:3d}%)  "
                f"{elapsed:.0f}s elapsed  eta {eta:.0f}s",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Phase 4: cleanup scratch
    del mm
    os.remove(scratch)
    print(
        f"[{split}] done — {total_rows:,} rows → {actual_shards} shards  "
        f"({time.time()-t_write:.1f}s write)",
        flush=True,
    )
    return total_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _chunk(files: list, n: int) -> list:
    """Split files into n roughly equal chunks."""
    n    = min(n, len(files))
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
    print(f"  source      : {args.source_root}")
    print(f"  output      : {args.output_root}")
    print(f"  tokenizer   : {args.tokenizer}")
    print(f"  T           : {T}")
    print(f"  workers     : {args.workers}")
    print(f"  seed        : {args.seed}")
    print(f"  train files : {len(train_files)}")
    print(f"  val files   : {len(val_files)}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Chunk files and build tasks
    train_chunks = _chunk(train_files, args.workers)
    val_chunks   = _chunk(val_files,   args.workers) if val_files else []

    all_tasks = []
    for i, chunk in enumerate(train_chunks):
        all_tasks.append({
            "files": chunk, "tokenizer_path": args.tokenizer, "T": T,
            "chunk_id": i, "split": "train", "output_root": args.output_root,
        })
    for i, chunk in enumerate(val_chunks):
        all_tasks.append({
            "files": chunk, "tokenizer_path": args.tokenizer, "T": T,
            "chunk_id": i, "split": "val", "output_root": args.output_root,
        })

    n_workers = min(args.workers, len(all_tasks))
    n_total   = len(all_tasks)
    print(
        f"Submitting {len(train_chunks)} train + {len(val_chunks)} val chunks "
        f"({n_workers} parallel workers)...\n",
        flush=True,
    )

    train_results: list[dict] = []
    val_results:   list[dict] = []
    n_done   = 0
    t_global = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_tokenize_files, task): task for task in all_tasks}

        for fut in as_completed(futs):
            task = futs[fut]
            split, chunk_id = task["split"], task["chunk_id"]
            try:
                result = fut.result()
            except Exception as exc:
                print(f"FAILED [{split}|W{chunk_id:02d}]: {exc}", file=sys.stderr)
                raise

            n_done  += 1
            elapsed  = time.time() - t_global
            print(
                f">>> [{split}|W{chunk_id:02d}] chunk done — {result['n_rows']:,} rows  "
                f"({n_done}/{n_total} chunks, {elapsed:.0f}s elapsed)\n",
                flush=True,
            )
            if split == "train":
                train_results.append(result)
            else:
                val_results.append(result)

    # ------------------------------------------------------------------
    # Merge, shuffle, write — train
    num_train = _write_split(
        "train", train_results, args.output_root,
        args.num_shards, T_plus_1, args.rows_per_rg, args.seed,
    )

    # Merge, shuffle, write — val
    num_val = _write_split(
        "val", val_results, args.output_root,
        args.num_val_shards, T_plus_1, args.rows_per_rg, args.seed + 1,
    ) if val_results else 0

    if not val_results:
        print("\n[val] WARNING: no val files found — skipping", flush=True)

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

    total_elapsed = time.time() - t_global
    print(f"\n{'='*60}")
    print(f"Done in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"  Train : {num_train:,} rows  →  {args.num_shards} shards")
    print(f"  Val   : {num_val:,} rows  →  {args.num_val_shards} shards")
    print(f"  Meta  : {out_meta_path}")
    print(f"\nAt world_size=8: {args.num_shards // 8} train shards per rank")


if __name__ == "__main__":
    main()
