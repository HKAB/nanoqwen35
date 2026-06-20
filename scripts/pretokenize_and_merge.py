"""
Pretokenize and merge: raw text parquets → flat tokenized shards.

Strategy:
  1. Collect all parquet files, shuffle the FILE LIST (file-level shuffle).
     Every parquet has ~equal rows, so this approximates row-level shuffle
     at essentially zero cost.
  2. Chunk the shuffled list across --workers processes.
  3. Each worker streams through its files in order, tokenizing and writing
     output shards CONTINUOUSLY — as soon as --rows-per-shard rows accumulate,
     a shard is flushed to disk and memory is freed.
  4. No scratch file, no global sort, no huge in-memory accumulation.

Peak RAM per worker:
  ≈ (rows_in_largest_single_file + rows_per_shard) × (T+1) × 4 bytes

  rows_in_largest_file = file_text_tokens / (T+1)
    e.g. a 1 GB text file with avg 50-token sentences × 1 M sentences
         ≈ 50 M tokens / 4097 ≈ 12 200 rows → ~200 MB

  So: peak_per_worker ≈ largest_file_MB + rows_per_shard × 16 KB
      total_peak      ≈ n_workers × peak_per_worker

  Tune: reduce --workers to cut memory; reduce --rows-per-shard to cut
        shard-buffer overhead (minor effect vs file size).

Output is read directly by base_train.py (looks for merged_metadata.json).

Shard naming: {split}_{chunk_id:04d}_{local_shard:06d}.parquet
  Sorted, they interleave workers — DDP file-sharding gets even mixing.

Usage:
    python -m scripts.pretokenize_and_merge \\
        --source-root /path/to/vi_en_parquet_v1 \\
        --output-root /path/to/vi_en_merged_v1 \\
        --tokenizer   /path/to/Qwen3.5-0.8B-Base \\
        --T 4096 --rows-per-shard 4096 --workers 8
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
# Shard writer (used both by workers and main)
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


# ---------------------------------------------------------------------------
# Worker — top-level so ProcessPoolExecutor can pickle it
# ---------------------------------------------------------------------------

def _tokenize_and_write_chunk(task: dict) -> dict:
    """
    Tokenize files in order and write shards continuously.

    Memory at any point: current_file_rows + shard_buffer_rows.
    As soon as rows_per_shard rows accumulate a shard is flushed to disk.

    Returns {"n_rows": int, "n_shards": int, "elapsed": float}.
    """
    import time as _time
    from tokenizers import Tokenizer as HFTokenizer

    files          = task["files"]
    tok_path       = task["tokenizer_path"]
    T              = task["T"]
    T1             = T + 1
    chunk_id       = task["chunk_id"]
    split          = task["split"]
    output_root    = task["output_root"]
    rows_per_shard = task["rows_per_shard"]
    rows_per_rg    = task["rows_per_rg"]
    label          = f"[{split}|W{chunk_id:02d}]"
    n_files        = len(files)

    if not files:
        return {"n_rows": 0, "n_shards": 0, "elapsed": 0.0}

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    tok    = HFTokenizer.from_pretrained(tok_path)
    eos_id = tok.token_to_id("<|endoftext|>")
    assert eos_id is not None, "EOS token <|endoftext|> not found in tokenizer vocab"

    # State persisted across files
    overflow:  np.ndarray       = np.empty(0, dtype=np.int32)
    shard_buf: list[np.ndarray] = []
    buf_rows:  int              = 0
    shard_idx: int              = 0
    total_rows: int             = 0
    t_start = _time.time()

    def _flush_shard(rows: np.ndarray) -> None:
        nonlocal shard_idx
        path = os.path.join(output_root, f"{split}_{chunk_id:04d}_{shard_idx:06d}.parquet")
        _write_shard(path, rows, T1, rows_per_rg)
        print(
            f"{label}   -> shard {shard_idx:04d}  {len(rows):,} rows  "
            f"{os.path.basename(path)}",
            flush=True,
        )
        shard_idx += 1

    def _drain_buffer() -> None:
        """Write as many complete shards as the buffer allows."""
        nonlocal shard_buf, buf_rows
        while buf_rows >= rows_per_shard:
            all_buf   = np.concatenate(shard_buf)
            to_write  = all_buf[:rows_per_shard].copy()
            leftover  = all_buf[rows_per_shard:]
            del all_buf
            _flush_shard(to_write)
            del to_write
            shard_buf = [leftover] if len(leftover) > 0 else []
            buf_rows  = len(leftover)

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
                new_rows = stream[:n_complete * T1].reshape(n_complete, T1).copy()
                shard_buf.append(new_rows)
                buf_rows   += n_complete
                total_rows += n_complete
                overflow    = stream[n_complete * T1:].copy()
                del stream
                _drain_buffer()    # write shards as they fill up
            else:
                overflow = stream.copy()

        new_rows = total_rows - rows_before
        dt_file  = _time.time() - t_file
        # Estimate MB held in shard buffer
        buf_mb = buf_rows * T1 * 4 / 1e6
        print(
            f"{label} {f_idx+1:3d}/{n_files} done   {fname}  "
            f"+{new_rows:>8,} rows  total {total_rows:>10,}  "
            f"buf {buf_rows:,} ({buf_mb:.0f} MB)  ({dt_file:.1f}s)",
            flush=True,
        )

    # Flush remaining rows as one final (partial) shard
    if buf_rows > 0:
        _flush_shard(np.concatenate(shard_buf))
        shard_buf = []
        buf_rows  = 0

    elapsed = _time.time() - t_start
    print(
        f"{label} COMPLETE — {n_files} files  {total_rows:,} rows  "
        f"{shard_idx} shards  {elapsed:.1f}s",
        flush=True,
    )
    return {"n_rows": total_rows, "n_shards": shard_idx, "elapsed": elapsed}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(files: list, n: int) -> list:
    """Split files into n roughly equal chunks."""
    n    = min(n, len(files))
    size = math.ceil(len(files) / n)
    return [files[i:i + size] for i in range(0, len(files), size)]


def _peak_ram_estimate(n_workers: int, rows_per_shard: int, T: int) -> str:
    T1           = T + 1
    shard_mb     = rows_per_shard * T1 * 4 / 1e6
    # File size is unknown upfront; give a formula
    return (
        f"Peak RAM ≈ n_workers × (largest_file_rows + rows_per_shard) × (T+1) × 4 bytes\n"
        f"         = {n_workers} × (largest_file_rows + {rows_per_shard:,}) × {T1} × 4\n"
        f"  shard buffer per worker : {shard_mb:.1f} MB  (fixed)\n"
        f"  file tokenization       : depends on file size (dominant term)\n"
        f"  -> reduce --workers or --rows-per-shard to cut memory\n"
        f"  -> actual file rows will be logged as each file completes"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretokenize and merge raw text parquets into flat tokenized shards"
    )
    parser.add_argument("--source-root",    required=True, help="root directory containing raw text parquets (searched recursively)")
    parser.add_argument("--output-root",    required=True, help="output directory for merged flat shards")
    parser.add_argument("--tokenizer",      default="Qwen/Qwen3.5-0.8B-Base", help="HF tokenizer path or model ID")
    parser.add_argument("--T",              type=int, default=4096, help="sequence length (each output row = T+1 tokens)")
    parser.add_argument("--rows-per-shard", type=int, default=4096, help="flush a shard every N rows; controls shard size and peak shard-buffer RAM")
    parser.add_argument("--rows-per-rg",    type=int, default=1024, help="parquet row-group size within each output shard")
    parser.add_argument("--workers",        type=int, default=8,    help="parallel worker processes")
    parser.add_argument("--seed",           type=int, default=42,   help="file-shuffle seed")
    parser.add_argument("--force",          action="store_true",    help="overwrite existing output")
    args = parser.parse_args()

    from nanoqwen35.dataset import list_all_parquet_files

    # ------------------------------------------------------------------
    # Discover and shuffle files
    all_files   = list_all_parquet_files(args.source_root)
    train_files = [f for f in all_files if "train" in os.path.basename(f)]
    val_files   = [f for f in all_files if "val"   in os.path.basename(f)]

    if not train_files:
        print(f"ERROR: no train parquet files found under {args.source_root}", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    rng.shuffle(train_files)                              # file-level shuffle
    rng.shuffle(val_files)

    T        = args.T
    T_plus_1 = T + 1

    os.makedirs(args.output_root, exist_ok=True)
    out_meta = os.path.join(args.output_root, "merged_metadata.json")
    if not args.force and os.path.exists(out_meta):
        print(f"Output already exists: {out_meta}\nUse --force to overwrite.")
        sys.exit(0)

    print(f"{'='*65}")
    print(f"pretokenize_and_merge")
    print(f"  source       : {args.source_root}")
    print(f"  output       : {args.output_root}")
    print(f"  tokenizer    : {args.tokenizer}")
    print(f"  T            : {T}")
    print(f"  rows_per_shard: {args.rows_per_shard:,}  ({args.rows_per_shard * T_plus_1 * 4 / 1e6:.1f} MB/shard)")
    print(f"  workers      : {args.workers}")
    print(f"  seed         : {args.seed}")
    print(f"  train files  : {len(train_files)}")
    print(f"  val files    : {len(val_files)}")
    print(f"\n{_peak_ram_estimate(args.workers, args.rows_per_shard, T)}")
    print(f"{'='*65}\n")

    # ------------------------------------------------------------------
    # Build tasks
    def make_tasks(files, split):
        chunks = _chunk(files, args.workers)
        return [
            {
                "files": chunk, "tokenizer_path": args.tokenizer, "T": T,
                "chunk_id": i, "split": split, "output_root": args.output_root,
                "rows_per_shard": args.rows_per_shard, "rows_per_rg": args.rows_per_rg,
            }
            for i, chunk in enumerate(chunks)
        ]

    train_tasks = make_tasks(train_files, "train")
    val_tasks   = make_tasks(val_files,   "val") if val_files else []
    all_tasks   = train_tasks + val_tasks
    n_workers   = min(args.workers, len(all_tasks))
    n_total     = len(all_tasks)

    print(
        f"Submitting {len(train_tasks)} train + {len(val_tasks)} val chunks "
        f"({n_workers} workers)...\n",
        flush=True,
    )

    train_rows = 0; train_shards = 0
    val_rows   = 0; val_shards   = 0
    n_done     = 0
    t_global   = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_tokenize_and_write_chunk, task): task for task in all_tasks}

        for fut in as_completed(futs):
            task = futs[fut]
            split, chunk_id = task["split"], task["chunk_id"]
            try:
                r = fut.result()
            except Exception as exc:
                print(f"FAILED [{split}|W{chunk_id:02d}]: {exc}", file=sys.stderr)
                raise

            n_done  += 1
            elapsed  = time.time() - t_global
            print(
                f">>> [{split}|W{chunk_id:02d}] done — {r['n_rows']:,} rows  "
                f"{r['n_shards']} shards  ({n_done}/{n_total} chunks, {elapsed:.0f}s)\n",
                flush=True,
            )
            if split == "train":
                train_rows   += r["n_rows"]
                train_shards += r["n_shards"]
            else:
                val_rows   += r["n_rows"]
                val_shards += r["n_shards"]

    # ------------------------------------------------------------------
    # Write metadata
    meta = {
        "T":                T,
        "tokenizer":        args.tokenizer,
        "source_root":      args.source_root,
        "num_train_rows":   train_rows,
        "num_train_shards": train_shards,
        "num_val_rows":     val_rows,
        "num_val_shards":   val_shards,
        "seed":             args.seed,
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    total_elapsed = time.time() - t_global
    print(f"\n{'='*65}")
    print(f"Done in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"  Train : {train_rows:,} rows  →  {train_shards} shards")
    print(f"  Val   : {val_rows:,} rows  →  {val_shards} shards")
    print(f"  Meta  : {out_meta}")
    if val_rows == 0:
        print("  (no val files found — val skipped)")


if __name__ == "__main__":
    main()
