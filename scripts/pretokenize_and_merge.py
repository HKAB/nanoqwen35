"""
Pretokenize and merge: raw text parquets → flat tokenized shards.

Strategy:
  1. Pre-scan all files (metadata only, no data) — skips empty / corrupt / missing-column files.
  2. Shuffle the valid FILE LIST (file-level shuffle ≈ row-level shuffle when files are equal-sized).
  3. Chunk the shuffled list across --workers processes.
  4. Each worker streams through its files, tokenizing and writing output shards CONTINUOUSLY —
     as soon as --rows-per-shard rows accumulate, a shard is flushed and memory is freed.

Peak RAM per worker:
  ≈ (rows_in_largest_single_file + rows_per_shard) × (T+1) × 4 bytes
  rows_in_largest_file ≈ file_text_tokens / (T+1)

  example: file with 24 000 rows at T=4096 → 393 MB per worker
  With 8 workers: ~8 × (393 + 67) MB ≈ 3.7 GB peak

  Tune: lower --workers or --rows-per-shard to reduce peak.

Usage:
    python -m scripts.pretokenize_and_merge \\
        --source-root /path/to/vi_en_parquet_v1 \\
        --output-root /path/to/vi_en_merged_v1 \\
        --tokenizer   /path/to/Qwen3.5-0.8B-Base \\
        --T 4096 --rows-per-shard 4096 --workers 8

Shard naming: {split}_{chunk_id:04d}_{local_shard:06d}.parquet
"""

import os
import sys
import json
import math
import time
import argparse
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Pre-validation (runs in main process, metadata-only — no data read)
# ---------------------------------------------------------------------------

def _scan_files(files: list, split: str) -> list:
    """
    Quick metadata scan of every parquet file.
    Skips files that are empty, missing the 'text' column, or unreadable.
    Returns the list of valid file paths.
    """
    valid   = []
    n_skip  = 0
    print(f"[scan:{split}] checking {len(files)} files...", flush=True)
    for path in files:
        basename = os.path.basename(path)
        try:
            pf  = pq.ParquetFile(path)
            meta = pf.metadata
            if meta.num_rows == 0:
                print(f"  SKIP empty       : {basename}", flush=True)
                n_skip += 1
                continue
            col_names = pf.schema_arrow.names
            if "text" not in col_names:
                print(f"  SKIP no 'text'   : {basename}  (columns: {col_names})", flush=True)
                n_skip += 1
                continue
            valid.append(path)
        except Exception as exc:
            print(f"  SKIP corrupt      : {basename}  ({exc})", flush=True)
            n_skip += 1

    print(
        f"[scan:{split}] {len(valid)} valid, {n_skip} skipped\n",
        flush=True,
    )
    return valid


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


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _tokenize_and_write_chunk(task: dict) -> dict:
    """
    Tokenize files in order and write shards continuously.

    Each file is wrapped in try/except — bad files are logged and skipped,
    the worker continues with the remaining files.

    Peak RAM ≈ current_file_rows + shard_buffer_rows (one file at a time).
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
        return {"n_rows": 0, "n_shards": 0, "n_errors": 0, "elapsed": 0.0}

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    tok    = HFTokenizer.from_pretrained(tok_path)
    eos_id = tok.token_to_id("<|endoftext|>")
    assert eos_id is not None, "EOS token <|endoftext|> not found in tokenizer vocab"

    # State persisted across files
    overflow:   np.ndarray       = np.empty(0, dtype=np.int32)
    shard_buf:  list[np.ndarray] = []
    buf_rows:   int              = 0
    shard_idx:  int              = 0
    total_rows: int              = 0
    n_errors:   int              = 0
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
        nonlocal shard_buf, buf_rows
        while buf_rows >= rows_per_shard:
            all_buf  = np.concatenate(shard_buf)
            to_write = all_buf[:rows_per_shard].copy()
            leftover = all_buf[rows_per_shard:]
            del all_buf
            _flush_shard(to_write)
            del to_write
            shard_buf = [leftover] if len(leftover) > 0 else []
            buf_rows  = len(leftover)

    for f_idx, filepath in enumerate(files):
        fname  = os.path.basename(filepath)
        t_file = _time.time()
        rows_before = total_rows
        print(f"{label} {f_idx+1:3d}/{n_files} start  {fname}", flush=True)

        # Save overflow before this file; restored on error so the next file's
        # token stream doesn't get contaminated by a partial failed read.
        file_overflow = overflow.copy()

        try:
            pf    = pq.ParquetFile(filepath)
            n_rgs = pf.num_row_groups

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
                    _drain_buffer()
                else:
                    overflow = stream.copy()

        except Exception as exc:
            n_errors += 1
            # Restore overflow so the next file's stream starts clean.
            # Rows already in shard_buf or flushed to disk from this file are
            # left in place — they are valid tokenized data. Only the overflow
            # linkage matters for correctness across file boundaries.
            overflow = file_overflow
            print(
                f"{label} {f_idx+1:3d}/{n_files} ERROR  {fname}  "
                f"— {type(exc).__name__}: {exc}  (file skipped)",
                flush=True,
            )
            continue

        new_rows = total_rows - rows_before
        dt_file  = _time.time() - t_file
        buf_mb   = buf_rows * T1 * 4 / 1e6
        print(
            f"{label} {f_idx+1:3d}/{n_files} done   {fname}  "
            f"+{new_rows:>8,} rows  total {total_rows:>10,}  "
            f"buf {buf_rows:,} ({buf_mb:.0f} MB)  ({dt_file:.1f}s)",
            flush=True,
        )

    # Flush remaining rows as the final (partial) shard
    if buf_rows > 0:
        _flush_shard(np.concatenate(shard_buf))

    elapsed = _time.time() - t_start
    print(
        f"{label} COMPLETE — {n_files} files  {total_rows:,} rows  "
        f"{shard_idx} shards  {n_errors} errors  {elapsed:.1f}s",
        flush=True,
    )
    return {"n_rows": total_rows, "n_shards": shard_idx, "n_errors": n_errors, "elapsed": elapsed}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(files: list, n: int) -> list:
    n    = min(n, len(files))
    size = math.ceil(len(files) / n)
    return [files[i:i + size] for i in range(0, len(files), size)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretokenize and merge raw text parquets into flat tokenized shards"
    )
    parser.add_argument("--source-root",    required=True)
    parser.add_argument("--output-root",    required=True)
    parser.add_argument("--tokenizer",      default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--T",              type=int, default=4096,  help="sequence length")
    parser.add_argument("--rows-per-shard", type=int, default=4096,  help="rows per output shard (controls shard size and shard-buffer RAM)")
    parser.add_argument("--rows-per-rg",    type=int, default=1024,  help="parquet row-group size within each shard")
    parser.add_argument("--workers",        type=int, default=8,     help="parallel worker processes — lower to reduce peak RAM")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--force",          action="store_true",     help="overwrite existing output")
    parser.add_argument("--validate-only",  action="store_true",     help="scan files and exit without tokenizing")
    args = parser.parse_args()

    from nanoqwen35.dataset import list_all_parquet_files

    # ------------------------------------------------------------------
    # Discover files
    all_files   = list_all_parquet_files(args.source_root)
    train_files = [f for f in all_files if "train" in os.path.basename(f)]
    val_files   = [f for f in all_files if "val"   in os.path.basename(f)]

    if not train_files:
        print(f"ERROR: no train parquet files found under {args.source_root}", file=sys.stderr)
        sys.exit(1)

    T        = args.T
    T_plus_1 = T + 1
    shard_mb = args.rows_per_shard * T_plus_1 * 4 / 1e6

    print(f"{'='*65}")
    print(f"pretokenize_and_merge")
    print(f"  source        : {args.source_root}")
    print(f"  output        : {args.output_root}")
    print(f"  tokenizer     : {args.tokenizer}")
    print(f"  T             : {T}")
    print(f"  rows_per_shard: {args.rows_per_shard:,}  ({shard_mb:.1f} MB/shard)")
    print(f"  workers       : {args.workers}")
    print(f"  train files   : {len(train_files)}")
    print(f"  val files     : {len(val_files)}")
    print(f"\nPeak RAM estimate:")
    print(f"  per worker ≈ largest_file_rows × (T+1) × 4 B  +  {shard_mb:.0f} MB shard buffer")
    print(f"  total      ≈ {args.workers} × peak_per_worker")
    print(f"  e.g. 24k-row files → {args.workers} × (393 + {shard_mb:.0f}) MB ≈ "
          f"{args.workers * (393 + shard_mb):.0f} MB")
    print(f"  actual file rows are logged after each file completes")
    print(f"  -> lower --workers to reduce peak RAM")
    print(f"{'='*65}\n")

    # ------------------------------------------------------------------
    # Pre-scan: skip empty/corrupt/wrong-schema files
    train_files = _scan_files(train_files, "train")
    val_files   = _scan_files(val_files,   "val") if val_files else []

    if not train_files:
        print("ERROR: no valid train files after scan.", file=sys.stderr)
        sys.exit(1)

    if args.validate_only:
        print("--validate-only: scan complete, exiting.")
        sys.exit(0)

    os.makedirs(args.output_root, exist_ok=True)
    out_meta = os.path.join(args.output_root, "merged_metadata.json")
    if not args.force and os.path.exists(out_meta):
        print(f"Output already exists: {out_meta}\nUse --force to overwrite.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Shuffle file lists (file-level shuffle)
    rng = np.random.default_rng(args.seed)
    train_files = list(rng.permutation(train_files))
    val_files   = list(rng.permutation(val_files)) if val_files else []

    # ------------------------------------------------------------------
    # Build tasks
    def make_tasks(files, split):
        return [
            {
                "files": chunk, "tokenizer_path": args.tokenizer, "T": T,
                "chunk_id": i, "split": split, "output_root": args.output_root,
                "rows_per_shard": args.rows_per_shard, "rows_per_rg": args.rows_per_rg,
            }
            for i, chunk in enumerate(_chunk(files, args.workers))
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

    train_rows = 0;  train_shards = 0;  train_errors = 0
    val_rows   = 0;  val_shards   = 0;  val_errors   = 0
    n_done     = 0
    t_global   = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_tokenize_and_write_chunk, task): task for task in all_tasks}

        for fut in as_completed(futs):
            task  = futs[fut]
            split = task["split"]
            cid   = task["chunk_id"]

            try:
                r = fut.result()
            except concurrent.futures.process.BrokenProcessPool:
                print(f"\n{'!'*65}", file=sys.stderr)
                print(f"FATAL: worker [{split}|W{cid:02d}] was killed by the OS.", file=sys.stderr)
                print(f"Most likely cause: OOM (out of memory).", file=sys.stderr)
                print(f"\nFiles assigned to this worker:", file=sys.stderr)
                for f in task["files"]:
                    print(f"  {os.path.basename(f)}", file=sys.stderr)
                print(f"\nSuggestions to fix:", file=sys.stderr)
                print(f"  1. Lower --workers (currently {args.workers}) — reduces parallel RAM", file=sys.stderr)
                print(f"  2. Lower --rows-per-shard (currently {args.rows_per_shard}) — reduces shard buffer", file=sys.stderr)
                print(f"  3. Run with --validate-only first to catch corrupt files", file=sys.stderr)
                print(f"  4. Check system memory: free -h", file=sys.stderr)
                print(f"{'!'*65}\n", file=sys.stderr)
                raise
            except Exception as exc:
                print(f"\nFATAL [{split}|W{cid:02d}]: {type(exc).__name__}: {exc}", file=sys.stderr)
                print(f"Files: {[os.path.basename(f) for f in task['files']]}", file=sys.stderr)
                raise

            n_done  += 1
            elapsed  = time.time() - t_global
            print(
                f">>> [{split}|W{cid:02d}] done — {r['n_rows']:,} rows  "
                f"{r['n_shards']} shards  {r['n_errors']} file-errors  "
                f"({n_done}/{n_total} chunks, {elapsed:.0f}s)\n",
                flush=True,
            )
            if split == "train":
                train_rows   += r["n_rows"];  train_shards += r["n_shards"];  train_errors += r["n_errors"]
            else:
                val_rows     += r["n_rows"];  val_shards   += r["n_shards"];  val_errors   += r["n_errors"]

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
    print(f"  Train : {train_rows:,} rows  {train_shards} shards  ({train_errors} file errors)")
    print(f"  Val   : {val_rows:,} rows  {val_shards} shards  ({val_errors} file errors)")
    print(f"  Meta  : {out_meta}")
    if train_errors + val_errors > 0:
        print(f"\n  WARNING: {train_errors + val_errors} files were skipped due to errors.")
        print(f"  Grep the output above for 'ERROR' to see which files and why.")


if __name__ == "__main__":
    main()
