"""
Pre-tokenization script: converts raw text parquet → packed token-ID parquet.

Each output row contains exactly T+1 token IDs. Documents are separated by EOS
tokens and the token stream is sliced at T+1 boundaries (standard GPT-style
pretraining packing — documents may be cut at sequence boundaries, but EOS
tokens always mark document boundaries within a sequence).

Output parquet schema:
    input_ids: fixed_size_list<int32>[T+1]

Run:
    python -m scripts.pretokenize \\
        --source-root /path/to/raw_data \\
        --output-root /path/to/tokenized_data \\
        --tokenizer Qwen/Qwen3.5-0.8B-Base \\
        --T 4096 --min-shards 32 --workers 13
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
from tokenizers import Tokenizer as HFTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_num_shards(total_rows: int, min_shards: int, rows_per_shard: int) -> int:
    """Return target shard count, honouring min_shards for file-level DDP sharding."""
    if total_rows == 0:
        return 0
    natural = math.ceil(total_rows / rows_per_shard)
    if total_rows < min_shards:
        return total_rows  # 1 row per shard; caller warns
    return max(min_shards, natural)


def _write_shard(path: str, rows_np: np.ndarray, T_plus_1: int, rows_per_rg: int) -> None:
    """Atomically write rows_np (shape N × T+1, dtype int32) to a parquet shard."""
    tmp = path + ".tmp"
    schema = pa.schema([("input_ids", pa.list_(pa.int32(), T_plus_1))])
    with pq.ParquetWriter(tmp, schema, compression="zstd", compression_level=3) as writer:
        for start in range(0, len(rows_np), rows_per_rg):
            chunk = rows_np[start : start + rows_per_rg]
            flat  = pa.array(chunk.ravel(), type=pa.int32())
            fsl   = pa.FixedSizeListArray.from_arrays(flat, T_plus_1)
            writer.write_table(pa.table({"input_ids": fsl}, schema=schema))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Worker (top-level so ProcessPoolExecutor can pickle it)
# ---------------------------------------------------------------------------

def tokenize_domain_split(task: dict) -> dict:
    """
    Tokenize and pack one (domain, split) pair into parquet shards.

    Algorithm:
    1. Stream source row-groups, tokenize with HF Rust tokenizer (encode_batch).
    2. Append EOS after every document before adding to the token stream.
    3. Slice stream into complete T+1 rows; keep sub-row overflow across row-groups.
    4. Flush a new shard whenever rows_per_shard rows accumulate.
    5. After all source files: if total_shards < min_shards (small domain),
       redistribute rows evenly across min_shards shards (loaded into memory —
       safe because the domain is small).
    """
    domain      = task["domain"]
    split       = task["split"]
    src_files   = task["source_files"]
    out_dir     = task["output_dir"]
    T           = task["T"]
    tok_path    = task["tokenizer_path"]
    min_shards  = task["min_shards"]
    rps         = task["rows_per_shard"]
    rows_per_rg = task["rows_per_rg"]
    force       = task["force"]

    T1  = T + 1
    tag = f"[{domain}/{split}]"
    os.makedirs(out_dir, exist_ok=True)

    # Resume: skip if per-split meta says all shards are already on disk
    meta_path = os.path.join(out_dir, f".{split}_meta.json")
    if not force and os.path.exists(meta_path):
        with open(meta_path) as f:
            m = json.load(f)
        if all(
            os.path.exists(os.path.join(out_dir, f"{split}_{i:04d}.parquet"))
            for i in range(m["num_shards"])
        ):
            print(f"  {tag} already done ({m['num_shards']} shards, {m['num_rows']:,} rows) — skip", flush=True)
            return {"domain": domain, "split": split, **m}

    # Load tokenizer (each worker gets its own copy; Rust thread pool is process-local)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    tok    = HFTokenizer.from_pretrained(tok_path)
    eos_id = tok.token_to_id("<|endoftext|>")
    assert eos_id is not None, "EOS token <|endoftext|> not found in tokenizer vocab"

    # ---- streaming tokenization ----
    overflow    = np.empty(0, dtype=np.int32)   # tokens that didn't fill last row
    row_chunks: list[np.ndarray] = []            # list of (n, T1) arrays not yet flushed
    chunk_rows  = 0                              # total rows in row_chunks
    shard_idx   = 0
    total_rows  = 0
    shard_paths: list[str] = []

    def flush_shard(rows_np: np.ndarray) -> None:
        nonlocal shard_idx, total_rows
        path = os.path.join(out_dir, f"{split}_{shard_idx:04d}.parquet")
        _write_shard(path, rows_np, T1, rows_per_rg)
        shard_paths.append(path)
        total_rows += len(rows_np)
        shard_idx  += 1

    def drain() -> None:
        """Write complete rps-sized shards until chunk buffer drops below rps rows."""
        nonlocal row_chunks, chunk_rows
        while chunk_rows >= rps:
            all_rows = np.concatenate(row_chunks, axis=0)
            flush_shard(all_rows[:rps])
            rest       = all_rows[rps:]
            row_chunks = [rest] if len(rest) else []
            chunk_rows = len(rest)

    for filepath in src_files:
        pf = pq.ParquetFile(filepath)
        for rg_i in range(pf.num_row_groups):
            rg    = pf.read_row_group(rg_i, columns=["text"])
            texts = rg.column("text").to_pylist()
            encs  = tok.encode_batch(texts, add_special_tokens=False)

            # Build numpy token stream: doc tokens + EOS after every document
            pieces: list[np.ndarray] = [overflow]
            for enc in encs:
                if enc.ids:
                    pieces.append(np.array(enc.ids, dtype=np.int32))
                pieces.append(np.array([eos_id], dtype=np.int32))
            stream = np.concatenate(pieces) if len(pieces) > 1 else overflow.copy()

            n_complete = len(stream) // T1
            if n_complete > 0:
                new_rows = stream[: n_complete * T1].reshape(n_complete, T1).copy()
                row_chunks.append(new_rows)
                chunk_rows += n_complete
                overflow    = stream[n_complete * T1 :].copy()
                drain()
            else:
                overflow = stream.copy()

    # Flush remaining rows as the final (possibly partial) shard
    if chunk_rows > 0:
        flush_shard(np.concatenate(row_chunks, axis=0))

    if total_rows == 0:
        print(f"  {tag} WARNING: 0 rows produced (empty domain or split)", flush=True)
        return {"domain": domain, "split": split, "num_rows": 0, "num_shards": 0}

    # Redistribute into min_shards shards if natural count is too low (small domains)
    if shard_idx < min_shards:
        target = min(total_rows, min_shards)
        print(
            f"  {tag} redistributing {total_rows} rows into {target} shards"
            f" (natural={shard_idx} < min_shards={min_shards})",
            flush=True,
        )
        # Reload all rows — safe: small domain means small total size
        all_chunks: list[np.ndarray] = []
        for p in shard_paths:
            pf2 = pq.ParquetFile(p)
            for rg_i in range(pf2.num_row_groups):
                col = pf2.read_row_group(rg_i, columns=["input_ids"]).column("input_ids")
                c   = col.combine_chunks() if col.num_chunks > 1 else col.chunks[0]
                flat_np = c.values.to_numpy(zero_copy_only=False)
                all_chunks.append(flat_np.reshape(len(c), T1))
            os.remove(p)
        all_rows_np = np.concatenate(all_chunks, axis=0)
        for i, sub in enumerate(np.array_split(all_rows_np, target)):
            if len(sub) == 0:
                continue
            _write_shard(os.path.join(out_dir, f"{split}_{i:04d}.parquet"), sub, T1, rows_per_rg)
        shard_idx = target
        if target < min_shards:
            print(
                f"  {tag} WARNING: only {target} shards possible "
                f"(total_rows={total_rows} < min_shards={min_shards}). "
                f"world_size > {target} will use row-group fallback for this domain.",
                flush=True,
            )

    result = {"num_rows": total_rows, "num_shards": shard_idx}
    with open(meta_path, "w") as f:
        json.dump(result, f)
    print(f"  {tag} done: {total_rows:,} rows → {shard_idx} shards", flush=True)
    return {"domain": domain, "split": split, **result}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-tokenize parquet dataset for zero-overhead distributed training"
    )
    parser.add_argument("--source-root",    required=True, help="Root dir of raw text parquets (domain subdirectories)")
    parser.add_argument("--output-root",    required=True, help="Output root for tokenized parquet shards")
    parser.add_argument("--tokenizer",      default="Qwen/Qwen3.5-0.8B-Base", help="HF tokenizer path or model ID")
    parser.add_argument("--T",              type=int, default=4096, help="Sequence length (each row = T+1 tokens)")
    parser.add_argument("--min-shards",     type=int, default=32,   help="Min shards per (domain, split) — guarantees file-level sharding for world_size ≤ min-shards")
    parser.add_argument("--rows-per-shard", type=int, default=4096, help="Target rows per shard file")
    parser.add_argument("--rows-per-rg",    type=int, default=1024, help="Row-groups per shard (affects read granularity)")
    parser.add_argument("--workers",        type=int, default=8,    help="Parallel worker processes")
    parser.add_argument("--force",          action="store_true",    help="Re-process even if output already exists")
    args = parser.parse_args()

    # Import here — avoids importing project code inside worker processes before fork
    from nanoqwen35.dataset import list_parquet_files_by_domain

    domain_file_map = list_parquet_files_by_domain(args.source_root)
    if not domain_file_map:
        print(f"ERROR: no domains found under {args.source_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Domains ({len(domain_file_map)}): {', '.join(sorted(domain_file_map))}")
    print(f"T={args.T} | min_shards={args.min_shards} | rows_per_shard={args.rows_per_shard} | workers={args.workers}")
    os.makedirs(args.output_root, exist_ok=True)

    tasks: list[dict] = []
    for domain, files in sorted(domain_file_map.items()):
        out_dir = os.path.join(args.output_root, domain)
        for split in ("train", "val"):
            split_files = sorted(f for f in files if split in os.path.basename(f))
            if split_files:
                tasks.append({
                    "domain":         domain,
                    "split":          split,
                    "source_files":   split_files,
                    "output_dir":     out_dir,
                    "T":              args.T,
                    "tokenizer_path": args.tokenizer,
                    "min_shards":     args.min_shards,
                    "rows_per_shard": args.rows_per_shard,
                    "rows_per_rg":    args.rows_per_rg,
                    "force":          args.force,
                })

    print(f"Tasks: {len(tasks)} (domain × split)\n")

    domain_results: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        futs = {pool.submit(tokenize_domain_split, t): t for t in tasks}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                r = fut.result()
                domain_results.setdefault(r["domain"], {})[r["split"]] = {
                    "num_rows":   r["num_rows"],
                    "num_shards": r["num_shards"],
                }
            except Exception as exc:
                print(f"FAILED {t['domain']}/{t['split']}: {exc}", file=sys.stderr)
                raise

    metadata = {
        "T":              args.T,
        "tokenizer":      args.tokenizer,
        "source_root":    args.source_root,
        "min_shards":     args.min_shards,
        "rows_per_shard": args.rows_per_shard,
        "domains":        domain_results,
    }
    meta_path = os.path.join(args.output_root, "pretokenize_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    total_train_rows   = sum(v.get("train", {}).get("num_rows",   0) for v in domain_results.values())
    total_train_shards = sum(v.get("train", {}).get("num_shards", 0) for v in domain_results.values())
    print(f"\n{'='*60}")
    print(f"Pretokenize complete.")
    print(f"  Train rows  : {total_train_rows:,}")
    print(f"  Train shards: {total_train_shards:,}")
    print(f"  Metadata    : {meta_path}")
    if total_train_shards > 0:
        print(f"\nAt world_size=8: ~{total_train_shards // 8:,} files per rank (train, average across domains)")


if __name__ == "__main__":
    main()
