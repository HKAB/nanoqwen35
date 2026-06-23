"""
Count the total number of pre-tokenized tokens in a parquet dataset.

Each parquet stores an "input_ids" column of fixed-length packed blocks (one
block per row), so the total token count is simply:

    total_tokens = sum(num_rows over all files) * tokens_per_row

Only parquet *metadata* is read (no token data is decoded), so this is fast even
for hundreds of files. Reads are parallelized across worker processes.

Usage:
    python -m scripts.count_tokens --data-dir /path/to/pretokenized
    python -m scripts.count_tokens --data-dir /path/to/pretokenized --tokens-per-row 8192 --workers 16
"""

import os
import glob
import argparse
import multiprocessing as mp

import pyarrow.parquet as pq
from tqdm import tqdm


def find_parquet_files(data_dir):
    """Sorted list of all .parquet files under data_dir (recursive, skip .tmp)."""
    files = glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True)
    return sorted(f for f in files if not f.endswith(".tmp"))


def detect_tokens_per_row(path):
    """Read a single row to determine the packed block length (tokens per row)."""
    pf = pq.ParquetFile(path)
    batch = next(pf.iter_batches(batch_size=1, columns=["input_ids"]))
    return len(batch.column("input_ids")[0].as_py())


def get_num_rows(path):
    """Return (path, num_rows, error). Reads metadata only — no data decode."""
    try:
        return path, pq.ParquetFile(path).metadata.num_rows, None
    except Exception as e:
        return path, 0, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Count total pre-tokenized tokens in a parquet dataset."
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Directory containing pretokenized .parquet files (searched recursively).",
    )
    parser.add_argument(
        "--tokens-per-row", type=int, default=None,
        help="Tokens per row (packed block size). If omitted, auto-detected from the first file.",
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(),
        help="Number of parallel workers for reading parquet metadata.",
    )
    args = parser.parse_args()

    files = find_parquet_files(args.data_dir)
    if not files:
        print(f"No .parquet files found under {args.data_dir}")
        return
    print(f"Found {len(files):,} parquet files under {args.data_dir}")

    tokens_per_row = args.tokens_per_row
    if tokens_per_row is None:
        tokens_per_row = detect_tokens_per_row(files[0])
        print(f"Auto-detected tokens per row: {tokens_per_row:,} (from {os.path.basename(files[0])})")

    total_rows = 0
    errors = []
    with mp.Pool(args.workers) as pool:
        for path, num_rows, err in tqdm(
            pool.imap_unordered(get_num_rows, files),
            total=len(files), desc="Reading metadata",
        ):
            if err is not None:
                errors.append((path, err))
            else:
                total_rows += num_rows

    total_tokens = total_rows * tokens_per_row

    print()
    print(f"Files read:     {len(files) - len(errors):,} / {len(files):,}")
    print(f"Total rows:     {total_rows:,}")
    print(f"Tokens per row: {tokens_per_row:,}")
    print(f"Total tokens:   {total_tokens:,}  ({total_tokens / 1e9:.3f}B)")

    if errors:
        print(f"\n{len(errors)} file(s) failed to read:")
        for path, err in errors[:10]:
            print(f"  {os.path.basename(path)}: {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")


if __name__ == "__main__":
    main()
