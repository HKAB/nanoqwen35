import os
import glob
import sqlite3
import orjson
import logging
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
import concurrent.futures
from transformers import AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pretokenize jsonl data into packed parquet shards."
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Glob pattern for input .jsonl files (e.g. '/path/**/*.jsonl').",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory where parquet shards, state DB and logs are written.",
    )
    parser.add_argument(
        "--tokenizer", required=True,
        help="Tokenizer model id or local path (passed to AutoTokenizer).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=8192,
        help="Number of tokens per packed block (default: 8192).",
    )
    parser.add_argument(
        "--cpu-count", type=int, default=16,
        help="Number of worker processes / chunks per file (default: 16).",
    )
    parser.add_argument(
        "--write-threshold", type=int, default=20000,
        help="Number of blocks buffered before flushing a parquet file (default: 20000).",
    )
    parser.add_argument(
        "--ignore", nargs="*", default=[
            "vi-finewiki-082025",
            "vi-wiki",
            "history_dedup_fixed_ocr",
            "history",
            "multilingual",
            "safety",
        ],
        help="Substrings; files whose path contains any of them are skipped.",
    )
    parser.add_argument(
        "--hf-home", default=None,
        help="Optional override for the HF_HOME cache directory.",
    )
    return parser.parse_args()


def init_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chunks
                 (chunk_id TEXT PRIMARY KEY, filename TEXT, start_byte INT, end_byte INT, status TEXT)''')
    conn.commit()
    conn.close()


def register_chunks(db_path, filename, file_size, cpu_count):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    chunk_size = file_size // cpu_count

    for i in range(cpu_count):
        start = i * chunk_size
        end = file_size if i == cpu_count - 1 else (i + 1) * chunk_size
        chunk_id = f"{os.path.basename(filename)}_part_{i}"
        c.execute("INSERT OR IGNORE INTO chunks (chunk_id, filename, start_byte, end_byte, status) VALUES (?, ?, ?, ?, 'PENDING')",
                  (chunk_id, filename, start, end))
    conn.commit()
    conn.close()


def get_pending_chunks(db_path, filename):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT chunk_id, start_byte, end_byte FROM chunks WHERE filename = ? AND status = 'PENDING'", (filename,))
    pending = c.fetchall()
    conn.close()
    return pending


def update_chunk_status(db_path, chunk_id, status):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("UPDATE chunks SET status = ? WHERE chunk_id = ?", (status, chunk_id))
    conn.commit()
    conn.close()


def process_chunk(chunk_id, filename, start_byte, end_byte,
                  tokenizer_id, output_dir, chunk_size, write_threshold):
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
        eos_id = tokenizer.eos_token_id

        local_buffer = []
        global_blocks = []
        file_counter = 0

        with open(filename, "rb") as f:
            f.seek(start_byte)
            if start_byte != 0:
                f.readline()

            while f.tell() < end_byte:
                line = f.readline()
                if not line: break

                try:
                    doc = orjson.loads(line)
                    text = doc.get('text', '')
                    if not text:
                        continue

                    tokens = tokenizer(text, add_special_tokens=False).input_ids + [eos_id]
                    local_buffer.extend(tokens)

                    while len(local_buffer) >= chunk_size:
                        global_blocks.append(local_buffer[:chunk_size])
                        local_buffer = local_buffer[chunk_size:]

                    if len(global_blocks) >= write_threshold:
                        out_path = os.path.join(output_dir, f"{chunk_id}_{file_counter}.parquet")
                        table = pa.Table.from_arrays([pa.array(global_blocks)], names=["input_ids"])
                        pq.write_table(table, out_path)
                        global_blocks = []
                        file_counter += 1

                except Exception:
                    continue

        if global_blocks:
            out_path = os.path.join(output_dir, f"{chunk_id}_{file_counter}.parquet")
            table = pa.Table.from_arrays([pa.array(global_blocks)], names=["input_ids"])
            pq.write_table(table, out_path)

        return chunk_id, "DONE", local_buffer

    except Exception as e:
        return chunk_id, "ERROR", str(e)


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.hf_home:
        os.environ['HF_HOME'] = args.hf_home

    db_path = os.path.join(args.output_dir, "chunk_state.db")
    log_file = os.path.join(args.output_dir, "pipeline.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )

    init_db(db_path)
    logger.info("Starting Tokenization Pipeline")

    # Use recursive globbing to find all jsonl files in subdirectories
    all_filename = glob.glob(args.input_dir, recursive=True)
    all_filename = [f for f in all_filename if not any(ignore in f for ignore in args.ignore)]

    if not all_filename:
        logger.warning(f"No .jsonl files found matching {args.input_dir}")
        return

    logger.info(f"Found {len(all_filename)} total files to process.")

    global_leftover_buffer = []

    for filename in all_filename:
        file_size = os.path.getsize(filename)
        logger.info(f"Processing {filename} ({file_size / (1024**3):.2f} GB)")

        register_chunks(db_path, filename, file_size, args.cpu_count)
        pending_chunks = get_pending_chunks(db_path, filename)

        if not pending_chunks:
            logger.info(f"All chunks for {filename} already processed. Skipping.")
            continue

        logger.info(f"Executing {len(pending_chunks)} pending chunks out of {args.cpu_count}...")

        with concurrent.futures.ProcessPoolExecutor(max_workers=args.cpu_count) as executor:
            futures = {
                executor.submit(
                    process_chunk, c_id, filename, start, end,
                    args.tokenizer, args.output_dir, args.chunk_size, args.write_threshold,
                ): c_id
                for c_id, start, end in pending_chunks
            }

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Chunks ({os.path.basename(filename)})"):
                chunk_id, status, result = future.result()

                if status == "DONE":
                    update_chunk_status(db_path, chunk_id, "DONE")
                    global_leftover_buffer.extend(result)
                else:
                    logger.error(f"Error in {chunk_id}: {result}")
                    update_chunk_status(db_path, chunk_id, "ERROR")

    # Process Master Leftovers
    if global_leftover_buffer:
        logger.info(f"Aggregating and packing global leftovers ({len(global_leftover_buffer)} total tokens)...")
        leftover_blocks = []

        while len(global_leftover_buffer) >= args.chunk_size:
            leftover_blocks.append(global_leftover_buffer[:args.chunk_size])
            global_leftover_buffer = global_leftover_buffer[args.chunk_size:]

        if leftover_blocks:
            out_path = os.path.join(args.output_dir, "aggregated_master_leftovers.parquet")
            table = pa.Table.from_arrays([pa.array(leftover_blocks)], names=["input_ids"])
            pq.write_table(table, out_path)
            logger.info(f"Saved {len(leftover_blocks)} master leftover blocks.")

        if global_leftover_buffer:
            logger.info(f"Dropped final absolute remainder of {len(global_leftover_buffer)} tokens.")

    logger.info("Pretokenized Dataset Generation Complete.")


if __name__ == "__main__":
    main()
