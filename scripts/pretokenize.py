import os
import glob
import bisect
import sqlite3
import logging
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
import concurrent.futures
from transformers import AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SFT helpers (ShareGPT -> messages -> render -> smart chunk -> neat packing)
# ---------------------------------------------------------------------------

# ShareGPT "from" tags -> chat-template roles. tool/observation results are
# masked (non-supervised) exactly like user turns.
SHAREGPT_ROLE_MAP = {
    "human": "user", "user": "user",
    "gpt": "assistant", "assistant": "assistant", "function_call": "assistant",
    "system": "system",
    "tool": "tool", "observation": "tool", "function_response": "tool",
}


def sharegpt_to_messages(doc):
    """Convert one ShareGPT record into the {"role","content"} messages format.

    Accepts the common ShareGPT layout ({"conversations":[{"from","value"}, ...]})
    and an optional top-level "system" field. Returns a messages list, or None if
    there is nothing usable.
    """
    convs = doc.get("conversations")
    if not convs:
        return None

    messages = []
    sys_field = doc.get("system")
    if isinstance(sys_field, str) and sys_field.strip():
        messages.append({"role": "system", "content": sys_field})

    for turn in convs:
        role = SHAREGPT_ROLE_MAP.get(turn.get("from"))
        content = turn.get("value")
        if role is None or content is None:
            continue
        # A system turn is only valid at the very beginning of the conversation.
        if role == "system" and messages:
            continue
        messages.append({"role": role, "content": content})

    return messages or None


def smart_chunk_conversation(ids, mask, boundaries, L, policy="smart_chunk"):
    """Split a rendered conversation so every piece fits in L tokens.

    Returns (items, dropped) where items is a list of (ids, mask) pieces and
    dropped is 1 if the whole datapoint was discarded, else 0.

    - len(ids) <= L                -> single piece, nothing to do.
    - policy == "drop"             -> discard the whole conversation.
    - policy == "smart_chunk"      -> cut only at user-turn boundaries
      (boundaries[1:]; the leading system header stays with the first piece),
      greedily grouping complete turns into <= L pieces. If a single turn
      segment alone exceeds L, the whole datapoint is dropped.
    """
    if len(ids) <= L:
        return [(ids, mask)], 0
    if policy == "drop":
        return [], 1

    # Segment edges: never cut before the first user turn, always cut at the rest.
    edges = [0] + [b for b in boundaries[1:]] + [len(ids)]
    seg_lens = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
    if any(sl > L for sl in seg_lens):
        return [], 1  # an indivisible turn is too long -> drop (burden, per spec)

    # Greedily pack consecutive segments into pieces of <= L tokens.
    items = []
    piece_start = edges[0]
    acc = 0
    for i, sl in enumerate(seg_lens):
        if acc + sl > L:
            items.append((ids[piece_start:edges[i]], mask[piece_start:edges[i]]))
            piece_start = edges[i]
            acc = sl
        else:
            acc += sl
    items.append((ids[piece_start:edges[-1]], mask[piece_start:edges[-1]]))
    return items, 0


def bfd_pack(items, L, pad_id):
    """Best-fit-decreasing knapsack packing of (ids, mask) items into L-length bins.

    Items are sorted by length descending and each is placed into the open bin with
    the smallest remaining capacity that still fits (best-fit), opening a new bin
    when none fits. Every returned bin is padded to exactly L with pad_id (mask 0),
    and the trailing pad is recorded as its own segment so real tokens never attend
    to padding.

    Returns a list of (input_ids[L], loss_mask[L], seg_lens) tuples.
    """
    items = sorted(items, key=lambda im: len(im[0]), reverse=True)

    bins = []          # each: {"ids": [...], "mask": [...], "segs": [...], "rem": int}
    rem_keys = []      # remaining capacities of open bins, kept sorted (parallel to rem_bins)
    rem_bins = []      # bin indices, parallel to rem_keys

    for ids, mask in items:
        s = len(ids)
        if s == 0:
            continue
        pos = bisect.bisect_left(rem_keys, s)  # smallest remaining capacity >= s
        if pos < len(rem_keys):
            b = rem_bins.pop(pos)
            rem_keys.pop(pos)
        else:
            b = len(bins)
            bins.append({"ids": [], "mask": [], "segs": [], "rem": L})
        bn = bins[b]
        bn["ids"].extend(ids)
        bn["mask"].extend(mask)
        bn["segs"].append(s)
        bn["rem"] -= s
        if bn["rem"] > 0:
            ip = bisect.bisect_left(rem_keys, bn["rem"])
            rem_keys.insert(ip, bn["rem"])
            rem_bins.insert(ip, b)

    out = []
    for bn in bins:
        pad = L - len(bn["ids"])
        if pad > 0:
            bn["ids"].extend([pad_id] * pad)
            bn["mask"].extend([0] * pad)
            bn["segs"].append(pad)
        out.append((bn["ids"], bn["mask"], bn["segs"]))
    return out


def _write_sft_blocks(blocks, out_path):
    """Write packed SFT blocks to a zstd-compressed parquet with 3 columns."""
    table = pa.table(
        {
            "input_ids": pa.array([b[0] for b in blocks], type=pa.list_(pa.int32())),
            "loss_mask": pa.array([b[1] for b in blocks], type=pa.list_(pa.uint8())),
            "seq_lens":  pa.array([b[2] for b in blocks], type=pa.list_(pa.int32())),
        }
    )
    pq.write_table(table, out_path, compression="zstd")


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
        "--mode", choices=["pretrain", "sft"], default="pretrain",
        help="pretrain: pack plain 'text' into fixed token blocks. "
             "sft: render ShareGPT conversations and neat-pack with best-fit knapsack (default: pretrain).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=8192,
        help="[pretrain] Number of tokens per packed block (default: 8192).",
    )
    parser.add_argument(
        "--seq-len", type=int, default=2049,
        help="[sft] Stored block length L (bin capacity). Train --max-seq-len must equal L-1 (default: 2049).",
    )
    parser.add_argument(
        "--pack-chunk-size", type=int, default=50000,
        help="[sft] Conversations buffered before each best-fit knapsack pass (default: 50000).",
    )
    parser.add_argument(
        "--long-doc", choices=["smart_chunk", "drop"], default="smart_chunk",
        help="[sft] Policy for conversations longer than seq-len: smart_chunk (split at turn "
             "boundaries) or drop (default: smart_chunk).",
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
        import orjson
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


def process_chunk_sft(chunk_id, filename, start_byte, end_byte,
                      tokenizer_id, output_dir, seq_len, pack_chunk_size,
                      long_doc, write_threshold):
    """SFT worker: read a jsonl byte-range of ShareGPT records, render + smart-chunk
    them, and neat-pack (best-fit) into seq_len-length blocks written as parquet.

    Packing is done per pack_chunk_size buffer so memory stays bounded; the last
    (under-filled) bin of each buffer is simply padded. Returns per-worker stats.
    """
    try:
        import orjson
        from nanoqwen35.tokenizer import get_tokenizer
        tokenizer = get_tokenizer(tokenizer_id)
        pad_id = tokenizer.get_bos_token_id()
        # Generous per-conversation cap; smart_chunk handles anything longer.
        render_max = seq_len * 16

        item_buffer = []          # list of (ids, mask) awaiting packing
        block_buffer = []         # list of packed (ids, mask, segs) awaiting flush
        file_counter = 0
        n_docs = n_dropped = n_empty = n_blocks = 0

        def flush_blocks():
            nonlocal block_buffer, file_counter
            if not block_buffer:
                return
            out_path = os.path.join(output_dir, f"{chunk_id}_sft_{file_counter}.parquet")
            _write_sft_blocks(block_buffer, out_path)
            block_buffer = []
            file_counter += 1

        def pack_buffer():
            nonlocal item_buffer, block_buffer, n_blocks
            if not item_buffer:
                return
            blocks = bfd_pack(item_buffer, seq_len, pad_id)
            block_buffer.extend(blocks)
            n_blocks += len(blocks)
            item_buffer = []
            if len(block_buffer) >= write_threshold:
                flush_blocks()

        with open(filename, "rb") as f:
            f.seek(start_byte)
            if start_byte != 0:
                f.readline()

            while f.tell() < end_byte:
                line = f.readline()
                if not line:
                    break
                try:
                    doc = orjson.loads(line)
                    messages = sharegpt_to_messages(doc)
                    if not messages:
                        continue
                    n_docs += 1

                    ids, mask, boundaries = tokenizer.render_conversation(
                        {"messages": messages}, max_tokens=render_max, return_boundaries=True,
                    )
                    if not ids or sum(mask) == 0:
                        n_empty += 1  # nothing to supervise
                        continue

                    items, dropped = smart_chunk_conversation(ids, mask, boundaries, seq_len, long_doc)
                    n_dropped += dropped
                    item_buffer.extend(items)

                    if len(item_buffer) >= pack_chunk_size:
                        pack_buffer()

                except Exception:
                    continue

        pack_buffer()   # pack the trailing partial buffer
        flush_blocks()  # flush any remaining blocks

        stats = {"docs": n_docs, "blocks": n_blocks, "dropped": n_dropped, "empty": n_empty}
        return chunk_id, "DONE", stats

    except Exception as e:
        return chunk_id, "ERROR", str(e)


def run_sft(args, db_path, all_filename):
    """SFT pretokenization: render ShareGPT, neat-pack, write 3-column parquet shards."""
    logger.info(
        f"SFT mode | seq_len={args.seq_len} (train max-seq-len must be {args.seq_len - 1}) | "
        f"pack_chunk_size={args.pack_chunk_size} | long_doc={args.long_doc}"
    )

    totals = {"docs": 0, "blocks": 0, "dropped": 0, "empty": 0}
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
                    process_chunk_sft, c_id, filename, start, end,
                    args.tokenizer, args.output_dir, args.seq_len, args.pack_chunk_size,
                    args.long_doc, args.write_threshold,
                ): c_id
                for c_id, start, end in pending_chunks
            }

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                               desc=f"Chunks ({os.path.basename(filename)})"):
                chunk_id, status, result = future.result()
                if status == "DONE":
                    update_chunk_status(db_path, chunk_id, "DONE")
                    for k in totals:
                        totals[k] += result.get(k, 0)
                else:
                    logger.error(f"Error in {chunk_id}: {result}")
                    update_chunk_status(db_path, chunk_id, "ERROR")

    # Dataset metadata so the loader/trainer can validate seq_len and find the pad id.
    import orjson
    pad_id = get_tokenizer_sft(args).get_bos_token_id()
    meta = {"mode": "sft", "seq_len": args.seq_len, "pad_id": pad_id}
    meta_path = os.path.join(args.output_dir, "pretokenize_metadata.json")
    with open(meta_path, "wb") as f:
        f.write(orjson.dumps(meta, option=orjson.OPT_INDENT_2))

    logger.info(
        f"SFT pretokenization complete. docs={totals['docs']:,} blocks={totals['blocks']:,} "
        f"dropped={totals['dropped']:,} empty(no-supervision)={totals['empty']:,}"
    )


def get_tokenizer_sft(args):
    """Load the project tokenizer wrapper used for SFT rendering."""
    from nanoqwen35.tokenizer import get_tokenizer
    return get_tokenizer(args.tokenizer)


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

    if args.mode == "sft":
        run_sft(args, db_path, all_filename)
        return

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
