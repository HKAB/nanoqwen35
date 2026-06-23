"""
Distributed dataloaders.

Two loaders, each for a different training stage:

  pretrain_loader / pretrain_loader_with_state   — pretrain
      Reads pre-tokenized parquet files produced by scripts/pretokenize.py.
      Each parquet stores an "input_ids" column of packed token blocks; the
      loader streams those tokens and re-chunks them into T+1 sequences, so no
      runtime tokenization is needed and the block size is decoupled from T.

      All files under dataset_root are deterministically shuffled (fixed seed)
      and partitioned once: each rank gets `files_per_rank` disjoint training
      files, and the leftover files form a shared validation set (see
      _partition_files). The partition is identical on every rank.

  sft_loader                                     — SFT / RL
      Reads raw parquet files with a 'messages' field (chat format).
      Tokenizes online via tokenizer.render_conversation(), then packs with
      best-fit knapsack: fills each batch row with as many complete conversations
      as possible, padding the remainder and masking non-assistant targets with -1.
"""

import os
import json as _json
import random

import numpy as np
import torch
import pyarrow.parquet as pq

from nanoqwen35.common import get_dist_info, print0
from nanoqwen35.dataset import list_parquet_files_by_domain, list_all_parquet_files


# ---------------------------------------------------------------------------
# Pre-tokenized dataloader (pretrain)
# ---------------------------------------------------------------------------

def _partition_files(dataset_root, world_size, seed):
    """
    Deterministically shuffle every parquet file under dataset_root and split
    into per-rank training files plus a shared validation set.

    `files_per_rank = total // world_size`, decremented until the leftover
    (validation) count is at least `world_size` so val can shard across ranks.
    Example: 925 files, 8 ranks → 114 train/rank (912 total) + 13 val.

    Returns (train_files, val_files, files_per_rank). The result is identical on
    every rank since both the file listing and the shuffle are deterministic.
    """
    files = list_all_parquet_files(dataset_root)
    assert files, f"No parquet files found under {dataset_root}"

    random.Random(seed).shuffle(files)

    files_per_rank = len(files) // world_size
    while files_per_rank > 0 and len(files) - files_per_rank * world_size < world_size:
        files_per_rank -= 1
    assert files_per_rank > 0, (
        f"Only {len(files)} files for {world_size} ranks — too few to partition."
    )

    train_files = files[:files_per_rank * world_size]
    val_files   = files[files_per_rank * world_size:]
    return train_files, val_files, files_per_rank


def _row_iter(paths, T1, resume_state_dict=None):
    """
    Infinite iterator over T1-length token sequences from the given parquet files.

    The "input_ids" column of each row group is read as one flat token stream and
    re-chunked into fixed T1 (= T+1) windows, carrying the remainder across row
    groups. Each yield is (row_np, (pq_idx, rg_idx, epoch)) with row_np a
    np.int32 array of shape (T1,).

    On resume, iteration restarts after (pq_idx, rg_idx); the in-flight carry of
    < T1 tokens is dropped (negligible for pretraining).
    """
    assert paths, "No parquet files to iterate"

    resume_pq  = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg  = resume_state_dict.get("rg_idx") if resume_state_dict is not None else None
    epoch      = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1

    first_pass = True
    carry = np.empty(0, dtype=np.int32)

    while True:
        start_pq = resume_pq if first_pass else 0
        for pq_idx in range(start_pq, len(paths)):
            pf = pq.ParquetFile(paths[pq_idx])

            start_rg = 0
            if first_pass and resume_rg is not None and pq_idx == start_pq:
                start_rg = resume_rg + 1
                resume_rg = None

            for rg_idx in range(start_rg, pf.num_row_groups):
                col = pf.read_row_group(rg_idx, columns=["input_ids"]).column("input_ids")
                c = col.combine_chunks() if col.num_chunks > 1 else col.chunks[0]
                flat_np = c.values.to_numpy(zero_copy_only=False)

                stream = np.concatenate([carry, flat_np]) if carry.size else flat_np
                n_full = (len(stream) // T1) * T1
                for off in range(0, n_full, T1):
                    yield stream[off:off + T1], (pq_idx, rg_idx, epoch)
                carry = stream[n_full:].copy()

        first_pass = False
        epoch += 1


def pretrain_loader_with_state(
    B, T, split,
    dataset_root,
    device="cuda",
    resume_state_dict=None,
    seed=42,
):
    """
    Pretrain dataloader over pre-tokenized parquet files in dataset_root.

    Files are partitioned by _partition_files: each rank reads its disjoint
    training files, while validation reads the leftover files strided across
    ranks. Rows are streamed and re-chunked into T+1 sequences.

    State dict format: {"pq_idx": int, "rg_idx": int, "epoch": int}
    """
    assert split in ("train", "val"), "split must be 'train' or 'val'"

    _, ddp_rank, _, ddp_world_size = get_dist_info()
    train_files, val_files, files_per_rank = _partition_files(dataset_root, ddp_world_size, seed)

    if split == "train":
        my_files = train_files[ddp_rank * files_per_rank:(ddp_rank + 1) * files_per_rank]
        resume = resume_state_dict
    else:
        my_files = val_files[ddp_rank::ddp_world_size]
        resume = None  # val is rebuilt fresh on every eval
    assert my_files, f"Rank {ddp_rank} has no {split} files"

    print0(
        f"  [pretrain] {split}: {len(train_files)} train files "
        f"({files_per_rank}/rank × {ddp_world_size}) + {len(val_files)} val files"
    )

    T1 = T + 1
    row_iter = _row_iter(my_files, T1, resume)

    row_buffer = np.empty((B, T1), dtype=np.int32)
    use_cuda   = device == "cuda"
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs  = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs  = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    state = {"pq_idx": 0, "rg_idx": 0, "epoch": 1}

    while True:
        for row_idx in range(B):
            row_np, (pq_idx, rg_idx, epoch) = next(row_iter)
            state = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
            row_buffer[row_idx] = row_np

        cpu_inputs.copy_(torch.from_numpy(row_buffer[:, :-1]).long())
        cpu_targets.copy_(torch.from_numpy(row_buffer[:, 1:]).long())
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, dict(state)


def pretrain_loader(*args, **kwargs):
    """pretrain_loader_with_state without the state dict yield."""
    for inputs, targets, _ in pretrain_loader_with_state(*args, **kwargs):
        yield inputs, targets


# ---------------------------------------------------------------------------
# SFT / RL dataloader (best-fit knapsack, online tokenization)
# ---------------------------------------------------------------------------

def _sft_conv_iter(split, parquet_files, tokenizer, max_tokens):
    """
    Infinite iterator over (token_ids, loss_mask) pairs from a parquet domain.

    Reads the 'messages' column (JSON string or pyarrow list-of-structs) and
    tokenizes with tokenizer.render_conversation().  Empty conversations are
    skipped.  Conversations longer than max_tokens are truncated by
    render_conversation's own max_tokens parameter.

    Uses the same file-level / row-group-fallback sharding as _row_iter.
    """
    _, ddp_rank, _, ddp_world_size = get_dist_info()

    paths = sorted(p for p in parquet_files if split in os.path.basename(p))
    assert paths, f"No {split} parquet files found for SFT loader"

    file_sharding = len(paths) >= ddp_world_size
    if file_sharding:
        paths = paths[ddp_rank::ddp_world_size]

    while True:
        for path in paths:
            pf = pq.ParquetFile(path)
            for rg_idx in range(pf.num_row_groups):
                if not file_sharding and rg_idx % ddp_world_size != ddp_rank:
                    continue
                msgs_col = pf.read_row_group(rg_idx, columns=["messages"]).column("messages").to_pylist()
                for msgs in msgs_col:
                    # Handle JSON string storage or already-parsed list-of-dicts
                    if isinstance(msgs, (str, bytes)):
                        msgs = _json.loads(msgs)
                    conversation = {"messages": msgs}
                    ids, mask = tokenizer.render_conversation(conversation, max_tokens=max_tokens)
                    if ids:
                        yield list(ids), list(mask)


def sft_loader(
    B, T, split,
    dataset_root,
    tokenizer,
    device="cuda",
    buffer_size=128,
    mask_history=False,
):
    """
    Multi-domain SFT / RL dataloader with best-fit knapsack packing.

    Reads raw parquet files whose 'messages' column contains chat-format conversations
    ([{"role": "user"/"assistant"/"system", "content": "..."}]).
    Online tokenization via tokenizer.render_conversation() — no pre-tokenization needed.

    Packing strategy (best-fit, no cropping):
      - Each batch row has T+1 token slots.
      - Greedily pack complete conversations using best-fit decreasing.
      - Remaining space is padded; padded and non-assistant positions have target = -1.

    Args:
        mask_history: if True, only the last assistant turn is supervised (RL-style).
                      if False (default), all assistant turns are supervised (SFT-style).
    """
    assert split in ("train", "val"), "split must be 'train' or 'val'"

    domain_file_map = list_parquet_files_by_domain(dataset_root)
    assert domain_file_map, f"No domain subdirectories with parquet files found in {dataset_root}"

    domains = sorted(domain_file_map.keys())

    _, _, _, ddp_world_size = get_dist_info()
    for d in domains:
        files = [f for f in domain_file_map[d] if split in os.path.basename(f)]
        mode = ("file-sharding" if len(files) >= ddp_world_size
                else f"row-group-fallback ({len(files)} files < {ddp_world_size} ranks)")
        print0(f"  [sft] domain '{d}': {len(files)} {split} files → {mode}")

    T1     = T + 1
    pad_id = tokenizer.get_bos_token_id()  # EOS used as pad token

    # render_conversation handles truncation via max_tokens; we pass T so every
    # conversation fits in T slots, guaranteeing at least one conv per row.
    _max_tok = T if not mask_history else T
    domain_iters   = {
        d: _sft_conv_iter(split, domain_file_map[d], tokenizer, _max_tok)
        for d in domains
    }
    domain_buffers = {d: [] for d in domains}

    # Pre-allocated numpy row scratch (avoids per-step allocation)
    row_ids_scratch  = np.empty(T1, dtype=np.int32)
    row_mask_scratch = np.zeros(T1, dtype=np.int8)

    # Pre-allocated pinned CPU ↔ GPU tensor pair
    use_cuda   = device == "cuda"
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs  = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs  = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            d   = random.choice(domains)
            buf = domain_buffers[d]
            it  = domain_iters[d]

            # Top up buffer
            while len(buf) < buffer_size:
                buf.append(next(it))

            # Initialise row with padding
            row_ids_scratch[:]  = pad_id
            row_mask_scratch[:] = 0
            pos       = 0
            remaining = T1

            # Best-fit knapsack: pack complete conversations, no cropping
            while remaining > 0:
                best_i, best_len = -1, 0
                for i, (cids, _) in enumerate(buf):
                    clen = len(cids)
                    if clen <= remaining and clen > best_len:
                        best_i, best_len = i, clen

                if best_i < 0:
                    break  # nothing fits — remaining slots stay as padding

                cids, cmask = buf.pop(best_i)
                row_ids_scratch[pos:pos + best_len]  = cids
                row_mask_scratch[pos:pos + best_len] = cmask
                pos       += best_len
                remaining -= best_len

            # inp = row[:-1], tgt = row[1:] with mask applied
            inp_t = torch.from_numpy(row_ids_scratch[:-1].astype(np.int64))
            tgt_t = torch.from_numpy(row_ids_scratch[1:].astype(np.int64))
            msk_t = torch.from_numpy(row_mask_scratch[1:])  # mask aligned to targets
            tgt_t[msk_t == 0] = -1

            cpu_inputs[row_idx].copy_(inp_t)
            cpu_targets[row_idx].copy_(tgt_t)

        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets
