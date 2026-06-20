"""
Distributed dataloaders.

Two loaders, each for a different training stage:

  pretrain_loader / pretrain_loader_with_state   — pretrain
      Reads pre-tokenized flat shards produced by scripts/pretokenize_and_merge.py.
      Each row is already a packed T+1 token sequence; no runtime tokenization.
      File-level DDP sharding: each rank reads disjoint shards.

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
from nanoqwen35.dataset import list_parquet_files_by_domain


# ---------------------------------------------------------------------------
# Pre-tokenized dataloader (pretrain)
# ---------------------------------------------------------------------------

def _row_iter(split, resume_state_dict, parquet_files):
    """
    Infinite iterator over pre-packed rows from pretokenized parquet files.

    Each yield is (row_np, (pq_idx, rg_idx, epoch)) where row_np is a
    np.int32 array of shape (T+1,).

    Sharding strategy:
    - File-level (preferred): each rank owns disjoint files [rank, rank+W, ...].
      No two ranks open the same file — eliminates NFS metadata contention.
    - Row-group fallback: used when domain has fewer files than ranks. All ranks
      open all files but stride row-groups by world_size.
    """
    _, ddp_rank, _, ddp_world_size = get_dist_info()

    parquet_paths = list(parquet_files)
    assert parquet_paths, "No pretokenized parquet files found"

    basename = os.path.basename
    if split == "train":
        parquet_paths = [p for p in parquet_paths if "train" in basename(p)]
    else:
        parquet_paths = [p for p in parquet_paths if "val" in basename(p)]
    assert parquet_paths, f"No parquet files found for split '{split}'"

    file_sharding = len(parquet_paths) >= ddp_world_size
    if file_sharding:
        parquet_paths = parquet_paths[ddp_rank::ddp_world_size]

    raw_resume_pq  = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx  = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch   = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    resume_pq_idx  = raw_resume_pq // ddp_world_size if file_sharding else raw_resume_pq

    first_pass = True
    epoch = resume_epoch

    while True:
        start_pq = resume_pq_idx if first_pass else 0
        for local_pq_idx in range(start_pq, len(parquet_paths)):
            global_pq_idx = local_pq_idx * ddp_world_size + ddp_rank if file_sharding else local_pq_idx
            pf = pq.ParquetFile(parquet_paths[local_pq_idx])

            if file_sharding:
                start_rg = 0
                if first_pass and resume_rg_idx is not None and local_pq_idx == start_pq:
                    start_rg = resume_rg_idx + 1
                    resume_rg_idx = None
                for rg_idx in range(start_rg, pf.num_row_groups):
                    col = pf.read_row_group(rg_idx, columns=["input_ids"]).column("input_ids")
                    c = col.combine_chunks() if col.num_chunks > 1 else col.chunks[0]
                    flat_np = c.values.to_numpy(zero_copy_only=False)
                    rows_np = flat_np.reshape(len(c), -1)
                    for row_i in range(len(rows_np)):
                        yield rows_np[row_i], (global_pq_idx, rg_idx, epoch)
            else:
                if first_pass and resume_rg_idx is not None and local_pq_idx == start_pq:
                    base_idx = resume_rg_idx // ddp_world_size + 1
                    rg_idx = base_idx * ddp_world_size + ddp_rank
                    if rg_idx >= pf.num_row_groups:
                        continue
                    resume_rg_idx = None
                else:
                    rg_idx = ddp_rank
                while rg_idx < pf.num_row_groups:
                    col = pf.read_row_group(rg_idx, columns=["input_ids"]).column("input_ids")
                    c = col.combine_chunks() if col.num_chunks > 1 else col.chunks[0]
                    flat_np = c.values.to_numpy(zero_copy_only=False)
                    rows_np = flat_np.reshape(len(c), -1)
                    for row_i in range(len(rows_np)):
                        yield rows_np[row_i], (global_pq_idx, rg_idx, epoch)
                    rg_idx += ddp_world_size

        first_pass = False
        epoch += 1


# ---------------------------------------------------------------------------
# Merged flat dataloader (output of scripts/pretokenize_and_merge.py)
# ---------------------------------------------------------------------------

def pretrain_loader_with_state(
    B, T, split,
    dataset_root,
    device="cuda",
    resume_state_dict=None,
):
    """
    Pretrain dataloader for merged flat shards from scripts/pretokenize_and_merge.py.

    Reads train_XXXX.parquet / val_XXXX.parquet directly from dataset_root with
    file-level DDP sharding: each rank reads disjoint shards.

    State dict format: {"pq_idx": int, "rg_idx": int, "epoch": int}
    """
    assert split in ("train", "val"), "split must be 'train' or 'val'"

    from nanoqwen35.dataset import get_merged_metadata

    meta = get_merged_metadata(dataset_root)
    assert meta is not None, (
        f"merged_metadata.json not found in {dataset_root}. "
        "Run: python -m scripts.pretokenize_and_merge --source-root ... --output-root ..."
    )
    assert meta["T"] == T, (
        f"Sequence length mismatch: merged dataset has T={meta['T']} but training uses T={T}. "
        f"Re-run pretokenize_and_merge.py with --T {T}."
    )

    # Collect flat shard files for this split
    shard_files = sorted(
        os.path.join(dataset_root, f)
        for f in os.listdir(dataset_root)
        if f.startswith(f"{split}_") and f.endswith(".parquet") and not f.endswith(".tmp")
    )
    assert shard_files, f"No {split} shards found in {dataset_root}"

    _, _, _, ddp_world_size = get_dist_info()
    mode = (
        "file-sharding"
        if len(shard_files) >= ddp_world_size
        else f"row-group-fallback ({len(shard_files)} shards < {ddp_world_size} ranks)"
    )
    print0(f"  [merged] {split}: {len(shard_files)} shards → {mode}")

    # _row_iter filters by split name and handles file-level DDP sharding
    row_iter = _row_iter(split, resume_state_dict, shard_files)

    row_buffer = np.empty((B, T + 1), dtype=np.int32)
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
