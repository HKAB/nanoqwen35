"""
Distributed dataloaders for pretraining.

Bestfit:
   - Documents packed using best-fit algorithm to minimize cropping
   - When no document fits remaining space, crops a document to fill exactly
   - 100% utilization (no padding)
"""

import random

import torch
import pyarrow.parquet as pq

from nanoqwen35.common import get_dist_info
from nanoqwen35.dataset import list_parquet_files, list_parquet_files_by_domain

def _document_batches(split, resume_state_dict, tokenizer_batch_size, dataset_path, parquet_files=None):
    """
    Infinite iterator over document batches (list of text strings) from parquet files.

    Handles DDP sharding and approximate resume. Each yield is (text_batch, (pq_idx, rg_idx, epoch))
    where text_batch is a list of document strings, indices track position for resumption,
    and epoch counts how many times we've cycled through the dataset (starts at 1).

    Pass parquet_files to supply a pre-computed file list (bypasses list_parquet_files/dataset_path).
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    if parquet_files is not None:
        parquet_paths = parquet_files
    else:
        parquet_paths = list_parquet_files(dataset_path)
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:  # iterate infinitely (multi-epoch)
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            # Start from resume point if resuming on same file, otherwise from DDP rank
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1  # advance by 1 so we don't repeat data after resuming
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None  # only do this once
            else:
                rg_idx = ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False
        epoch += 1

def tokenizing_distributed_data_loader_with_state_weighted(
    tokenizer, B, T, split,
    dataset_root,
    domain_weights=None,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=100,
):
    """
    Multi-domain weighted dataloader with best-fit cropping.

    Loads parquet files from subdirectories of dataset_root (each subdir = one domain).
    Each row in the batch is filled entirely from one domain, chosen via weighted random
    sampling. domain_weights is a {domain_name: float} dict; omit for uniform weights.

    Resume is approximate: each domain's sequential reader resumes from its saved
    pq_idx/rg_idx position, but the random domain-sampling order is not replicated.

    State dict format: {"domain_states": {domain: {"pq_idx": int, "rg_idx": int, "epoch": int}}}
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    domain_file_map = list_parquet_files_by_domain(dataset_root)
    assert domain_file_map, f"No domain subdirectories with parquet files found in {dataset_root}"

    domains = sorted(domain_file_map.keys())
    weights = [domain_weights.get(d, 1.0) if domain_weights else 1.0 for d in domains]

    saved_domain_states = (resume_state_dict or {}).get("domain_states", {})
    domain_batches = {
        d: _document_batches(split, saved_domain_states.get(d), tokenizer_batch_size, dataset_path=None, parquet_files=domain_file_map[d])
        for d in domains
    }
    domain_doc_buffers = {d: [] for d in domains}
    domain_states = {d: {"pq_idx": 0, "rg_idx": 0, "epoch": 1} for d in domains}

    def refill_domain(d):
        doc_batch, (pq_idx, rg_idx, epoch) = next(domain_batches[d])
        domain_states[d] = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
        token_lists = tokenizer.encode(doc_batch, num_threads=tokenizer_threads)
        domain_doc_buffers[d].extend(token_lists)

    row_capacity = T + 1
    use_cuda = device == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            d = random.choices(domains, weights=weights, k=1)[0]
            buf = domain_doc_buffers[d]
            pos = 0
            while pos < row_capacity:
                while len(buf) < buffer_size:
                    refill_domain(d)

                remaining = row_capacity - pos

                best_idx = -1
                best_len = 0
                for i, doc in enumerate(buf):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = buf.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(buf)), key=lambda i: len(buf[i]))
                    doc = buf.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        state_dict = {"domain_states": {d: dict(domain_states[d]) for d in domains}}

        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict


def tokenizing_distributed_data_loader_weighted(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state_weighted(*args, **kwargs):
        yield inputs, targets
