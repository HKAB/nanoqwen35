"""
Tests for the pretokenized-SFT dataloader (nanoqwen35.dataloader.sft_pretokenized_loader).

Run: python -m pytest tests/test_sft_pretokenized_loader.py -v

These build a tiny synthetic packed dataset on disk (no GPU, no tokenizer) and verify
that the loader reconstructs inputs/targets, applies the loss mask as ignore_index=-1,
and produces correct block-diagonal cu_seqlens / position_ids.
"""
import json
import pytest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from nanoqwen35.dataloader import sft_pretokenized_loader, _input_space_segs


PAD = 99


def _write_dataset(tmp_path, L, blocks_per_file=2):
    """Write two shards of identical packed blocks + metadata. Returns the root dir."""
    # block 0: two convs of len 5 and 3 (mask: assistant tokens = 1)
    b0 = ([10, 11, 12, 13, 14, 20, 21, 22], [0, 0, 1, 1, 1, 0, 1, 1], [5, 3])
    # block 1: convs of len 3 and 4 + 1 pad
    b1 = ([30, 31, 32, 40, 41, 42, 43, PAD], [0, 1, 1, 0, 0, 1, 1, 0], [3, 4, 1])
    assert all(len(b[0]) == L for b in (b0, b1))
    table = pa.table({
        "input_ids": pa.array([b[0] for b in (b0, b1)], type=pa.list_(pa.int32())),
        "loss_mask": pa.array([b[1] for b in (b0, b1)], type=pa.list_(pa.uint8())),
        "seq_lens":  pa.array([b[2] for b in (b0, b1)], type=pa.list_(pa.int32())),
    })
    for name in ("shard_a_0.parquet", "shard_b_0.parquet"):
        pq.write_table(table, str(tmp_path / name), compression="zstd")
    (tmp_path / "pretokenize_metadata.json").write_text(
        json.dumps({"mode": "sft", "seq_len": L, "pad_id": PAD}))
    return str(tmp_path)


class TestInputSpaceSegs:
    def test_caps_at_T(self):
        # block segments sum to L = T+1; input space must sum to T (drops last token)
        assert _input_space_segs([5, 3], 7) == [5, 2]
        assert _input_space_segs([3, 4, 1], 7) == [3, 4]   # trailing len-1 pad dropped
        assert _input_space_segs([8], 7) == [7]


class TestLoader:
    def test_masking_and_packing(self, tmp_path):
        L = 8
        root = _write_dataset(tmp_path, L)
        loader = sft_pretokenized_loader(B=2, T=L - 1, split="train", dataset_root=root, device="cpu")
        x, y, cu, pos = next(loader)

        assert x.shape == (2, L - 1) and y.shape == (2, L - 1)
        # row 0: ids[:-1] inputs, targets = ids[1:] with mask[1:]==0 -> -1
        assert x[0].tolist() == [10, 11, 12, 13, 14, 20, 21]
        assert y[0].tolist() == [-1, 12, 13, 14, -1, 21, 22]
        # position ids reset per input-space segment: segs [5,3] -> [5,2]
        assert pos[0].tolist() == [0, 1, 2, 3, 4, 0, 1]
        # cu_seqlens over flattened batch: row0 [5,2], row1 [3,4] -> [0,5,7,10,14]
        assert cu.dtype == torch.int32
        assert cu.tolist() == [0, 5, 7, 10, 14]
        assert cu[-1].item() == 2 * (L - 1)

    def test_seq_len_mismatch_raises(self, tmp_path):
        root = _write_dataset(tmp_path, L=8)
        with pytest.raises(AssertionError):
            # T+1 (=8) must equal stored seq_len (=8); here T=10 -> mismatch
            next(sft_pretokenized_loader(B=1, T=10, split="train", dataset_root=root, device="cpu"))

    def test_targets_are_ignore_index(self, tmp_path):
        root = _write_dataset(tmp_path, L=8)
        x, y, cu, pos = next(sft_pretokenized_loader(B=2, T=7, split="train", dataset_root=root, device="cpu"))
        # every masked target is exactly -1 (model uses ignore_index=-1)
        assert ((y == -1) | (y >= 0)).all()
        assert (y == -1).sum() > 0
