"""
Tests for SFT pretokenization helpers (scripts/pretokenize.py --mode sft).

Run: python -m pytest tests/test_sft_pretokenize.py -v

The pure-logic tests (ShareGPT conversion, best-fit packing, smart chunking) run
anywhere. The render-based tests require the Qwen3.5 tokenizer; they are skipped if
it cannot be loaded (set QWEN_TOKENIZER to a local tokenizer dir to enable them).
"""
import os
import pytest

import scripts.pretokenize as pt


# -----------------------------------------------------------------------------
# Tokenizer fixture (skips cleanly if unavailable)

_TOKENIZER_CANDIDATES = [
    os.environ.get("QWEN_TOKENIZER"),
    "/home/truongnp5/Desktop/qwen35/Qwen3.5-0.8B",
]


def _load_tokenizer():
    from nanoqwen35.tokenizer import HuggingFaceTokenizer, get_tokenizer
    for cand in _TOKENIZER_CANDIDATES:
        if cand and os.path.isfile(os.path.join(cand, "tokenizer.json")):
            return HuggingFaceTokenizer.from_directory(cand)
    try:
        return get_tokenizer("Qwen/Qwen3.5-0.8B-Base")
    except Exception:
        return None


@pytest.fixture(scope="module")
def tokenizer():
    tok = _load_tokenizer()
    if tok is None:
        pytest.skip("Qwen3.5 tokenizer not available (set QWEN_TOKENIZER)")
    return tok


# -----------------------------------------------------------------------------
# Pure-logic tests (no tokenizer / GPU)

class TestShareGPTConversion:
    def test_basic_role_mapping(self):
        doc = {"conversations": [
            {"from": "human", "value": "hi"},
            {"from": "gpt", "value": "hello"},
        ]}
        msgs = pt.sharegpt_to_messages(doc)
        assert msgs == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_system_and_tool(self):
        doc = {"system": "be nice", "conversations": [
            {"from": "human", "value": "q"},
            {"from": "gpt", "value": "a"},
            {"from": "observation", "value": "tool out"},
            {"from": "gpt", "value": "final"},
        ]}
        msgs = pt.sharegpt_to_messages(doc)
        assert msgs[0] == {"role": "system", "content": "be nice"}
        assert msgs[3] == {"role": "tool", "content": "tool out"}

    def test_empty_returns_none(self):
        assert pt.sharegpt_to_messages({}) is None
        assert pt.sharegpt_to_messages({"conversations": []}) is None


class TestBFDPack:
    def test_bins_full_and_conserve_tokens(self):
        items = [(list(range(5)), [1] * 5), (list(range(3)), [1] * 3),
                 (list(range(4)), [0] * 4), (list(range(2)), [1] * 2)]
        L, pad = 8, 99
        bins = pt.bfd_pack(items, L, pad_id=pad)
        for ids, mask, segs in bins:
            assert len(ids) == L and len(mask) == L
            assert sum(segs) == L
        real = sum(len(i[0]) for i in items)
        padded = sum(ids.count(pad) for ids, _, _ in bins)
        assert real + padded == len(bins) * L

    def test_no_bin_exceeds_capacity(self):
        items = [(list(range(L)), [1] * L) for L in (7, 7, 7, 1)]
        bins = pt.bfd_pack(items, 8, pad_id=0)
        for ids, _, _ in bins:
            assert len(ids) == 8

    def test_padding_is_masked(self):
        items = [(list(range(3)), [1] * 3)]
        bins = pt.bfd_pack(items, 8, pad_id=0)
        ids, mask, segs = bins[0]
        # the 5 padding positions must have mask 0
        assert mask[3:] == [0] * 5
        assert segs == [3, 5]


class TestSmartChunk:
    def test_short_passthrough(self):
        out, dropped = pt.smart_chunk_conversation(list(range(10)), [1] * 10, [0, 5], L=20)
        assert dropped == 0 and len(out) == 1

    def test_splits_on_user_boundary(self):
        ids, mask, boundaries = list(range(20)), [1] * 20, [0, 10]
        out, dropped = pt.smart_chunk_conversation(ids, mask, boundaries, L=12)
        assert dropped == 0
        assert all(len(o[0]) <= 12 for o in out)
        # reconstruct: concatenation of pieces equals original
        assert sum((o[0] for o in out), []) == ids

    def test_drops_oversized_single_turn(self):
        # a single user turn (boundaries=[0]) longer than L cannot be split -> drop
        out, dropped = pt.smart_chunk_conversation(list(range(20)), [1] * 20, [0], L=12)
        assert dropped == 1 and out == []

    def test_policy_drop(self):
        out, dropped = pt.smart_chunk_conversation(list(range(20)), [1] * 20, [0, 10], L=12, policy="drop")
        assert dropped == 1 and out == []


# -----------------------------------------------------------------------------
# Render-based tests (need the tokenizer)

class TestRenderBoundariesAndMask:
    def _conv(self):
        return {"messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "FIRSTUSERQUERY"},
            {"role": "assistant", "content": "FIRSTREPLY"},
            {"role": "user", "content": "SECONDUSERQUERY"},
            {"role": "assistant", "content": "SECONDREPLY"},
        ]}

    def test_boundaries_point_to_user_turns(self, tokenizer):
        ids, mask, boundaries = tokenizer.render_conversation(
            self._conv(), max_tokens=4096, return_boundaries=True)
        im_start = tokenizer.encode_special("<|im_start|>")
        assert len(boundaries) == 2  # two user turns
        for b in boundaries:
            assert ids[b] == im_start

    def test_mask_invariant_only_assistant(self, tokenizer):
        conv = self._conv()
        ids, mask, _ = tokenizer.render_conversation(conv, max_tokens=4096, return_boundaries=True)
        # user/system content tokens must never be supervised
        for secret in ("FIRSTUSERQUERY", "SECONDUSERQUERY", "You are helpful."):
            sec_ids = tokenizer.encode(secret)
            for i in range(len(ids) - len(sec_ids) + 1):
                if ids[i:i + len(sec_ids)] == sec_ids:
                    assert all(m == 0 for m in mask[i:i + len(sec_ids)]), f"{secret} supervised!"
        # at least some assistant tokens are supervised
        assert sum(mask) > 0
