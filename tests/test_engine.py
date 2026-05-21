"""
Tests for the Engine class.

Run with:
    python -m pytest tests/test_engine.py -v
"""

import torch
from collections import deque
from dataclasses import dataclass
from nanoqwen35.engine import KVCache, Engine, parse_tool_call, dispatch_tool


# -----------------------------------------------------------------------------
# Shared mock infrastructure

IM_END_ID  = 151645   # <|im_end|> in Qwen3.5 vocab (used as sentinel in mocks)
EOS_ID     = 151643   # <|endoftext|>
VOCAB_SIZE = 152064   # Qwen3.5 vocab size (doesn't matter for logic, just needs to be large enough)


@dataclass
class MockConfig:
    """Minimal config that Engine and KVCache need."""
    n_kv_groups: int = 4
    head_dim: int = 8
    n_layers: int = 2
    context_length: int = 512


class MockTokenizer:
    """
    UTF-8 byte-level tokenizer with Qwen3.5 special tokens.

    Token IDs:
      0-255  : raw UTF-8 bytes
      151643 : <|endoftext|>   (EOS / pad)
      151645 : <|im_end|>      (end of assistant turn)
    """
    _SPECIAL = {
        "<|endoftext|>": EOS_ID,
        "<|im_end|>":    IM_END_ID,
        "<|im_start|>":  151644,
    }
    _ID_TO_SPECIAL = {v: k for k, v in _SPECIAL.items()}

    def encode_special(self, s):
        return self._SPECIAL.get(s)

    def token_to_id(self, s):
        return self._SPECIAL.get(s)

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, token_ids):
        chunks = []
        for t in token_ids:
            if t in self._ID_TO_SPECIAL:
                chunks.append(self._ID_TO_SPECIAL[t])
            elif 0 <= t <= 255:
                try:
                    chunks.append(bytes([t]).decode("utf-8"))
                except UnicodeDecodeError:
                    chunks.append("?")
        return "".join(chunks)


class UniformModel:
    """Returns uniform logits — every token equally likely."""
    def __init__(self):
        self.config = MockConfig()
        self._device = torch.device("cpu")

    def get_device(self):
        return self._device

    def forward(self, ids, kv_cache=None):
        B, T = ids.shape
        if kv_cache is not None:
            kv_cache.advance(T)
        return torch.zeros(B, T, VOCAB_SIZE)


class ScriptedModel:
    """
    Returns logits that force generation of a pre-scripted token sequence.
    After the script is exhausted, forces <|im_end|>.
    """
    def __init__(self, script_tokens):
        self.script = deque(script_tokens)
        self.config = MockConfig()
        self._device = torch.device("cpu")

    def get_device(self):
        return self._device

    def forward(self, ids, kv_cache=None):
        B, T = ids.shape
        if kv_cache is not None:
            kv_cache.advance(T)
        logits = torch.full((B, T, VOCAB_SIZE), -1e9)
        next_tok = self.script.popleft() if self.script else IM_END_ID
        logits[:, -1, next_tok] = 1e9
        return logits


def _token_seq(text, eos=True):
    """Encode text to byte tokens, optionally append <|im_end|>."""
    toks = list(text.encode("utf-8"))
    if eos:
        toks.append(IM_END_ID)
    return toks


# -----------------------------------------------------------------------------
# KVCache tests

def test_kv_cache_basic():
    kv = KVCache(batch_size=2, num_heads=3, seq_len=64, head_dim=5,
                 num_layers=6, device="cpu", dtype=torch.float32)
    assert kv.get_pos() == 0
    assert kv.k_cache.shape == (6, 2, 64, 3, 5)
    assert kv.v_cache.shape == (6, 2, 64, 3, 5)

    kv.advance(10)
    assert kv.get_pos() == 10
    kv.advance(5)
    assert kv.get_pos() == 15

    kv.reset()
    assert kv.get_pos() == 0

    k0, v0 = kv.get_layer_cache(0)
    assert k0.shape == (2, 64, 3, 5)
    assert v0.shape == (2, 64, 3, 5)


def test_kv_cache_prefill():
    src = KVCache(batch_size=1, num_heads=4, seq_len=32, head_dim=8,
                  num_layers=2, device="cpu", dtype=torch.float32)
    src.k_cache[0, 0, :16] = 1.0
    src.v_cache[0, 0, :16] = 2.0
    src.advance(16)

    dst = KVCache(batch_size=1, num_heads=4, seq_len=64, head_dim=8,
                  num_layers=2, device="cpu", dtype=torch.float32)
    dst.prefill(src)

    assert dst.get_pos() == 16
    assert (dst.k_cache[0, 0, :16] == 1.0).all()
    assert (dst.v_cache[0, 0, :16] == 2.0).all()


# -----------------------------------------------------------------------------
# parse_tool_call and dispatch_tool unit tests

def test_parse_tool_call_basic():
    text = (
        "<tool_call>\n"
        "<function=add>\n"
        "<parameter=a>3</parameter>\n"
        "<parameter=b>4</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    result = parse_tool_call(text)
    assert result is not None
    func_name, kwargs = result
    assert func_name == "add"
    assert kwargs == {"a": "3", "b": "4"}


def test_parse_tool_call_multiline_param():
    text = (
        "<tool_call>\n"
        "<function=greet>\n"
        "<parameter=message>\n"
        "Hello,\n"
        "world!\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    result = parse_tool_call(text)
    assert result is not None
    func_name, kwargs = result
    assert func_name == "greet"
    assert "Hello," in kwargs["message"]
    assert "world!" in kwargs["message"]


def test_parse_tool_call_missing_function():
    text = "<tool_call>\n<parameter=x>1</parameter>\n</tool_call>"
    assert parse_tool_call(text) is None


def test_parse_tool_call_no_block():
    assert parse_tool_call("just some plain text") is None


def test_dispatch_tool_basic():
    def add(a: int, b: int) -> int:
        return a + b

    result = dispatch_tool("add", {"a": "3", "b": "4"}, [add])
    assert result == "7"


def test_dispatch_tool_unknown():
    result = dispatch_tool("nonexistent", {}, [])
    assert "not found" in result


def test_dispatch_tool_error():
    def boom(x: int):
        raise ValueError("intentional error")

    result = dispatch_tool("boom", {"x": "1"}, [boom])
    assert "Error" in result


def test_dispatch_tool_type_coercion():
    def mul(x: float, y: float) -> float:
        return x * y

    result = dispatch_tool("mul", {"x": "2.5", "y": "4.0"}, [mul])
    assert result == "10.0"


# -----------------------------------------------------------------------------
# Engine generation tests

def test_seed_reproducibility():
    engine = Engine(UniformModel(), MockTokenizer())
    prompt = list("Hello".encode())

    for seed in [1, 42, 123]:
        r1, _ = engine.generate_batch(prompt, max_tokens=5, seed=seed)
        r2, _ = engine.generate_batch(prompt, max_tokens=5, seed=seed)
        assert r1 == r2, f"seed={seed}: same seed must produce identical output"


def test_temperature_zero_determinism():
    engine = Engine(UniformModel(), MockTokenizer())
    prompt = list("Hi".encode())

    r1, _ = engine.generate_batch(prompt, temperature=0.0, max_tokens=5, seed=1)
    r2, _ = engine.generate_batch(prompt, temperature=0.0, max_tokens=5, seed=99)
    assert r1 == r2, "temperature=0 must give identical output regardless of seed"


def test_max_tokens_respected():
    engine = Engine(UniformModel(), MockTokenizer())
    prompt = list("Hi".encode())

    for max_tokens in [1, 4, 16]:
        results, _ = engine.generate_batch(prompt, max_tokens=max_tokens)
        n_generated = len(results[0]) - len(prompt)
        assert n_generated <= max_tokens


def test_num_samples_count():
    engine = Engine(UniformModel(), MockTokenizer())
    prompt = list("Hi".encode())

    for n in [1, 4, 8]:
        results, _ = engine.generate_batch(prompt, num_samples=n, max_tokens=3)
        assert len(results) == n


def test_multi_sample_diversity():
    """With uniform logits + temperature=1, 16 samples should not all be identical."""
    engine = Engine(UniformModel(), MockTokenizer())
    prompt = list("Hello".encode())

    first_toks = []
    for col, _ in engine.generate(prompt, num_samples=16, max_tokens=1, temperature=1.0, seed=42):
        first_toks = col

    assert len(set(first_toks)) > 1, (
        "All 16 samples produced the same first token — looks like broadcasting bug."
    )


def test_eos_stops_generation():
    """<|im_end|> token must end generation for that row."""
    # Script: generate 3 real tokens then im_end
    tok = MockTokenizer()
    script = list("abc".encode()) + [IM_END_ID]
    engine = Engine(ScriptedModel(script), tok)

    results, _ = engine.generate_batch(list("X".encode()), max_tokens=20)
    generated = results[0][1:]  # strip prompt
    assert IM_END_ID not in generated, "<|im_end|> should be stripped from results"
    assert tok.decode(generated) == "abc"


def test_eos_id_stops_generation():
    """<|endoftext|> token must also end generation."""
    tok = MockTokenizer()
    script = list("hi".encode()) + [EOS_ID]
    engine = Engine(ScriptedModel(script), tok)

    results, _ = engine.generate_batch(list("X".encode()), max_tokens=20)
    generated = results[0][1:]
    assert EOS_ID not in generated
    assert tok.decode(generated) == "hi"


# -----------------------------------------------------------------------------
# Tool calling integration tests

def _make_tool_call_script(func_name, params: dict):
    """Build the byte-token script that spells out a tool_call block + <|im_end|>."""
    lines = ["<tool_call>\n", f"<function={func_name}>\n"]
    for k, v in params.items():
        lines.append(f"<parameter={k}>{v}</parameter>\n")
    lines.append("</function>\n</tool_call>")
    text = "".join(lines)
    return list(text.encode("utf-8")) + [IM_END_ID]


def test_tool_call_add():
    """Engine should detect <tool_call>, call add(a,b), inject <tool_response>."""
    def add(a: int, b: int) -> int:
        return a + b

    tok = MockTokenizer()
    script = _make_tool_call_script("add", {"a": "3", "b": "4"})
    engine = Engine(ScriptedModel(script), tok)

    results, masks = engine.generate_batch(
        list("Q".encode()), tools=[add], max_tokens=512
    )
    full_text = tok.decode(results[0])
    assert "<tool_response>" in full_text
    assert "7" in full_text
    assert "</tool_response>" in full_text


def test_tool_call_unknown_function():
    """Engine should inject an error message when the called function is not in the tools list."""
    tok = MockTokenizer()
    script = _make_tool_call_script("unknown_fn", {"x": "1"})
    engine = Engine(ScriptedModel(script), tok)

    # tools=[] means tool-calling is enabled but the list is empty — dispatch returns "not found"
    results, _ = engine.generate_batch(
        list("Q".encode()), tools=[], max_tokens=512
    )
    full_text = tok.decode(results[0])
    assert "<tool_response>" in full_text
    assert "not found" in full_text


def test_tool_call_no_tools_registered():
    """When tools=None, engine should NOT inject any tool response."""
    tok = MockTokenizer()
    script = _make_tool_call_script("add", {"a": "1", "b": "2"})
    engine = Engine(ScriptedModel(script), tok)

    results, _ = engine.generate_batch(
        list("Q".encode()), tools=None, max_tokens=512
    )
    full_text = tok.decode(results[0])
    assert "<tool_response>" not in full_text


def test_tool_call_mask_is_zero_for_forced_tokens():
    """Tokens injected as tool responses must have mask=0."""
    def add(a: int, b: int) -> int:
        return a + b

    tok = MockTokenizer()
    script = _make_tool_call_script("add", {"a": "10", "b": "20"})
    engine = Engine(ScriptedModel(script), tok)

    _, masks = engine.generate_batch(
        list("Q".encode()), tools=[add], max_tokens=512
    )
    combined_masks = masks[0]
    # prompt tokens are 0, sampled tokens are 1, forced (tool response) tokens are 0
    # The tool response should introduce some 0-mask tokens after 1-mask tokens
    # (prompt is all 0s, then model generates 1s, then forced response is 0s again)
    assert 0 in combined_masks[len(list("Q".encode())):], (
        "Tool response tokens should have mask=0"
    )


def test_tool_call_string_return():
    """Tool that returns a string should be injected correctly."""
    def get_weather(city: str) -> str:
        return f"Sunny in {city}, 25°C"

    tok = MockTokenizer()
    script = _make_tool_call_script("get_weather", {"city": "Paris"})
    engine = Engine(ScriptedModel(script), tok)

    results, _ = engine.generate_batch(
        list("Q".encode()), tools=[get_weather], max_tokens=512
    )
    full_text = tok.decode(results[0])
    assert "Sunny in Paris" in full_text


def test_tool_call_exception_is_caught():
    """A tool that raises should inject an error message, not crash the engine."""
    def broken_tool(x: int) -> int:
        raise RuntimeError("disk on fire")

    tok = MockTokenizer()
    script = _make_tool_call_script("broken_tool", {"x": "1"})
    engine = Engine(ScriptedModel(script), tok)

    results, _ = engine.generate_batch(
        list("Q".encode()), tools=[broken_tool], max_tokens=512
    )
    full_text = tok.decode(results[0])
    assert "<tool_response>" in full_text
    assert "Error" in full_text
