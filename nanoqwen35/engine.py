"""
Engine for efficient inference of Qwen3.5 models.

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

Notes:
- The engine knows nothing about tokenization except for EOS/special tokens.
- Supports Qwen3.5 native tool calling format.

Tool calling format (Qwen3.5 default):
    <tool_call>
    <function=FNAME>
    <parameter=P1>value1</parameter>
    <parameter=P2>value2</parameter>
    </function>
    </tool_call>

The engine parses this, calls the matching Python function, and injects:
    <tool_response>
    result
    </tool_response>
"""

import re
import inspect
import torch
import torch.nn.functional as F
from collections import deque
from nanoqwen35.common import compute_init, autodetect_device_type
from nanoqwen35.checkpoint_manager import load_pretrained_hf

# -----------------------------------------------------------------------------
# Tool call helpers

def parse_tool_call(text):
    """
    Parse a Qwen3.5 <tool_call>...</tool_call> block.
    Returns (func_name, kwargs_dict) or None on parse failure.
    """
    m = re.search(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    if not m:
        return None
    content = m.group(1)
    fn_m = re.search(r'<function=(\w+)', content)
    if not fn_m:
        return None
    func_name = fn_m.group(1)
    kwargs = {}
    for pm in re.finditer(r'<parameter=(\w+)>(.*?)</parameter>', content, re.DOTALL):
        kwargs[pm.group(1)] = pm.group(2).strip()
    return func_name, kwargs


def dispatch_tool(func_name, kwargs, tools):
    """
    Find a tool by name in `tools` and call it with the given kwargs.
    Attempts to coerce string args to annotated types.
    Returns a string result (or an error message).
    """
    func = next((f for f in tools if f.__name__ == func_name), None)
    if func is None:
        return f"Error: function '{func_name}' not found"
    try:
        hints = {}
        try:
            hints = {k: v.annotation for k, v in inspect.signature(func).parameters.items()
                     if v.annotation is not inspect.Parameter.empty}
        except Exception:
            pass
        coerced = {}
        for k, v in kwargs.items():
            if k in hints and hints[k] is not inspect.Parameter.empty:
                try:
                    coerced[k] = hints[k](v)
                except Exception:
                    coerced[k] = v
            else:
                coerced[k] = v
        return str(func(**coerced))
    except Exception as e:
        return f"Error: {e}"


# -----------------------------------------------------------------------------
class KVCache:
    """
    KV Cache designed for Flash Attention 3's flash_attn_with_kvcache API.

    Key differences from FA2-style cache:
    - Tensors are (B, T, H, D) not (B, H, T, D)
    - FA3 updates the cache in-place during flash_attn_with_kvcache
    - Position tracked per batch element via cache_seqlens tensor
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype, config=None):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        # Pre-allocate cache tensors: (n_layers, B, T, H, D)
        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        # Current sequence length per batch element (FA3 needs int32)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        # Linear states
        self.has_previous_state = False
        v_head_dim = getattr(config, 'linear_value_head_dim', 128)
        k_head_dim = getattr(config, 'linear_key_head_dim', 128)
        n_k_heads = getattr(config, 'linear_num_key_heads', 16)
        conv_dim = (k_head_dim * n_k_heads) * 2 + (v_head_dim * getattr(config, 'linear_num_value_heads', 16))
        conv_kernel_size = getattr(config, 'linear_conv_kernel_dim', 4)

        self.linear_conv_states = [torch.zeros(batch_size, conv_dim, conv_kernel_size - 1, device=device, dtype=dtype) for _ in range(num_layers)]
        self.linear_recurrent_states = [torch.zeros(batch_size, n_k_heads, k_head_dim, v_head_dim, device=device, dtype=dtype) for _ in range(num_layers)]

        # Previous token's normalized embedding for smear (set by model forward pass)
        self.prev_embedding = None

    def reset(self):
        """Reset cache to empty state."""
        self.cache_seqlens.zero_()
        self.has_previous_state = False
        for s in self.linear_conv_states: s.zero_()
        for s in self.linear_recurrent_states: s.zero_()
        self.prev_embedding = None

    def get_pos(self):
        """Get current position (assumes all batch elements at same position)."""
        return self.cache_seqlens[0].item()

    def get_layer_cache(self, layer_idx):
        """Return (k_cache, v_cache) views for a specific layer."""
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens):
        """Advance the cache position by num_tokens."""
        self.cache_seqlens += num_tokens
        self.has_previous_state = True

    def prefill(self, other):
        """
        Copy cached KV from another cache into this one.
        Used when we do batch=1 prefill and then want to generate multiple samples in parallel.
        """
        assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
        assert self.n_layers == other.n_layers and self.n_heads == other.n_heads and self.head_dim == other.head_dim
        assert self.max_seq_len >= other.max_seq_len
        other_pos = other.get_pos()
        self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
        self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
        self.cache_seqlens.fill_(other_pos)
        self.has_previous_state = other.has_previous_state
        for i in range(self.n_layers):
            self.linear_conv_states[i] = other.linear_conv_states[i].expand(self.batch_size, -1, -1).clone()
            self.linear_recurrent_states[i] = other.linear_recurrent_states[i].expand(self.batch_size, -1, -1, -1).clone()

        if other.prev_embedding is not None:
            self.prev_embedding = other.prev_embedding.expand(self.batch_size, -1, -1).clone()

# -----------------------------------------------------------------------------
@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)

# -----------------------------------------------------------------------------

class RowState:
    """Per-row state tracking during generation."""
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or []
        self.forced_tokens = deque()       # queue of tokens to force-inject
        self.text_buf = ""                 # decoded text of tokens generated so far (for tool call detection)
        self.in_tool_call = False          # currently inside a <tool_call> block
        self.tool_call_buf = ""            # text accumulated inside the tool call block
        self.completed = False


class Engine:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(self, tokens, tools=None, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """
        Generate tokens from a prompt.

        Args:
            tokens: list[int] — prompt token ids
            tools: list[callable] | None — Python functions the model may call.
                   Each function's __name__ must match what the model uses in <function=NAME>.
                   Parameter types are inferred from annotations for automatic coercion.
            num_samples: number of independent samples to generate in parallel
            max_tokens: maximum number of tokens to generate (None = unlimited)
            temperature: sampling temperature (0.0 = greedy)
            top_k: top-k sampling (None = full softmax)
            seed: RNG seed

        Yields:
            (token_column, token_masks) where:
              - token_column: list[int] of length num_samples — the token chosen for each row
              - token_masks: list[int] of length num_samples — 1 if sampled, 0 if forced (tool output)
        """
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        device = self.model.get_device()
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # EOS tokens for Qwen3.5:
        # <|im_end|> ends the assistant turn in chat format
        # <|endoftext|> is the document-level EOS / pad token
        im_end_id = self.tokenizer.encode_special("<|im_end|>")
        eos_id = self.tokenizer.token_to_id("<|endoftext|>")
        eos_ids = {t for t in [im_end_id, eos_id] if t is not None}

        # 1) Batch-1 prefill of the prompt
        m = self.model.config
        kv_model_kwargs = {"num_heads": m.n_kv_groups, "head_dim": m.head_dim, "num_layers": m.n_layers, "config": m}
        kv_cache_prefill = KVCache(batch_size=1, seq_len=len(tokens), device=device, dtype=dtype, **kv_model_kwargs)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :].expand(num_samples, -1)  # (num_samples, vocab_size)

        # 2) Replicate the KV cache for all samples
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else getattr(self.model.config, 'context_length', 4096)
        kv_cache_decode = KVCache(batch_size=num_samples, seq_len=kv_length_hint, device=device, dtype=dtype, **kv_model_kwargs)
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill

        # 3) Initialize per-row state
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) Main generation loop
        num_generated = 0
        while True:
            if max_tokens is not None and num_generated >= max_tokens:
                break
            if all(state.completed for state in row_states):
                break

            next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
            sampled_tokens = next_ids[:, 0].tolist()

            token_column = []
            token_masks = []
            for i, state in enumerate(row_states):
                is_forced = len(state.forced_tokens) > 0
                token_masks.append(0 if is_forced else 1)
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                state.current_tokens.append(next_token)

                # EOS check
                if next_token in eos_ids:
                    state.completed = True
                    continue

                # Skip tool-call logic entirely when tools=None (disabled)
                if tools is None:
                    continue

                # Decode this token and append to the text buffer
                chunk = self.tokenizer.decode([next_token])
                state.text_buf += chunk

                # Detect entry into a <tool_call> block
                if not state.in_tool_call:
                    if "<tool_call>" in state.text_buf:
                        state.in_tool_call = True
                        # Reset tool_call_buf to everything after <tool_call>
                        state.tool_call_buf = state.text_buf.split("<tool_call>", 1)[1]
                        state.text_buf = ""  # reset to avoid repeated triggers
                elif state.in_tool_call:
                    state.tool_call_buf += chunk
                    if "</tool_call>" in state.tool_call_buf:
                        state.in_tool_call = False
                        full_block = "<tool_call>" + state.tool_call_buf
                        state.tool_call_buf = ""
                        parsed = parse_tool_call(full_block)
                        if parsed is not None:
                            func_name, kwargs = parsed
                            result = dispatch_tool(func_name, kwargs, tools)
                            response = f"\n<tool_response>\n{result}\n</tool_response>\n"
                            response_tokens = self.tokenizer.encode(response)
                            state.forced_tokens.extend(response_tokens)

            yield token_column, token_masks
            num_generated += 1

            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            logits = self.model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]

    def generate_batch(self, tokens, tools=None, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that returns the final token sequences.

        Returns:
            results: list[list[int]] — generated token sequences (prompt + response, EOS excluded)
            masks:   list[list[int]] — per-token masks (1 = sampled, 0 = forced / prompt)
        """
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples

        im_end_id = self.tokenizer.encode_special("<|im_end|>")
        eos_id = self.tokenizer.token_to_id("<|endoftext|>")
        eos_ids = {t for t in [im_end_id, eos_id] if t is not None}

        for token_column, token_masks in self.generate(tokens, tools=tools, num_samples=num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token in eos_ids:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            if all(completed):
                break

        return results, masks


if __name__ == "__main__":
    """
    Quick inline test: verify that the naive model.generate() and Engine.generate()
    produce identical token sequences.
    """
    import time
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    model, tokenizer, meta = load_pretrained_hf("/home/truongnp5/Desktop/qwen35/Qwen3.5-0.8B", device, phase="eval")
    kwargs = dict(max_tokens=64, temperature=0.0)
    prompt_tokens = tokenizer.encode("The chemical formula of water is")

    # Reference: model.generate()
    generated_tokens = []
    torch.cuda.synchronize()
    t0 = time.time()
    for token in model.generate(prompt_tokens, **kwargs):
        generated_tokens.append(token)
        print(tokenizer.decode([token]), end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Reference time: {t1 - t0:.2f}s")
    reference_ids = generated_tokens

    # Engine.generate()
    generated_tokens = []
    engine = Engine(model, tokenizer)
    torch.cuda.synchronize()
    t0 = time.time()
    for token_column, token_masks in engine.generate(prompt_tokens, num_samples=1, **kwargs):
        token = token_column[0]
        generated_tokens.append(token)
        print(tokenizer.decode([token]), end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Engine time: {t1 - t0:.2f}s")

    for i in range(min(len(reference_ids), len(generated_tokens))):
        if reference_ids[i] != generated_tokens[i]:
            print(f"Mismatch at {i}: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"Match: {reference_ids == generated_tokens}")
