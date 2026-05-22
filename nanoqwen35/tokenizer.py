"""
Tokenizer for Qwen3.5.
We use HuggingFace Tokenizer since Qwen3.5 has a fixed vocabulary and we do not need to train one.
"""

import os
import copy
from tokenizers import Tokenizer as HFTokenizer

class HuggingFaceTokenizer:
    """Light wrapper around HuggingFace Tokenizer for some utilities"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path):
        # init from a HuggingFace pretrained tokenizer (e.g. "Qwen/Qwen1.5-0.5B")
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        # init from a local directory on disk
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens

    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)
    
    def token_to_id(self, token):
        return self.tokenizer.token_to_id(token)

    def _encode_one(self, text, prepend=None, append=None, num_threads=None):
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids

    def encode_special(self, text):
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self):
        return self.tokenizer.token_to_id("<|endoftext|>")

    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir):
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")

    def render_conversation(self, conversation, max_tokens=2048, mask_history=False):
        """
        Render a conversation dict into token ids and loss mask.

        mask_history=False (default): all assistant turns are supervised (mask=1).
        mask_history=True: only the last assistant turn (after the last real user query)
            is supervised — useful when only the final response is the training target.
        """
        import json

        ids, mask = [], []

        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            if token_ids is None:
                return
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        messages = conversation["messages"]
        tools = conversation.get("tools") or []

        im_start = self.encode_special("<|im_start|>")
        im_end = self.encode_special("<|im_end|>")
        if im_start is None:
            im_start = self.encode_special("<|user_start|>")
            im_end = self.encode_special("<|user_end|>")

        # Find last_query_index: last user message that is NOT a bare tool_response.
        # Assistant turns after this index receive <think> wrapping.
        last_query_index = 0
        for idx, msg in enumerate(messages):
            if msg["role"] == "user":
                c = msg.get("content") or ""
                if isinstance(c, str):
                    c = c.strip()
                    if not (c.startswith("<tool_response>") and c.endswith("</tool_response>")):
                        last_query_index = idx
                else:
                    last_query_index = idx

        # --- system / tools header ---
        start_idx = 0
        if tools:
            add_tokens(im_start, 0)
            add_tokens(self.encode("system\n"), 0)
            add_tokens(self.encode("# Tools\n\nYou have access to the following functions:\n\n<tools>"), 0)
            for tool in tools:
                add_tokens(self.encode("\n" + json.dumps(tool)), 0)
            add_tokens(self.encode("\n</tools>"), 0)
            tool_instructions = (
                "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
                "<tool_call>\n<function=example_function_name>\n"
                "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
                "<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
                "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
                "<IMPORTANT>\nReminder:\n"
                "- Function calls MUST follow the specified format: an inner <function=...></function> "
                "block must be nested within <tool_call></tool_call> XML tags\n"
                "- Required parameters MUST be specified\n"
                "- You may provide optional reasoning for your function call in natural language BEFORE "
                "the function call, but NOT after\n"
                "- If there is no function call available, answer the question like normal with your "
                "current knowledge and do not tell the user about function calls\n</IMPORTANT>"
            )
            add_tokens(self.encode(tool_instructions), 0)
            if messages[0]["role"] == "system":
                sys_content = (messages[0].get("content") or "").strip()
                if sys_content:
                    add_tokens(self.encode("\n\n" + sys_content), 0)
                start_idx = 1
            add_tokens(im_end, 0)
            add_tokens(self.encode("\n"), 0)
        elif messages and messages[0]["role"] == "system":
            sys_content = (messages[0].get("content") or "").strip()
            add_tokens(im_start, 0)
            add_tokens(self.encode("system\n"), 0)
            add_tokens(self.encode(sys_content), 0)
            add_tokens(im_end, 0)
            add_tokens(self.encode("\n"), 0)
            start_idx = 1

        # --- message loop ---
        for i, message in enumerate(messages[start_idx:], start=start_idx):
            role = message["role"]
            content = message.get("content") or ""

            is_after_last_query = i > last_query_index
            # When mask_history=True only supervise turns after the last real user query.
            assistant_mask = 1 if (not mask_history or is_after_last_query) else 0

            if role == "system":
                raise ValueError("System message must be at the beginning.")

            elif role == "user":
                add_tokens(im_start, 0)
                add_tokens(self.encode("user\n"), 0)
                add_tokens(self.encode(content), 0)
                add_tokens(im_end, 0)
                add_tokens(self.encode("\n"), 0)

            elif role == "assistant":
                # Split out reasoning content if not already a separate field.
                reasoning_content = message.get("reasoning_content") or ""
                if not reasoning_content and "</think>" in content:
                    parts = content.split("</think>")
                    reasoning_content = parts[0].split("<think>")[-1].strip("\n")
                    content = parts[-1].lstrip("\n")
                reasoning_content = reasoning_content.strip()

                add_tokens(im_start, 0)
                add_tokens(self.encode("assistant\n"), 0)

                if is_after_last_query:
                    add_tokens(self.encode("<think>\n"), assistant_mask)
                    if reasoning_content:
                        add_tokens(self.encode(reasoning_content), assistant_mask)
                    add_tokens(self.encode("\n</think>\n\n"), assistant_mask)

                if content:
                    add_tokens(self.encode(content), assistant_mask)

                # Tool calls embedded in the assistant turn.
                tool_calls = message.get("tool_calls") or []
                for j, tool_call in enumerate(tool_calls):
                    fn = tool_call.get("function", tool_call)
                    if j == 0:
                        prefix = "\n\n<tool_call>\n" if content.strip() else "<tool_call>\n"
                    else:
                        prefix = "\n<tool_call>\n"
                    add_tokens(self.encode(prefix + f"<function={fn['name']}>\n"), assistant_mask)
                    args = fn.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    for arg_name, arg_value in args.items():
                        if isinstance(arg_value, (dict, list)):
                            arg_str = json.dumps(arg_value)
                        else:
                            arg_str = str(arg_value)
                        add_tokens(self.encode(f"<parameter={arg_name}>\n{arg_str}\n</parameter>\n"), assistant_mask)
                    add_tokens(self.encode("</function>\n</tool_call>"), assistant_mask)

                add_tokens(im_end, assistant_mask)
                add_tokens(self.encode("\n"), assistant_mask)

            elif role == "tool":
                # Consecutive tool messages are batched inside a single user turn.
                prev_role = messages[start_idx + (i - start_idx) - 1]["role"] if i > start_idx else None
                if prev_role != "tool":
                    add_tokens(im_start, 0)
                    add_tokens(self.encode("user"), 0)
                add_tokens(self.encode("\n<tool_response>\n"), 0)
                add_tokens(self.encode(content), 0)
                add_tokens(self.encode("\n</tool_response>"), 0)
                next_idx = i + 1
                next_role = messages[next_idx]["role"] if next_idx < len(messages) else None
                if next_role != "tool":
                    add_tokens(im_end, 0)
                    add_tokens(self.encode("\n"), 0)

            else:
                raise ValueError(f"Unexpected message role: {role!r}")

        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def render_for_completion(self, conversation, enable_thinking=True):
        conversation = copy.deepcopy(conversation)
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant"
        messages.pop()
        ids, _ = self.render_conversation(conversation)

        im_start = self.encode_special("<|im_start|>")
        if im_start is None:
            im_start = self.encode_special("<|user_start|>")
        ids.append(im_start)
        ids.extend(self.encode("assistant\n"))
        if enable_thinking:
            ids.extend(self.encode("<think>\n"))
        else:
            ids.extend(self.encode("<think>\n\n</think>\n\n"))
        return ids

def get_tokenizer(model_id):
    return HuggingFaceTokenizer.from_pretrained(model_id)

def get_token_bytes(device="cpu"):
    # Since we don't have token_bytes.pt anymore without training, we return None
    return None
