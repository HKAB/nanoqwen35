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
        # Qwen3.5 does not have a bos token
        return None
    

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

    def render_conversation(self, conversation, max_tokens=2048):
        ids, mask = [], []
        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            if token_ids is None:
                return
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # Merge system into user message if present
        if conversation["messages"][0]["role"] == "system":
            conversation = copy.deepcopy(conversation)
            messages = conversation["messages"]
            assert messages[1]["role"] == "user"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        else:
            messages = conversation["messages"]

        im_start = self.encode_special("<|im_start|>")
        im_end = self.encode_special("<|im_end|>")
        if im_start is None:
            im_start = self.encode_special("<|user_start|>")
            im_end = self.encode_special("<|user_end|>")

        for i, message in enumerate(messages):
            content = message["content"]
            if message["role"] == "user":
                add_tokens(im_start, 0)
                add_tokens(self.encode("user\n"), 0)
                add_tokens(self.encode(content), 0)
                add_tokens(im_end, 0)
                add_tokens(self.encode("\n"), 0)
            elif message["role"] == "assistant":
                add_tokens(im_start, 0)
                add_tokens(self.encode("assistant\n"), 0)
                add_tokens(self.encode(content), 1)
                add_tokens(im_end, 1)
                add_tokens(self.encode("\n"), 1)

        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def render_for_completion(self, conversation):
        conversation = copy.deepcopy(conversation)
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant"
        messages.pop()
        ids, mask = self.render_conversation(conversation)
        
        im_start = self.encode_special("<|im_start|>")
        if im_start is None:
            im_start = self.encode_special("<|user_start|>")
        ids.append(im_start)
        ids.extend(self.encode("assistant\n"))
        return ids

def get_tokenizer(pretrained_dir=None):
    from nanoqwen35.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    if os.path.exists(os.path.join(tokenizer_dir, "tokenizer.json")):
        return HuggingFaceTokenizer.from_directory(tokenizer_dir)
    else:
        return HuggingFaceTokenizer.from_pretrained(pretrained_dir)

def get_token_bytes(device="cpu"):
    # Since we don't have token_bytes.pt anymore without training, we return None
    return None
