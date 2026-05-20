"""
Utilities for saving and loading model/optim/state checkpoints.
"""
import os
import re
import glob
import json
import logging
import torch

from nanoqwen35.common import get_base_dir
from nanoqwen35.qwen import Qwen3_5Model, Qwen3_5ModelConfig
from nanoqwen35.tokenizer import HuggingFaceTokenizer, get_tokenizer
from nanoqwen35.common import setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)

def _patch_missing_config_keys(model_config_kwargs):
    """Add default values for new config keys missing in old checkpoints."""
    # Old models were trained with full context (no sliding window)
    if "window_pattern" not in model_config_kwargs:
        model_config_kwargs["window_pattern"] = "L"
        log0(f"Patching missing window_pattern in model config to 'L'")

def _patch_missing_keys(model_data, model_config):
    """Add default values for new parameters that may be missing in old checkpoints."""
    n_layer = model_config.n_layers
    # resid_lambdas defaults to 1.0 (identity scaling)
    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(n_layer)
        log0(f"Patching missing resid_lambdas in model data to 1.0")
    # x0_lambdas defaults to 0.0 (disabled)
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(n_layer)
        log0(f"Patching missing x0_lambdas in model data to 0.0")

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save the model state parameters
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        logger.info(f"Saved model parameters to: {model_path}")
        # Save the metadata dict as json
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"Saved metadata to: {meta_path}")
    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    if optimizer_data is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        torch.save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer state to: {optimizer_path}")

def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    # Load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    _patch_missing_config_keys(model_config_kwargs)
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = Qwen3_5ModelConfig(**model_config_kwargs)
    _patch_missing_keys(model_data, model_config)
    with torch.device("meta"):
        model = Qwen3_5Model(model_config)
    # Load the model state
    model.to_empty(device=device)
    model.init_weights() # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    # Load the Tokenizer
    tokenizer = get_tokenizer()
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model config vocab size {model_config_kwargs['vocab_size']}"
    return model, tokenizer, meta_data


def find_largest_model(checkpoints_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
    return last_step

# -----------------------------------------------------------------------------
# convenience functions that take into account nanoqwen35's directory structure

def load_pretrained_hf(pretrained_dir, device, phase="eval", **kwargs):
    import json
    import glob
    from safetensors.torch import load_file
    
    config_path = os.path.join(pretrained_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        hf_config = json.load(f)
    
    model_config_kwargs = {
        "vocab_size": hf_config["text_config"]["vocab_size"],
        "context_length": hf_config["text_config"]["max_position_embeddings"],
        "emb_dim": hf_config["text_config"]["hidden_size"],
        "n_heads": hf_config["text_config"]["num_attention_heads"],
        "n_layers": hf_config["text_config"]["num_hidden_layers"],
        "hidden_dim": hf_config["text_config"]["intermediate_size"],
        "head_dim": hf_config["text_config"].get("head_dim", 256),
        "qk_norm": hf_config["text_config"].get("qk_norm", True),
        "n_kv_groups": hf_config["text_config"]["num_key_value_heads"],
        "rope_base": hf_config["text_config"].get("rope_parameters", {}).get("rope_theta", 1000000.0),
        "partial_rotary_factor": hf_config["text_config"].get("rope_parameters", {}).get("partial_rotary_factor", 1.0),
        "rms_norm_eps": hf_config["text_config"]["rms_norm_eps"],
        "layer_types": hf_config["text_config"]["layer_types"],
        "linear_num_value_heads": hf_config["text_config"].get("linear_num_value_heads", 16),
        "linear_num_key_heads": hf_config["text_config"].get("linear_num_key_heads", 16),
        "linear_key_head_dim": hf_config["text_config"].get("linear_key_head_dim", 128),
        "linear_value_head_dim": hf_config["text_config"].get("linear_value_head_dim", 128),
        "linear_conv_kernel_dim": hf_config["text_config"].get("linear_conv_kernel_dim", 4),
        "hidden_act": hf_config["text_config"].get("hidden_act", "silu"),
    }
    
    model_config = Qwen3_5ModelConfig(**model_config_kwargs)
    with torch.device("meta"):
        model = Qwen3_5Model(model_config)
    model.to_empty(device=device)
    
    st_files = glob.glob(os.path.join(pretrained_dir, "*.safetensors"))
    hf_state_dict = {}
    for st_file in st_files:
        hf_state_dict.update(load_file(st_file, device=str(device)))
        
    state_dict = {}
    state_dict["transformer.wte.weight"] = hf_state_dict["model.language_model.embed_tokens.weight"]
    if "lm_head.weight" in hf_state_dict:
        state_dict["lm_head.weight"] = hf_state_dict["lm_head.weight"]
    elif hf_config.get("tie_word_embeddings", False) or hf_config.get("text_config", {}).get("tie_word_embeddings", False):
        state_dict["lm_head.weight"] = hf_state_dict["model.language_model.embed_tokens.weight"]
        
    state_dict["final_norm.weight"] = hf_state_dict["model.language_model.norm.weight"]
    
    for i in range(model_config.n_layers):
        prefix = f"transformer.h.{i}."
        hf_prefix = f"model.language_model.layers.{i}."
        
        state_dict[prefix + "norm1.weight"] = hf_state_dict[hf_prefix + "input_layernorm.weight"]
        state_dict[prefix + "norm2.weight"] = hf_state_dict[hf_prefix + "post_attention_layernorm.weight"]
        
        state_dict[prefix + "ff.fc3.weight"] = hf_state_dict[hf_prefix + "mlp.down_proj.weight"]
        state_dict[prefix + "ff.fc1.weight"] = hf_state_dict[hf_prefix + "mlp.gate_proj.weight"]
        state_dict[prefix + "ff.fc2.weight"] = hf_state_dict[hf_prefix + "mlp.up_proj.weight"]
        
        ltype = model_config.layer_types[i]
        if ltype == "full_attention":
            state_dict[prefix + "token_mixer.W_query.weight"] = hf_state_dict[hf_prefix + "self_attn.q_proj.weight"]
            state_dict[prefix + "token_mixer.W_key.weight"] = hf_state_dict[hf_prefix + "self_attn.k_proj.weight"]
            state_dict[prefix + "token_mixer.W_value.weight"] = hf_state_dict[hf_prefix + "self_attn.v_proj.weight"]
            state_dict[prefix + "token_mixer.out_proj.weight"] = hf_state_dict[hf_prefix + "self_attn.o_proj.weight"]
            state_dict[prefix + "token_mixer.q_norm.weight"] = hf_state_dict[hf_prefix + "self_attn.q_norm.weight"]
            state_dict[prefix + "token_mixer.k_norm.weight"] = hf_state_dict[hf_prefix + "self_attn.k_norm.weight"]
        elif ltype == "linear_attention":
            state_dict[prefix + "token_mixer.in_proj_qkv.weight"] = hf_state_dict[hf_prefix + "linear_attn.in_proj_qkv.weight"]
            state_dict[prefix + "token_mixer.in_proj_z.weight"] = hf_state_dict[hf_prefix + "linear_attn.in_proj_z.weight"]
            state_dict[prefix + "token_mixer.in_proj_b.weight"] = hf_state_dict[hf_prefix + "linear_attn.in_proj_b.weight"]
            state_dict[prefix + "token_mixer.in_proj_a.weight"] = hf_state_dict[hf_prefix + "linear_attn.in_proj_a.weight"]
            state_dict[prefix + "token_mixer.out_proj.weight"] = hf_state_dict[hf_prefix + "linear_attn.out_proj.weight"]
            state_dict[prefix + "token_mixer.conv1d.weight"] = hf_state_dict[hf_prefix + "linear_attn.conv1d.weight"]
            state_dict[prefix + "token_mixer.norm.weight"] = hf_state_dict[hf_prefix + "linear_attn.norm.weight"]
            state_dict[prefix + "token_mixer.A_log"] = hf_state_dict[hf_prefix + "linear_attn.A_log"]
            state_dict[prefix + "token_mixer.dt_bias"] = hf_state_dict[hf_prefix + "linear_attn.dt_bias"]
            
    model.load_state_dict(state_dict, strict=True, assign=True)
    if phase == "eval":
        model.eval()
    else:
        model.train()
        
    tokenizer = HuggingFaceTokenizer.from_directory(pretrained_dir)
    return model, tokenizer, {"model_config": model_config_kwargs}

def load_model(source, *args, **kwargs):
    model_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)


def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data

def load_optimizer_state(source, device, rank, model_tag=None, step=None):
    """Load just the optimizer shard for a given rank, without re-loading the model."""
    model_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    if model_tag is None:
        model_tag = find_largest_model(checkpoints_dir)
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        step = find_last_step(checkpoint_dir)
    optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
    if not os.path.exists(optimizer_path):
        log0(f"Optimizer checkpoint not found: {optimizer_path}")
        return None
    log0(f"Loading optimizer state from {optimizer_path}")
    optimizer_data = torch.load(optimizer_path, map_location=device)
    return optimizer_data
