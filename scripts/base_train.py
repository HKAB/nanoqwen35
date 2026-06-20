"""
Train model. From root directory of the project, run as:

python -m scripts.base_train --pretrained-model-path ./Qwen3.5-0.8B --dataset-path /path/to/dataset --run my_run_name

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# Persistent torch.compile cache — avoids recompiling on every restart
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "nanoqwen35", "inductor"))
import gc
import json
import time
import math
import argparse
from contextlib import contextmanager

import wandb
import torch
import torch.distributed as dist
import torch._inductor.config as inductor_config
inductor_config.fx_graph_cache = True  # persist compiled FX graphs to disk

from nanoqwen35.qwen import Linear
from nanoqwen35.dataloader import (
    pretrain_loader_with_state,
    pretrain_loader,
)
from nanoqwen35.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from nanoqwen35.tokenizer import get_tokenizer
from nanoqwen35.checkpoint_manager import save_checkpoint, load_checkpoint, load_pretrained_hf
from nanoqwen35.loss_eval import evaluate_loss
from nanoqwen35.engine import Engine
from nanoqwen35.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
parser.add_argument("--wandb-project", type=str, default="nanoqwen35", help="wandb project name")
parser.add_argument("--wandb-entity", type=str, default=None, help="wandb entity/team name (leave blank for personal account)")
parser.add_argument("--wandb-id", type=str, default=None, help="wandb run ID — pass the previous run ID with --resume-from-step to continue the same dashboard run")
parser.add_argument("--wandb-tags", type=str, default=None, help="comma-separated tags, e.g. '0.8B,pretrain,fp8'")
parser.add_argument("--wandb-notes", type=str, default=None, help="free-text notes attached to this wandb run")
parser.add_argument("--wandb-watch", action="store_true", help="log gradient/weight histograms via wandb.watch() — adds overhead, use for debugging")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU and torchao)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)")
# Model architecture
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
    # (window pattern not used for Qwen3.5)
# Training horizon (manual for continued pretraining)
parser.add_argument("--num-iterations", type=int, default=2000, help="number of optimization steps (manual)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. reduce to 16,8,4,... if OOM.")
parser.add_argument("--total-batch-size", type=int, default=131072, help="total batch size in tokens (manual)")
parser.add_argument("--embedding-lr", type=float, default=0.03, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.003, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.10, help="weight decay for matrix params")
parser.add_argument("--optimizer", type=str, default="muon", choices=["muon", "adamw"], help="optimizer for matrix params: muon (default) or adamw (for debugging)")
parser.add_argument("--matrix-lr", type=float, default=0.005, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.05, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--warmup-steps", type=int, default=100, help="number of steps for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.2, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.1, help="final LR as fraction of initial LR")
parser.add_argument("--gradient-checkpointing", action="store_true", help="recompute activations during backward to save memory (allows larger --device-batch-size)")
parser.add_argument("--no-compile", action="store_true", help="disable torch.compile (useful for debugging or unsupported hardware)")
parser.add_argument("--logit-softcap", type=float, default=0.0, help="tanh softcap applied to logits before cross-entropy loss (0 = disabled, 15-30 typical)")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val loss every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=80*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
parser.add_argument(
    "--pretrained-model-path",
    type=str,
    default="./Qwen3.5-0.8B",
    help="path or HF repo id of pretrained model/tokenizer",
)
parser.add_argument("--dataset-root", type=str, required=True, help="root of merged flat shards (output of pretokenize_and_merge.py)")
args = parser.parse_args()
user_config = vars(args).copy()  # for logging
# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
if use_dummy_wandb:
    wandb_run = DummyWandb()
else:
    tags = [t.strip() for t in args.wandb_tags.split(",")] if args.wandb_tags else None
    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        id=args.wandb_id,
        name=args.run,
        config=user_config,
        tags=tags,
        notes=args.wandb_notes,
        resume="allow",  # safe no-op when id=None; resumes the run when id is provided
    )
    # Use "step" as the primary x-axis for every metric panel
    wandb.define_metric("step")
    wandb.define_metric("*", step_metric="step")

# Flash Attention status
from nanoqwen35.flash_attention import USE_FA3
using_fa3 = USE_FA3
if using_fa3:
    print0("✓ Using Flash Attention 3 (Hopper GPU detected), efficient, new and awesome.")
else:
    print0("!" * 80)
    if HAS_FA3 and COMPUTE_DTYPE != torch.bfloat16:
        print0(f"WARNING: Flash Attention 3 only supports bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}. Using PyTorch SDPA fallback")
    else:
        print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")

    print0("!" * 80)

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
tokenizer = get_tokenizer("Qwen/Qwen3.5-0.8B-Base")
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model

# Load the pretrained model
print0(f"Loading pretrained model from: {args.pretrained_model_path}")
model, tokenizer, meta_data_loaded = load_pretrained_hf(args.pretrained_model_path, device, phase="train")
model_config_kwargs = meta_data_loaded["model_config"]
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else "pretrained_0.8B"
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # free up this memory after the copy

# Gradient checkpointing (must be set before torch.compile)
if args.gradient_checkpointing:
    model.enable_gradient_checkpointing()
    print0("✓ Gradient checkpointing enabled (activation memory saved, ~33% extra compute)")

# Logit softcap
if args.logit_softcap > 0:
    model.logit_softcap = args.logit_softcap
    print0(f"✓ Logit softcap enabled: {args.logit_softcap}")

# -----------------------------------------------------------------------------
# FP8 training initialization and management (this has to be done before torch.compile)

# Convert Linear layers to Float8Linear if --fp8 is set
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        # our custom fp8 is simpler than torchao, written for exact API compatibility
        from nanoqwen35.fp8 import Float8LinearConfig, convert_to_float8_training
        # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        # Filter: dims must be divisible by 16 (FP8 hardware requirement) large enough
        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            if min(mod.in_features, mod.out_features) < 128:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        num_skipped = num_linear - num_fp8
        print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} (too small)")

# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """
    import torch.nn as nn

    # Find all Float8Linear modules and their locations
    fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # No FP8 modules, nothing to do
        return

    # Swap Float8Linear -> Linear (our custom class that casts weights to match input dtype)
    # Use device="meta" to avoid VRAM spike - the weight tensor will be swapped in afterwards
    for parent, attr_name, fp8_module in fp8_locations:
        linear = Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device="meta",  # Use meta device to avoid unnecessary VRAM allocation
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # share, don't copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # Restore Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# Compile the model

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
if args.no_compile:
    print0("torch.compile disabled (--no-compile flag set)")
else:
    model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe
if args.wandb_watch and not use_dummy_wandb:
    wandb.watch(orig_model, log="gradients", log_freq=100, log_graph=False)
    print0("✓ wandb.watch() enabled: gradient histograms will be logged every 100 steps")

# -----------------------------------------------------------------------------
# Manual training setup for continued pretraining (no scaling-law auto-tuning)

# Get parameter counts for logging
param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops(args.device_batch_size*ddp_world_size, args.max_seq_len)
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Manually configured batch size / LR / WD
if args.total_batch_size <= 0:
    raise ValueError("--total-batch-size must be > 0")
total_batch_size = args.total_batch_size
batch_lr_scale = 1.0  # manual mode: no implicit LR scaling
weight_decay_scaled = args.weight_decay  # manual mode: no implicit WD scaling

print0("Using manual training config (continued pretraining mode, scaling laws disabled).")
print0(f"Configured total batch size: {total_batch_size:,} tokens")
print0(f"Configured LRs: embedding={args.embedding_lr}, unembedding={args.unembedding_lr}, matrix={args.matrix_lr}, scalar={args.scalar_lr}")
print0(f"Configured weight decay: {weight_decay_scaled}")

# -----------------------------------------------------------------------------
# Calculate the number of iterations we will train for and set up the various schedulers

if args.num_iterations <= 0:
    raise ValueError("--num-iterations must be > 0 in manual training mode")
num_iterations = args.num_iterations
print0(f"Using manually configured number of iterations: {num_iterations:,}")

total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# Learning rate schedule (linear warmup, constant, linear warmdown)
def get_lr_multiplier(it):
    warmup_iters = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# Momentum scheduler for Muon optimizer (warms up to 0.97, warms down to 0.90 during LR warmdown)
def get_muon_momentum(it):
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97

# Weight decay scheduler for Muon optimizer (cosine decay to zero over the course of training)
def get_weight_decay(it):
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

# -----------------------------------------------------------------------------
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
    use_muon=(args.optimizer == "muon"),
)
print0(f"Optimizer: {args.optimizer} for matrix params")

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# -----------------------------------------------------------------------------
# GradScaler for fp16 training (bf16/fp32 don't need it — bf16 has the same exponent range as fp32)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]

from nanoqwen35.dataset import get_merged_metadata
_merged_meta = get_merged_metadata(args.dataset_root)
assert _merged_meta is not None, (
    f"merged_metadata.json not found in {args.dataset_root}. "
    "Run: python -m scripts.pretokenize_and_merge --source-root ... --output-root ..."
)
print0(f"Merged dataset: T={_merged_meta['T']}, {_merged_meta['num_train_rows']:,} train rows, {_merged_meta['num_train_shards']} shards")
train_loader = pretrain_loader_with_state(
    args.device_batch_size, args.max_seq_len, split="train",
    dataset_root=args.dataset_root, device=device,
    resume_state_dict=dataloader_resume_state_dict,
)
build_val_loader = lambda: pretrain_loader(
    args.device_batch_size, args.max_seq_len, split="val",
    dataset_root=args.dataset_root, device=device,
)
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Training loop

# Loop state (variables updated by the training loop)
if not resuming:
    step = 0
    val_loss = None # will be set if eval_every > 0
    min_val_loss = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_loss = meta_data["val_loss"]
    min_val_loss = loop_state["min_val_loss"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# -----------------------------------------------------------------------------
# Pre-training kernel warmup
# FLA/tilelang JIT kernels are compiled lazily on first use and can take >10 min.
# Running a forward+backward warmup on all ranks before any NCCL collective is
# launched prevents NCCL timeout from hitting the default 10-minute limit.
if ddp and device_type == "cuda":
    print0("Pre-training warmup: compiling kernels on all ranks (forward + backward)...")
    model.train()
    _wx = torch.zeros((args.device_batch_size, args.max_seq_len), dtype=torch.long, device=device)
    _wy = torch.zeros((args.device_batch_size, args.max_seq_len), dtype=torch.long, device=device)
    _wl = model(_wx, _wy)   # compile forward (and torch.compile graph)
    _wl.backward()          # compile backward (triggers FLA chunk_bwd/tilelang kernels)
    model.zero_grad(set_to_none=True)
    del _wx, _wy, _wl
    dist.barrier()  # wait for ALL ranks to finish compilation before any NCCL collective
    print0("Kernel warmup complete — all ranks synchronized.")

# Go!
while True:
    last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val loss (all ranks participate)
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(model):
            val_loss = evaluate_loss(model, val_loader, eval_steps)
        print0(f"Step {step:05d} | Validation loss: {val_loss:.6f}")
        if val_loss < min_val_loss:
            min_val_loss = val_loss
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/loss": val_loss,
        })
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model):
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "Một cây làm chẳng nên non,",
            "Thấy Tấm bắt được một giỏ đầy, Cám bảo chị:",
            "Trăm năm trong cõi người ta,",
            "Cacbon có 2 hóa trị là",
            "Vừa gà vừa chó bó lại cho tròn 36 con 100 chân chẵn. Hỏi có bao nhiêu con gà, bao nhiêu con chó?"
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt)
            with disable_fp8(orig_model):
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            optimizer.state_dict(), # optimizer state
            { # metadata saved as json
                "step": step,
                "val_loss": val_loss, # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": total_batch_size,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_loss": min_val_loss,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            tokenizer=tokenizer,
            rank=ddp_rank,
        )

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        loss = model(x, y)
        train_loss = loss.detach() # for logging
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
    # step the optimizer
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    if scaler is not None:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
        # In distributed training, all ranks must agree on whether to skip the step.
        # Each rank may independently encounter inf/nan gradients, so we all-reduce
        # the found_inf flag (MAX = if any rank found inf, all ranks skip).
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
        optimizer.step()
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item() # .item() is a CPU-GPU sync point
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging (CPU action only)
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = f"ep{dataloader_state_dict['epoch']} pq{dataloader_state_dict['pq_idx']} rg{dataloader_state_dict['rg_idx']}"
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    if step % 100 == 0:
        lr_log = {f"train/lr_{g['kind']}": g["lr"] for g in optimizer.param_groups}
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/grad_norm": grad_norm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
            "system/gpu_memory_mb": get_max_memory() / 1024 / 1024,
            **lr_log,
        }
        wandb_run.log(log_data)

    # state update
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    if first_step_of_run:
        gc.collect() # manually collect a lot of garbage from setup
        gc.freeze() # immediately freeze all currently surviving objects and exclude them from GC
        gc.disable() # nuclear intervention here: disable GC entirely except:
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very, very long runs

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_loss is not None:
    print0(f"Minimum validation loss: {min_val_loss:.6f}")

# Log to report
from nanoqwen35.report import get_report
get_report().log(section="Base model training", data=[
    user_config, # CLI args
    { # stats about the training setup
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Configured number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "DDP world size": ddp_world_size,
        "warmup_steps": args.warmup_steps,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_frac": args.final_lr_frac,
    },
    { # stats about training outcomes
        "Minimum validation loss": min_val_loss if val_loss is not None else None,
        "Final validation loss": val_loss,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
