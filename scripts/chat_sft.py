"""
Supervised fine-tuning (SFT) or RL-preference training.
Run as:

    torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \\
        --dataset-root /path/to/sft_data \\
        --num-iterations 2000

Dataset format: parquet files under domain subdirectories, each with a 'messages' column.
'messages' is a list of {"role": "user"/"assistant"/"system", "content": "..."} dicts,
or a JSON-serialised string of the same structure.

For RL-style training where only the last assistant turn is supervised:
    --mask-history
"""

import gc
import argparse
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "nanoqwen35", "inductor"))
import time
import wandb
import torch
import torch._inductor.config as inductor_config
inductor_config.fx_graph_cache = True

from nanoqwen35.common import compute_init, compute_cleanup, print0, DummyWandb, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from nanoqwen35.checkpoint_manager import save_checkpoint, load_model, load_optimizer_state
from nanoqwen35.loss_eval import evaluate_loss
from nanoqwen35.dataloader import sft_loader, sft_pretokenized_loader
import torch.distributed as dist
from nanoqwen35.flash_attention import HAS_FA3
from nanoqwen35.engine import Engine
from scripts.chat_eval import run_chat_eval

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="SFT / RL fine-tuning")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model loading
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
parser.add_argument("--load-optimizer", type=int, default=1, help="warm-start optimizer from pretrained checkpoint (0=no, 1=yes)")
# Dataset
parser.add_argument("--dataset-root", type=str, required=True, help="root folder whose subdirectories are domains; each domain contains parquet files with a 'messages' column")
parser.add_argument("--buffer-size", type=int, default=128, help="per-domain conversation buffer size for best-fit packing")
parser.add_argument("--mask-history", action="store_true", help="only supervise the last assistant turn (RL-style); default: supervise all assistant turns (SFT-style)")
parser.add_argument("--pretokenized", action="store_true", help="dataset-root is an offline neat-packed SFT dataset (scripts/pretokenize.py --mode sft); uses block-diagonal attention")
# Training horizon
parser.add_argument("--num-iterations", type=int, required=True, help="number of optimisation steps")
# Batch sizes
parser.add_argument("--max-seq-len", type=int, default=None, help="max context length (default: inherit from pretrain checkpoint)")
parser.add_argument("--device-batch-size", type=int, default=None, help="per-device batch size (default: inherit from pretrain checkpoint)")
parser.add_argument("--total-batch-size", type=int, default=None, help="total batch size in tokens (default: inherit from pretrain checkpoint)")
# Optimisation
parser.add_argument("--embedding-lr", type=float, default=None, help="LR for embedding params (default: inherit from pretrain)")
parser.add_argument("--unembedding-lr", type=float, default=None, help="LR for unembedding params (default: inherit from pretrain)")
parser.add_argument("--matrix-lr", type=float, default=None, help="LR for matrix params (Muon; default: inherit from pretrain)")
parser.add_argument("--init-lr-frac", type=float, default=0.8, help="initial LR as fraction of base LR")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="fraction of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="fraction of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
# Evaluation
parser.add_argument("--eval-every", type=int, default=200, help="evaluate val loss every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=40*524288, help="tokens to use for val loss evaluation")
parser.add_argument("--chatcore-every", type=int, default=-1, help="evaluate ChatCORE metric every N steps (-1 = disable)")
parser.add_argument("--no-compile", action="store_true", help="disable torch.compile")
args = parser.parse_args()
assert args.num_iterations > 0, "--num-iterations must be > 0"
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float("inf")

use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanoqwen35-sft", name=args.run, config=user_config)

if not HAS_FA3:
    print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback. Training will be less efficient.")

# Load model + tokenizer
model, tokenizer, meta = load_model("base", device, phase="train", model_tag=args.model_tag, step=args.model_step)

# Inherit hyperparameters from pretrained checkpoint where not explicitly set
pretrain_user_config = meta.get("user_config", {})
for name, fallback, source in [
    ("max_seq_len",       2048,   meta),
    ("device_batch_size", 32,     meta),
    ("total_batch_size",  524288, meta),
    ("embedding_lr",      0.3,    pretrain_user_config),
    ("unembedding_lr",    0.004,  pretrain_user_config),
    ("matrix_lr",         0.02,   pretrain_user_config),
]:
    arg_val = getattr(args, name)
    pretrain_val = source.get(name)
    if arg_val is None:
        resolved = pretrain_val if pretrain_val is not None else fallback
        setattr(args, name, resolved)
        print0(f"Inherited {name}={resolved} from pretrained checkpoint")
    elif pretrain_val is not None and arg_val != pretrain_val:
        print0(f"NOTE: --{name.replace('_', '-')}={arg_val} overrides pretrained value of {pretrain_val}")
    else:
        print0(f"Using {name}={arg_val}")

orig_model = model
if args.pretokenized and not args.no_compile:
    print0("!" * 80)
    print0("WARNING: --pretokenized uses per-batch-varying cu_seqlens for block-diagonal")
    print0("attention, which is incompatible with torch.compile(dynamic=False). Forcing")
    print0("--no-compile. (Revisit dynamic=True once training is confirmed working.)")
    print0("!" * 80)
    args.no_compile = True
if args.no_compile:
    print0("torch.compile disabled (--no-compile flag set)")
else:
    model = torch.compile(model, dynamic=False)
model_config_kwargs = meta.get("model_config", {})
_flops_global_batch = args.device_batch_size * ddp_world_size
num_flops_per_token = model.estimate_flops(_flops_global_batch, args.max_seq_len) / (_flops_global_batch * args.max_seq_len)

tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert args.total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# Optimizer
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=0.0,
)

base_dir = get_base_dir()
if args.load_optimizer:
    optimizer_data = load_optimizer_state("base", device, rank=ddp_rank, model_tag=args.model_tag, step=args.model_step)
    if optimizer_data is not None:
        base_lrs = [group["lr"] for group in optimizer.param_groups]
        optimizer.load_state_dict(optimizer_data)
        del optimizer_data
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr
        for group in optimizer.param_groups:
            if group["kind"] == "muon":
                for p in group["params"]:
                    state = optimizer.state[p]
                    if "momentum_buffer" in state:
                        state["momentum_buffer"].zero_()
        print0("Loaded optimizer state (momentum buffers zeroed, second moments kept, LRs reset)")
    else:
        print0("WARNING: optimizer checkpoint not found, starting with fresh optimizer")

scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# Scale initial LR
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]

# LR schedule (warmup → constant → warmdown)
def get_lr_multiplier(step):
    progress = step / args.num_iterations
    if progress < args.warmup_ratio:
        return (progress + 1e-8) / args.warmup_ratio
    elif progress <= 1.0 - args.warmdown_ratio:
        return 1.0
    else:
        decay = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
        return (1 - decay) * 1.0 + decay * args.final_lr_frac

def get_muon_momentum(step):
    frac = min(step / 300, 1.0)
    return (1 - frac) * 0.85 + frac * 0.95

# -----------------------------------------------------------------------------
# Dataloaders
if args.pretokenized:
    print0("Loading SFT dataloader — mode: pretokenized (offline neat-packed, block-diagonal)")
    train_loader = sft_pretokenized_loader(
        args.device_batch_size, args.max_seq_len, split="train",
        dataset_root=args.dataset_root, device=device,
    )
    build_val_loader = lambda: sft_pretokenized_loader(
        args.device_batch_size, args.max_seq_len, split="val",
        dataset_root=args.dataset_root, device=device,
    )
else:
    mode_str = "RL (last-turn only)" if args.mask_history else "SFT (all assistant turns)"
    print0(f"Loading SFT dataloader — mode: {mode_str}")
    train_loader = sft_loader(
        args.device_batch_size, args.max_seq_len, split="train",
        dataset_root=args.dataset_root, tokenizer=tokenizer,
        device=device, buffer_size=args.buffer_size, mask_history=args.mask_history,
    )
    build_val_loader = lambda: sft_loader(
        args.device_batch_size, args.max_seq_len, split="val",
        dataset_root=args.dataset_root, tokenizer=tokenizer,
        device=device, buffer_size=args.buffer_size, mask_history=args.mask_history,
    )

# -----------------------------------------------------------------------------
# Training loop
def unpack_batch(batch):
    """sft_loader yields (x, y); sft_pretokenized_loader yields (x, y, cu_seqlens, position_ids)."""
    if len(batch) == 4:
        return batch
    x, y = batch
    return x, y, None, None

x, y, cu_seqlens, position_ids = unpack_batch(next(train_loader))
min_val_loss    = float("inf")
smooth_loss     = 0.0
ema_beta        = 0.9
total_train_time = 0.0
step = 0

while True:
    last_step = (step == args.num_iterations)

    # Synchronize last_step across all ranks to avoid hangs in distributed training
    if ddp:
        last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
        last_step = bool(last_step_tensor.item())

    # Validation loss
    if last_step or (args.eval_every > 0 and step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        tokens_per_eval = args.device_batch_size * args.max_seq_len * ddp_world_size
        eval_steps = max(1, args.eval_tokens // tokens_per_eval)
        val_loss = evaluate_loss(model, val_loader, eval_steps)
        print0(f"Step {step:05d} | val loss: {val_loss:.4f}")
        if val_loss < min_val_loss:
            min_val_loss = val_loss
        wandb_run.log({"step": step, "val/loss": val_loss, "total_training_time": total_train_time})
        model.train()

    # ChatCORE metric
    if args.chatcore_every > 0 and (last_step or (step > 0 and step % args.chatcore_every == 0)):
        model.eval()
        engine = Engine(orig_model, tokenizer)
        
        categorical_tasks = ["GlobalMMLU"]
        baseline_accuracies = {
            'GlobalMMLU': 0.25,  # Random baseline
        }
        task_results = {}
        for task_name in categorical_tasks:
            acc = run_chat_eval(task_name, model, tokenizer, engine, batch_size=args.device_batch_size)
            task_results[task_name] = acc
            print0(f"Step {step:05d} | ChatCORE {task_name}: {100*acc:.2f}%")
        
        def centered_mean(tasks):
            return sum((task_results[t] - baseline_accuracies[t]) / (1.0 - baseline_accuracies[t]) for t in tasks) / len(tasks)

        chatcore_cat = centered_mean(categorical_tasks)

        print0(f"Step {step:05d} | ChatCORE_cat: {chatcore_cat:.4f}")
        wandb_run.log({
            "step": step,
            "chatcore_cat": chatcore_cat,
            **{f"chatcore_{t}": task_results[t] for t in categorical_tasks},
        })
        model.train()

    # Save checkpoint
    if last_step:
        checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", f"step_{step:05d}")
        save_checkpoint(
            checkpoint_dir, step,
            orig_model.state_dict(),
            optimizer.state_dict(),
            {
                "step": step,
                "val_loss": val_loss if "val_loss" in dir() else None,
                "model_config": model_config_kwargs,
                "user_config": user_config,
            },
            tokenizer=tokenizer,
            rank=ddp_rank,
        )

    if last_step:
        break

    # -------------------------------------------------------------------------
    # Single training step
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        loss = model(x, y, cu_seqlens=cu_seqlens, position_ids=position_ids)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        x, y, cu_seqlens, position_ids = unpack_batch(next(train_loader))

    lrm            = get_lr_multiplier(step)
    muon_momentum  = get_muon_momentum(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group["kind"] == "muon":
            group["momentum"] = muon_momentum
    if scaler is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    model.zero_grad(set_to_none=True)

    synchronize()
    t1 = time.time()
    dt = t1 - t0
    step += 1
    # -------------------------------------------------------------------------

    smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * train_loss.item()
    debiased    = smooth_loss / (1 - ema_beta**step)
    pct_done    = 100 * step / args.num_iterations
    tok_per_sec = int(args.total_batch_size / dt)
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_train_time += dt
    print0(f"step {step:05d}/{args.num_iterations} ({pct_done:.1f}%) | loss: {debiased:.6f} | lrm: {lrm:.3f} | dt: {dt*1000:.1f}ms | tok/s: {tok_per_sec:,} | mfu: {mfu:.1f}% | time: {total_train_time/60:.1f}m")
    if step % 10 == 0:
        wandb_run.log({
            "step": step,
            "train/loss": debiased,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "total_training_time": total_train_time,
        })

    if step == 1:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif step % 5000 == 0:
        gc.collect()

# final stats
print0(f"Peak memory: {get_max_memory() / 1024 / 1024:.1f} MiB")
print0(f"Total training time: {total_train_time/60:.1f}m")
print0(f"Min val loss: {min_val_loss:.4f}")

from nanoqwen35.report import get_report
get_report().log(section="SFT", data=[
    user_config,
    {"num_iterations": step, "ddp_world_size": ddp_world_size},
    {"min_val_loss": min_val_loss},
])

wandb_run.finish()
compute_cleanup()
