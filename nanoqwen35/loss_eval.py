"""
A number of functions that help with evaluating a base model.
"""
import math
import torch
import torch.distributed as dist

@torch.no_grad()
def evaluate_loss(model, batches, steps):
    """
    Returns the average bits per token (bpt).
    """
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_tokens = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        loss2d = model(x, y, loss_reduction='none') # (B, T)
        loss2d = loss2d.view(-1) # flatten
        y = y.view(-1) # flatten
        
        valid = y >= 0
        total_loss += (loss2d * valid).sum()
        total_tokens += valid.sum()

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

    total_loss = total_loss.item()
    total_tokens = total_tokens.item()
    if total_tokens == 0:
        return float('inf')
    bpt = total_loss / (math.log(2) * total_tokens)
    return bpt
