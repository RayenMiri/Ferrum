import json
import math
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

import config
from model import FerrumNet, save_checkpoint


def build_optimizer(model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
    )


def build_scheduler(optimizer, steps_per_epoch):
    warmup_steps = config.WARMUP_EPOCHS * steps_per_epoch
    total_steps = config.MAX_EPOCHS * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def training_step(model, batch, optimizer, device):
    state, policy_target, value_target, legal_mask = batch
    state = state.to(device)
    policy_target = policy_target.to(device)
    value_target = value_target.to(device)
    legal_mask = legal_mask.to(device)

    policy_logits, value_out = model(state)

    policy_logits = policy_logits.masked_fill(~legal_mask, -torch.inf)
    policy_loss = F.cross_entropy(policy_logits, policy_target)
    value_loss = F.mse_loss(value_out, value_target) * config.VALUE_LOSS_WEIGHT
    loss = policy_loss + value_loss

    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss {loss.item()} — check legal_mask data")

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()


def evaluate_accuracy(model, val_loader, device):
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            state, policy_target, value_target, legal_mask = batch
            state = state.to(device)
            policy_target = policy_target.to(device)
            legal_mask = legal_mask.to(device)

            policy_logits, _ = model(state)
            policy_logits = policy_logits.masked_fill(~legal_mask, -torch.inf)

            top5_indices = policy_logits.topk(5, dim=1).indices
            top1_indices = top5_indices[:, 0]

            correct_top1 += (top1_indices == policy_target).sum().item()
            correct_top5 += (top5_indices == policy_target.unsqueeze(1)).any(dim=1).sum().item()
            total += policy_target.size(0)

    if total == 0:
        return 0.0, 0.0
    return correct_top1 / total, correct_top5 / total


def train(model, train_loader, val_loader, start_epoch=0, start_step=0, optimizer=None):
    if optimizer is None:
        optimizer = build_optimizer(model)
    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(optimizer, steps_per_epoch)

    for _ in range(start_step):
        scheduler.step()

    metrics_path = config.RUNS_DIR / "metrics.jsonl"
    step = start_step
    vram_logged = step > 0

    for epoch in range(start_epoch, config.MAX_EPOCHS):
        model.train()
        epoch_loss_sum = 0.0
        epoch_batches = 0

        for batch in train_loader:
            loss_val = training_step(model, batch, optimizer, config.DEVICE)
            scheduler.step()
            step += 1
            epoch_loss_sum += loss_val
            epoch_batches += 1

            if not vram_logged and torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated()
                if peak > 3.5 * 1024 ** 3:
                    print(f"[VRAM WARNING] peak={peak / 1024**3:.2f} GB after step 1")
                else:
                    print(f"[VRAM] peak={peak / 1024**3:.2f} GB after step 1")
                vram_logged = True

            if step % config.CHECKPOINT_STEPS == 0:
                ckpt_path = config.CHECKPOINT_DIR / f"ckpt_epoch{epoch}_step{step}.pt"
                save_checkpoint(model, optimizer, epoch, step, ckpt_path)

        train_loss = epoch_loss_sum / max(1, epoch_batches)
        val_top1, val_top5 = evaluate_accuracy(model, val_loader, config.DEVICE)

        record = {
            "epoch": epoch,
            "step": step,
            "train_loss": train_loss,
            "val_top1": val_top1,
            "val_top5": val_top5,
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        print(
            f"epoch={epoch} step={step} "
            f"train_loss={train_loss:.4f} "
            f"val_top1={val_top1:.4f} val_top5={val_top5:.4f}"
        )
