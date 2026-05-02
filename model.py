import torch
import torch.nn as nn
import config


class ResBlock(nn.Module):
    def __init__(self, filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)
        nn.init.zeros_(self.bn2.weight)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class FerrumNet(nn.Module):
    def __init__(self, filters: int, num_blocks: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(18, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResBlock(filters) for _ in range(num_blocks)])

        # Policy head: spatial 1x1 -> flatten -> linear -> (B, 4672)
        self.policy_conv = nn.Conv2d(filters, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_relu = nn.ReLU(inplace=True)
        self.policy_fc = nn.Linear(32 * 64, 73 * 64)

        # Value head (dormant in Phase 1)
        self.value_conv = nn.Conv2d(filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_relu = nn.ReLU(inplace=True)
        self.value_fc1 = nn.Linear(64, 256)
        self.value_relu2 = nn.ReLU(inplace=True)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.trunk(self.stem(x))

        p = self.policy_relu(self.policy_bn(self.policy_conv(shared)))
        p = self.policy_fc(p.flatten(1))          # (B, 4672)
        p = p.view(p.size(0), 73, 8, 8).flatten(1)  # reshape then flatten = (B, 4672)

        v = self.value_relu(self.value_bn(self.value_conv(shared)))
        v = self.value_relu2(self.value_fc1(v.flatten(1)))
        v = torch.tanh(self.value_fc2(v))          # (B, 1)

        return p, v


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, step: int, path) -> None:
    torch.save({
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config_snapshot": {"FILTERS": config.FILTERS, "NUM_BLOCKS": config.NUM_BLOCKS},
    }, path)


def load_checkpoint(path, model: nn.Module,
                    optimizer: torch.optim.Optimizer | None = None) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    snap = ckpt.get("config_snapshot", {})
    if snap.get("FILTERS") != config.FILTERS or snap.get("NUM_BLOCKS") != config.NUM_BLOCKS:
        raise ValueError(
            f"Checkpoint architecture {snap} does not match current config "
            f"(FILTERS={config.FILTERS}, NUM_BLOCKS={config.NUM_BLOCKS})"
        )
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return {"epoch": ckpt["epoch"], "step": ckpt["step"],
            "config_snapshot": ckpt.get("config_snapshot")}
