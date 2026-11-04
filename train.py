# train.py

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataloader import make_train_loader
from MEF_EFormer import MEF_EFormer
from MEF_EFormer_loss import MEF_EFormer_Loss


BATCH_SIZE = 1
ACCUMULATION_STEPS = 8
LEARNING_RATE = 5e-4
MINIMUM_LEARNING_RATE = 1e-5
SCHEDULER_T_MAX = 100
SEED = 1234
NUM_WORKERS = 4


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    set_seed()

    device = torch.device("cuda")

    train_loader = make_train_loader(
        inputs_dir=args.inputs,
        targets_dir=args.targets,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = MEF_EFormer().to(device)
    criterion = MEF_EFormer_Loss().to(device)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=SCHEDULER_T_MAX, eta_min=MINIMUM_LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print("Precision: FP16 automatic mixed precision")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Micro-batch size: {BATCH_SIZE}")
    print(f"Accumulation steps: {ACCUMULATION_STEPS}")
    print(f"Effective batch size: {BATCH_SIZE * ACCUMULATION_STEPS}")
    print(f"Epochs: {args.epochs}")
    print(f"Initial learning rate: {LEARNING_RATE}")
    print(f"Minimum learning rate: {MINIMUM_LEARNING_RATE}")
    print(f"Scheduler T_max: {SCHEDULER_T_MAX}")
    print(f"Seed: {SEED}")

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_loss = 0.0
        optimizer_steps = 0

        for batch_index, (under, over, target) in enumerate(train_loader):
            under = under.to(device, non_blocking=True)
            over = over.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            window_start = (batch_index // ACCUMULATION_STEPS) * ACCUMULATION_STEPS
            window_size = min(ACCUMULATION_STEPS, len(train_loader) - window_start)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                output = model(under, over)
                loss = criterion(output, target)
                accumulated_loss = loss / window_size

            scaler.scale(accumulated_loss).backward()
            epoch_loss += loss.detach().float().item()

            accumulation_boundary = (batch_index + 1) % ACCUMULATION_STEPS == 0
            final_batch = batch_index + 1 == len(train_loader)

            if accumulation_boundary or final_batch:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

        mean_loss = epoch_loss / len(train_loader)
        current_learning_rate = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch + 1:03d}/{args.epochs} | Loss: {mean_loss:.6f} | LR: {current_learning_rate:.6e} | Optimizer steps: {optimizer_steps}")

        scheduler.step()

    checkpoint_path = output_dir / "MEF_EFormer.pth"
    torch.save(model.state_dict(), checkpoint_path)

    print(f"Final checkpoint saved to: {checkpoint_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train MEF-EFormer.")
    parser.add_argument("--inputs", type=str, required=True, help="Training directory containing under/ and over/.")
    parser.add_argument("--targets", type=str, required=True, help="Training target directory.")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())