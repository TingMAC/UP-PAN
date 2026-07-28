import argparse
import logging
import os
from datetime import datetime

import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from Dataset import CombineMatDataset, MatDataset
from model import Model


DATASETS = (
    ("GF", "GF1", 0),
    ("QB", "QB", 1),
    ("WV2", "WV2", 2),
    ("WV4", "WV4", 3),
)


def build_logger(path):
    logger = logging.getLogger("train_pansharpening")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(path, mode="w")
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s"
    ))
    logger.addHandler(handler)
    return logger


def one_hot(label, device, classes=4):
    result = torch.zeros(1, classes, device=device)
    result[0, label] = 1
    return result


def load_datasets(root, split):
    return [
        MatDataset(os.path.join(root, folder, split))
        for _, folder, _ in DATASETS
    ]


def validate(model, datasets, device):
    model.eval()
    scores = []
    with torch.inference_mode():
        for dataset, (_, _, label) in zip(datasets, DATASETS):
            total = 0.0
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            task = one_hot(label, device)
            for ms, pan, target in tqdm(loader, leave=False):
                ms = ms.float().to(device).permute(0, 3, 1, 2)
                pan = pan.float().to(device).unsqueeze(1)
                target = target.float().permute(0, 3, 1, 2)
                output = model(ms, pan, task).cpu()
                total += peak_signal_noise_ratio(
                    target.numpy()[0],
                    output.numpy()[0],
                    data_range=1.0,
                )
            scores.append(total / len(dataset))
    return scores


def train(args):
    device = torch.device(args.device)
    run_name = datetime.now().strftime("%m-%d_%H-%M_") + args.exp_name
    save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    logger = build_logger(os.path.join(save_dir, "train.log"))

    training_sets = load_datasets(args.data_root, "train")
    training_dataset = CombineMatDataset(
        training_sets, [item[2] for item in DATASETS]
    )
    validation_sets = load_datasets(args.data_root, "test")

    model = Model(
        Ch=8,
        stages=args.stages,
        nc=args.nc,
        codebook_checkpoint_path=args.codebook_checkpoint,
    ).to(device)
    optimizer = torch.optim.Adam(
        (
            parameter for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = CosineAnnealingLR(
        optimizer, args.epochs, eta_min=1e-6
    )
    criterion = nn.L1Loss()
    logger.info("args=%s", args)
    logger.info(
        "parameters=%.6fM",
        sum(parameter.numel() for parameter in model.parameters()) / 1e6,
    )

    best_scores = [-float("inf")] * len(DATASETS)
    scores = [float("nan")] * len(DATASETS)
    for epoch in range(args.epochs):
        training_dataset.shuffle()
        loader = DataLoader(
            training_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
        )
        model.train()
        progress = tqdm(loader, desc=f"unfolding epoch {epoch}")
        for grouped_samples in progress:
            for sample, _ in zip(grouped_samples, DATASETS):
                (ms, pan, target), task = sample
                ms = ms.float().to(device).permute(0, 3, 1, 2)
                pan = pan.float().to(device).unsqueeze(1)
                target = target.float().to(device).permute(0, 3, 1, 2)
                task = task.to(device)
                optimizer.zero_grad()
                output = model(ms, pan, task)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                scheduler.step()
            progress.set_postfix(loss=f"{loss.item():.6f}")

        if epoch % args.validation_interval == 0:
            scores = validate(model, validation_sets, device)
            logger.info(
                "epoch=%d GF=%.4f QB=%.4f WV2=%.4f WV4=%.4f",
                epoch, *scores
            )
            for index, score in enumerate(scores):
                if score > best_scores[index]:
                    best_scores[index] = score
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            save_dir, f"best_{DATASETS[index][0]}.pth"
                        ),
                    )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metric_value": scores,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            os.path.join(save_dir, f"epoch={epoch}.pth"),
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2: train the three-stage unfolding model"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--codebook-checkpoint", required=True)
    parser.add_argument("--save-dir", default="experiment")
    parser.add_argument("--exp-name", default="Dim24-Stage3-Wavelength")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--nc", type=int, default=32)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
