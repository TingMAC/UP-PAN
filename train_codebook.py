import argparse
import logging
import os
from datetime import datetime

import torch
from skimage.metrics import peak_signal_noise_ratio
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from Dataset import CombineMatDataset, MatDataset
from model.codebook.loss import CharbonnierLoss
from model.codebook.network3d import Network3D


DATASETS = (
    ("GF", "GF1", 0),
    ("QB", "QB", 1),
    ("WV2", "WV2", 2),
    ("WV4", "WV4", 3),
)


def build_logger(path):
    logger = logging.getLogger("train_codebook")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(path, mode="w")
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s"
    ))
    logger.addHandler(handler)
    return logger


def one_hot(label, classes=4):
    result = torch.zeros(classes)
    result[label] = 1
    return result.unsqueeze(0)


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
            task = one_hot(label).to(device)
            for _, _, target in tqdm(loader, leave=False):
                target = target.float().to(device).permute(0, 3, 1, 2)
                output = model(target, task)[0]
                total += peak_signal_noise_ratio(
                    target.cpu().numpy()[0],
                    output.cpu().numpy()[0],
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

    train_sets = load_datasets(args.data_root, "train")
    train_dataset = CombineMatDataset(
        train_sets, [item[2] for item in DATASETS]
    )
    validation_sets = load_datasets(args.data_root, "test")

    model = Network3D().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate,
        betas=(0.9, 0.999), eps=1e-8
    )
    scheduler = CosineAnnealingLR(
        optimizer, args.total_iterations, eta_min=1e-6
    )
    criterion = CharbonnierLoss(reduction="mean")
    logger.info("args=%s", args)
    logger.info(
        "parameters=%.6fM",
        sum(parameter.numel() for parameter in model.parameters()) / 1e6,
    )

    for epoch in range(args.epochs):
        train_dataset.shuffle()
        loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
        )
        model.train()
        progress = tqdm(loader, desc=f"codebook epoch {epoch}")
        for grouped_samples in progress:
            for sample, _ in zip(grouped_samples, DATASETS):
                (_, _, target), task = sample
                target = target.float().to(device).permute(0, 3, 1, 2)
                task = task.to(device)
                optimizer.zero_grad()
                output, codebook_loss, _, _ = model(target, task)
                loss = (
                    criterion(output, target)
                    + codebook_loss * target.shape[1]
                )
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
        torch.save(
            model.state_dict(),
            os.path.join(save_dir, f"epoch={epoch}.pth"),
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1: train the shared/private 3D codebook"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--save-dir", default="experiment")
    parser.add_argument("--exp-name", default="codebook")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--total-iterations", type=int, default=30000)
    parser.add_argument("--validation-interval", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
