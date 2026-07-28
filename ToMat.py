import argparse
import os
from pathlib import Path

import numpy as np
import scipy.io
import torch
from skimage.metrics import peak_signal_noise_ratio

from model import Model


SENSOR_IDS = {"GF1": 0, "QB": 1, "WV2": 2, "WV4": 3}


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()):
        return "mps"
    return "cpu"


def checkpoint_state(path):
    checkpoint = torch.load(path, map_location="cpu")
    if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint):
        return checkpoint["model_state_dict"]
    return checkpoint


def load_input(root, name, full_resolution):
    ms = scipy.io.loadmat(root / "MS_32" / name)["ms0"]
    pan = scipy.io.loadmat(root / "PAN_128" / name)["pan0"]
    target_folder = "MS_128" if full_resolution else "GT_128"
    target_key = "usms0" if full_resolution else "gt0"
    target = scipy.io.loadmat(
        root / target_folder / name
    )[target_key]
    return ms, pan, target


def run(args):
    device = torch.device(args.device)
    model = Model(
        stages=args.stages, nc=args.nc
    )
    model.load_state_dict(
        checkpoint_state(args.checkpoint), strict=True
    )
    model.to(device).eval()

    data_root = Path(args.data_root)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    task = torch.zeros(1, 4, device=device)
    task[0, SENSOR_IDS[args.sensor]] = 1
    image_names = sorted(
        name for name in os.listdir(data_root / "PAN_128")
        if name.endswith(".mat")
    )
    if args.limit is not None:
        image_names = image_names[:args.limit]

    scores = []
    with torch.inference_mode():
        for index, name in enumerate(image_names, start=1):
            ms, pan, target = load_input(
                data_root, name, args.full_resolution
            )
            ms_tensor = (
                torch.from_numpy(np.asarray(ms, dtype=np.float32))
                .unsqueeze(0)
                .permute(0, 3, 1, 2)
                .to(device)
            )
            pan_tensor = (
                torch.from_numpy(np.asarray(pan, dtype=np.float32))
                .unsqueeze(0)
                .unsqueeze(1)
                .to(device)
            )
            output = model(ms_tensor, pan_tensor, task).cpu().numpy()
            score = peak_signal_noise_ratio(
                np.asarray(target, dtype=np.float32),
                output[0].transpose(1, 2, 0),
                data_range=1.0,
            )
            scores.append(score)
            scipy.io.savemat(
                output_root / name, {"sr": output}
            )
            print(
                f"[{index}/{len(image_names)}] "
                f"{name}: PSNR={score:.4f}"
            )

    print(f"Average PSNR: {np.mean(scores):.4f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the retained unified pansharpening model"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--sensor", choices=tuple(SENSOR_IDS), required=True
    )
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--nc", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--full-resolution", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
