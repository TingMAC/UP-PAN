# Deep Unrolling for Unified Pansharpening with Satellite-specific and Image-adaptive Priors

## Dataset Organization

This project uses four satellite datasets: GF1, QuickBird, WorldView-2, and
WorldView-4. Organize the dataset root as follows:

```text
DATA_ROOT/
├── GF1/
│   ├── train/
│   │   ├── MS_32/
│   │   ├── PAN_128/
│   │   └── GT_128/
│   └── test/
│       ├── MS_32/
│       ├── PAN_128/
│       └── GT_128/
├── QB/
│   ├── train/{MS_32,PAN_128,GT_128}/
│   └── test/{MS_32,PAN_128,GT_128}/
├── WV2/
│   ├── train/{MS_32,PAN_128,GT_128}/
│   └── test/{MS_32,PAN_128,GT_128}/
└── WV4/
    ├── train/{MS_32,PAN_128,GT_128}/
    └── test/{MS_32,PAN_128,GT_128}/
```

Each sample must use the same `.mat` filename across `MS_32`, `PAN_128`, and
`GT_128`. The expected variables are:

| Directory | Variable | Typical shape |
|---|---|---|
| `MS_32` | `ms0` | `32 × 32 × C` |
| `PAN_128` | `pan0` | `128 × 128` |
| `GT_128` | `gt0` | `128 × 128 × C` |

Here, `C` is either 4 or 8.

## Training

Training consists of two stages.

### Stage 1: Train the Codebook

Use `train_codebook.py` to train the 3D codebook:

```bash
python train_codebook.py \
  --data-root /path/to/DATA_ROOT \
  --save-dir experiment \
  --device cuda
```

### Stage 2: Train the Deep Unrolling Network

Use `train_pansharpeningNotext.py` and provide the Stage 1 checkpoint through
`--codebook-checkpoint`:

```bash
python train_pansharpeningNotext.py \
  --data-root /path/to/DATA_ROOT \
  --codebook-checkpoint /path/to/codebook_checkpoint.pth \
  --save-dir experiment \
  --device cuda
```
