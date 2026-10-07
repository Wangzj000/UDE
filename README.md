# Universal Delay Embedding (UDE)

Code for **A Time-Series Foundation Model by Universal Delay Embedding**. This release preserves the model, training and evaluation implementation from the supplied UDE source archive, with small changes to the checkpoint smoke test and packaging. It is not a complete reproduction package for every manuscript result.

**Checkpoints and datasets are not included or uploaded to this repository.**

## Architecture

UDE transforms time series into Hankel matrices (delay embeddings), patches them into tokens, and processes them through a Transformer encoder with a linear prediction head:

```
Input Series -> Hankel Matrix -> Patch Tokenization -> Token Pooling -> Transformer Encoder -> Linear Head -> Prediction
```

Key components:
- **RevIN** (Reversible Instance Normalization): Normalizes each input window and reverses that normalization after prediction
- **Hankel Transform**: Converts 1D time series into 2D delay-embedded matrices
- **Patch Tokenization**: Splits the Hankel matrix into non-overlapping patches
- **Average Pooling**: Reduces token count for computational efficiency
- **Transformer Encoder**: Standard multi-head self-attention layers
- **Linear Prediction Head**: Maps encoder output to future predictions

## Model Variants

| Model | Layers | Heads | d_model | d_ff |
|-------|--------|-------|---------|------|
| UDE-Small | 6 | 8 | 512 | 2048 |
| UDE-Medium | 10 | 12 | 512 | 2048 |
| UDE-Large | 12 | 16 | 512 | 2048 |

## Installation

```bash
# Clone the repository
git clone https://github.com/Wangzj000/UDE.git
cd UDE

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python >= 3.9
- PyTorch >= 2.6.0 (for safe checkpoint loading in the smoke test)
- A CUDA-compatible PyTorch installation for the original GPU training/evaluation scripts

The Small-model smoke test was checked on CPU with Python 3.12, PyTorch 2.6.0, NumPy 1.26.4 and einops 0.8.1. For this test alone, `torch`, `numpy`, and `einops` are sufficient. The full requirements are for the archived experiment scripts.

## Pretrained Weights

Obtain a checkpoint separately and keep it locally. No public download is supplied in this release. You can pass an external path to `quick_test.py`; there is no need to copy weights into this repository.

The author-selected Small checkpoint used for validation has SHA-256:

```text
2caeea54bb13d219c4eee81d5f775cdfd6b6c92bb0a0373c563afaf9501234cd
```

For the archived shell scripts, set `CKPT_ROOT` to your local weight directory, or use the following layout (ignored by Git):

```
UDE/
└── 12Bcheckpoints/
    └── new/
        ├── small/
        │   └── checkpoint.pth
        ├── medium/
        │   └── checkpoint.pth
        └── large/
        │   └── checkpoint.pth
```

## Dataset Preparation

Organize your datasets under a root directory (set via `DATA_ROOT` environment variable):

```
$DATA_ROOT/
├── ETT-small/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── weather/
│   └── weather.csv
├── electricity/
│   └── electricity.csv
├── traffic/
│   └── traffic.csv
├── Monash/
│   └── *.csv
└── ucr_csv/
    └── *.csv
```

The ETT, weather, electricity, and traffic datasets can be obtained from the [Time-Series-Library](https://github.com/thuml/Time-Series-Library) repository.

## Quick Start

### 1. Verify Installation

```bash
python quick_test.py --model-size small --checkpoint /path/to/small/checkpoint.pth
```

This performs a CPU forward pass on synthetic input of shape `(2, 1024, 1)` and checks a finite output of shape `(2, 96, 1)`. It checks strict loading of all active model tensors after removing only six explicitly named inactive channel-mixing-head tensors, if present. Missing or unexpected active tensors cause failure. This is a compatibility check, not a benchmark result.

Use `--model-size medium` or `large` with the corresponding checkpoint. Running without arguments retains the original behavior of testing all three local checkpoint paths; all three files are then required.

### 2. Zero-Shot Evaluation

The original scripts are retained below. They require separately prepared datasets and checkpoints; the complete benchmark and training workflows have not been rerun as part of this publication check. Historical result provenance is not established by a successful smoke test.

Evaluate a pretrained model without fine-tuning:

```bash
# Set data path
export DATA_ROOT=/path/to/your/datasets

# Run zero-shot evaluation (options: small, medium, large)
bash scripts/zero_shot_test.sh small
```

### 3. Fine-Tuning

Fine-tune a pretrained model on a downstream dataset:

```bash
export DATA_ROOT=/path/to/your/datasets

# Fine-tune small model on ETTm2 with 10% training data
bash scripts/finetune_example.sh small ETT-small/ETTm2.csv 0.1
```

### 4. Multi-GPU Training

For SLURM-based HPC clusters, use the provided scripts:

```bash
# Zero-shot evaluation with 2 GPUs
sbatch test_small.sh

# Fine-tuning on weather dataset
cd finetune && sbatch weather_small.sh
```

## Project Structure

```
UDE/
├── models/                    # Model architectures
│   ├── delayformer.py        # Base Delayformer model
│   ├── delayformer_fixpooling.py  # Delayformer with fixed pooling (main model)
│   ├── iTransformer.py       # iTransformer baseline
│   ├── PatchTST.py           # PatchTST baseline
│   └── DLinear.py            # DLinear baseline
├── layers/                    # Network layer components
│   ├── Transformer_EncDec.py # Transformer encoder/decoder layers
│   ├── SelfAttention_Family.py # Attention mechanisms
│   ├── Embed.py              # Positional encodings & embeddings
│   └── RevIN.py              # Reversible Instance Normalization
├── exp/                       # Experiment management
│   ├── exp_basic.py          # Base experiment class
│   ├── exp_large.py          # Pretraining experiment
│   ├── exp_finetune.py       # Fine-tuning experiment
│   └── exp_long_term_forecasting.py  # Long-term forecasting
├── data_provider/             # Data loading utilities
│   ├── data_factory.py       # Dataset factory
│   └── data_loader.py        # Dataset implementations
├── utils/                     # Utility functions
│   ├── tools.py              # Training tools (EarlyStopping, etc.)
│   ├── metrics.py            # Evaluation metrics (MSE, MAE)
│   └── losses.py             # Loss functions
├── scripts/                   # Clean example scripts
│   ├── zero_shot_test.sh     # Zero-shot evaluation
│   └── finetune_example.sh   # Fine-tuning example
├── finetune/                  # Dataset-specific fine-tuning scripts
├── 12Bcheckpoints/            # Pretrained model weights (download separately)
├── run.py                     # Main entry point
├── quick_test.py              # Quick validation script
├── requirements.txt           # Python dependencies
├── train_data_list.json       # Pretraining dataset list
└── vali_data_list.json        # Validation dataset list
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATA_ROOT` | Root directory for datasets | `./data` |
| `CKPT_ROOT` | Root directory for checkpoints | `./12Bcheckpoints/new` |

### Key Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--seq_len` | Input sequence length | 1024 |
| `--pred_len` | Prediction length | 96 |
| `--L` | Hankel matrix column dimension (delay window) | 500 |
| `--p1, --p2` | Patch sizes for Hankel matrix | 25, 50 |
| `--pooling_kernel` | Token pooling kernel size | 30 |
| `--pooling_type` | Pooling type (avg/max) | avg |
| `--sampling_rate` | Fine-tuning data sampling rate | 0.01-1.0 |
| `--fc_only` | Only fine-tune the linear head | Flag |

## Checkpoint compatibility

The supplied `delayformer_fixpooling` code constructs a sinusoidal buffer for `pe='fix_pe'`, but its forward path checks `fix_pos`. The archived `fix_pe` configuration therefore does not apply that buffer. This behavior is retained deliberately for compatibility with the supplied checkpoint; no positional-encoding correction or model-architecture change is included here. The forecast head is a direct linear multi-horizon readout.

## Citation

Manuscript: **A Time-Series Foundation Model by Universal Delay Embedding**.

The model and experiment source files are retained from the supplied `UDE_release.tar.gz`; this publication does not assert a new license for those files.
