# Anomaly Quai

This repository contains a Python training script and a YAML configuration file for running anomaly detection experiments with [Anomalib](https://github.com/open-edge-platform/anomalib). The setup is designed for both training and testing, with optional Weights & Biases logging and CUDA-capable PyTorch/Anomalib installation.

## Repository contents

| File | Purpose |
| --- | --- |
| `anomaly_quai.py` | Main entry point for loading the config, preparing the dataset, creating the model, and running training or evaluation. |
| `configs/anomaly_quai.yaml` | Configuration file for paths, logging, callbacks, metrics, model choice, and engine settings. |

## Requirements

Use a virtual environment and install Anomalib with the optional dependencies you need. The Anomalib documentation supports hardware-specific installs such as CUDA extras, and the `full` extra installs all optional dependencies. For a GPU setup, install the CUDA build that matches your system, for example `cu121` or `cu130`, together with `full` if you want every optional feature. [web:4]

```bash
# Example: full optional dependencies + CUDA support
pip install "anomalib[full,cu130]"
```
For more info see their [github page](https://github.com/open-edge-platform/anomalib)

## Setup

1. Create and activate a Python environment.
2. Install Anomalib with the CUDA and optional dependency extras needed for your machine.
3. Make sure to create a `.env` file to store your `WANDB_API_KEY` if u want to use [Weights & Biases](https://wandb.ai/site/) logging.
4. Create a "datasets" folder in the root directory to put your datasets into.

## How to run

### Training

```bash
python anomaly_quai.py --mode train --config anomaly_quai.yaml --dataset MVTecAD --logger
```

### Testing from a checkpoint(model automatically derived from the checkpoint name)

```bash
python anomaly_quai.py --mode test --config anomaly_quai.yaml --checkpoint /path/to/model.ckpt
```

### Custom dataset

Use `--dataset` with a name that is not in the built-in Anomalib datamodule list to trigger folder-based loading. In that case, the script uses `anomalib.data.Folder` and expects the directory structure described in the YAML file.

## Python script explained

The script is a command-line runner that wires together configuration, data loading, logging, callbacks, metrics, and model execution.

### 1. Imports and environment loading

The script imports `anomalib`, `lightning.pytorch.callbacks`, `OmegaConf`, and helper utilities such as `Path` and `datetime`. It also loads environment variables from `.env` so the W&B API key can be read at runtime.

### 2. Dataset path validation

The helper function `setup_directory()` builds the dataset path from the config root and the chosen dataset name. It prints the resolved path and raises an error if the directory does not exist, which helps catch path problems early.

### 3. Command-line arguments

The script accepts `--config`, `--dataset`, `--logger`, `--mode`, and `--checkpoint`. The `--mode` flag controls whether the script trains a model or only evaluates an existing checkpoint, and `--checkpoint` becomes mandatory in test mode.

### 4. Configuration loading

The YAML file is loaded with `OmegaConf.load()`. The script then reads the root path, datamodule parameters, W&B settings, callbacks, metrics, model name, and engine options from that config.

### 5. Data module selection

If the dataset name matches one of the built-in Anomalib datamodules, the script creates that datamodule dynamically with `getattr(anomalib.data, args.dataset)`. Otherwise, it falls back to `anomalib.data.Folder`, which is intended for custom folder-structured datasets.

### 6. Logging

When `--logger` is enabled, the script logs into Weights & Biases and creates an `AnomalibWandbLogger` using the `wandb` section of the YAML file. If the flag is omitted, the logger is disabled.

### 7. Callbacks

Callbacks are enabled through `config.callbacks.enabled`. `ModelCheckpoint` is handled specially so the saved checkpoint filename includes the model name and a timestamp, while other callbacks are created from `lightning.pytorch.callbacks`.

### 8. Metrics and evaluator

The script builds separate validation and test metric lists from the `metrics` section of the YAML file. It then creates an `anomalib.metrics.Evaluator` instance, which is passed into the model so the configured metrics are used during validation and testing.

### 9. Model creation

In training mode, the model is created from `config.model.name`. In testing mode, the model name is inferred from the checkpoint filename, which means the checkpoint naming convention must match the model name prefix.

### 10. Engine execution

The script creates an `Engine` with the configured callbacks, logger, and engine parameters. It then runs `fit()` and `test()` in training mode, or `test()` with `ckpt_path` in test mode.

## Config file explained

The YAML file centralizes experiment settings so the script stays small and flexible.

### Root path

`root` points to the repository location used to build dataset, log, and checkpoint paths. The script uses this to resolve the data directory under `root/datasets/<dataset>`.

### Data section

The `data.prebuilt` block is for built-in Anomalib datamodules, such as a dataset category like `bottle`. The `data.custom` block is for folder-based datasets and defines the directory names for normal images, anomalous images, masks, and batch sizes.

### W&B section

The `wandb` block defines the project name, run name, entity, save directory, notes, group, and job type. These values are used only when logging is enabled from the command line.

### Callbacks section

`ModelCheckpoint` saves the best model according to `val_pixel_AUROC`, and `EarlyStopping` stops training when that metric stops improving. This setup emphasizes pixel-level localization quality rather than only image-level detection.

### Metrics section

Validation uses image-level AUROC and pixel-level AUROC. Testing is broader and includes AUROC, F1Score, AUPR, AUPRO, and AUPIMO for both image-level and pixel-level outputs.

### Model section

The model is set to `AnomalyDINO`. Because the script reads the model name dynamically, changing this field lets the same code run different Anomalib models without editing the script.

### Engine section

The engine is configured with `accelerator: auto`, so Anomalib will choose the best available device automatically. The `log_every_n_steps` value controls how often progress is logged during training.

## Notes

- The repo expects the dataset to exist before execution; it does not download data automatically.
- In test mode, the checkpoint path must be provided.
- If W&B logging is disabled, the script still runs normally.
- The checkpoint filename should begin with the model name if you rely on automatic model detection in test mode.

## Example folder layout

```text
repo/
├── anomaly_quai.py
├── anomaly_quai.yaml
├── datasets/
│   └── MVTecAD/
│       └── bottle/
├── checkpoints/
└── logs/
```
## Examples
See the examples folder for notebooks explaining anomalib.
