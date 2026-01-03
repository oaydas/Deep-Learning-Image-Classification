# Deep Learning for Image Classification

This repository implements a full deep learning pipeline for **dog breed image classification**, with a primary focus on distinguishing **Golden Retrievers** vs **Collies**. It includes data preprocessing, CNN training, transfer learning via supervised pretraining on additional breeds, a Vision Transformer (ViT) implementation, and a configurable challenge model for generating prediction files.

## Overview

The project explores multiple approaches to image classification and representation learning:

- **Data preprocessing** (per-channel normalization)
- **Convolutional Neural Networks (CNNs)** for binary classification
- **Transfer learning** via supervised pretraining on related breeds
- **Vision Transformers (ViT)** with custom attention + encoder implementation
- **Custom challenge model** with flexible architecture/training choices

## Dataset

- **12,775** PNG images across **10** dog breeds
- Image shape: **3 × 64 × 64** (RGB)
- Directory layout:
  - `data/images/` — image files
  - `data/dogs.csv` — labels + split metadata (train/val/test/challenge); challenge labels are withheld
- Data partitions:
  - Training set
  - Validation set
  - Test set
  - Held-out challenge set (for generating predictions)

## Repository Structure

Typical layout (may vary slightly depending on edits):

- `dataset.py`  
  Dataset loading and preprocessing utilities (including `ImageStandardizer`)
- `models/`
  - `target.py` — CNN for the **binary target task** (Golden Retriever vs Collie)
  - `source.py` — CNN for the **source task** (multi-class breed classification for pretraining)
  - `vit.py` — Vision Transformer implementation (attention + encoder + ViT forward pass)
  - `challenge.py` — custom model for the final challenge
- Training scripts:
  - `train_cnn.py` — train the target CNN from scratch with early stopping
  - `test_cnn.py` — evaluate target CNN on train/val/test
  - `train_source.py` — train source CNN for supervised pretraining
  - `train_target.py` — transfer learning experiments (freeze/unfreeze layers)
  - `train_vit.py` / `test_vit.py` — train/evaluate ViT on the target task
  - `train_challenge.py` — train custom challenge model
  - `predict_challenge.py` — generate challenge predictions CSV
- `utils.py` / `train_common.py`  
  Shared training utilities (metrics, checkpointing, early stopping, prediction helper)
- `checkpoints/`  
  Saved model weights (often excluded from commits)

## Setup

This project uses a Conda environment.

### Create and activate environment
```bash
conda create --name dl-image-classification --file requirements.txt
conda activate dl-image-classification
```

## Notes

- The default setup is CPU-friendly.
- If you have a compatible GPU, you can install a CUDA-enabled PyTorch build manually.

## Data Preprocessing

Images are standardized **per channel** (R, G, B) using statistics computed from the **training split only**:

1. Compute per-channel mean and standard deviation on training images
2. Apply to train/val/test/challenge using:
```
x = (x - mean) / std
```

per channel

Relevant code:
- `ImageStandardizer.fit()` and `ImageStandardizer.transform()` in `dataset.py`

## Models

### 1) Target CNN (Binary Classification)

**Goal:** classify **Golden Retriever vs Collie**

Common configuration:
- Loss: `torch.nn.CrossEntropyLoss`
- Optimizer: `torch.optim.Adam`
- Learning rate: `1e-3`
- Batch size: `32`
- Early stopping patience: `5`

Train:
```bash
python train_cnn.py
```
Test:
```bash
python test_cnn.py
```

### 2) Source CNN (Supervised Pretraining)

**Goal:** learn transferable features by training on additional breeds (multi-class classification), then reuse weights for the target task.

Common configuration:
- Loss: `torch.nn.CrossEntropyLoss`
- Optimizer: `torch.optim.Adam` (often with weight decay)
- Learning rate: `1e-3`
- Batch size: `64`
- Early stopping patience: `10`

Train the source model:
```bash
python train_source.py
```

Run transfer learning experiments (freezing/fine-tuning):
```bash
python train_target.py
```

Transfer experiments typically include:
- Freeze all convolutional layers (train only the fully connected layer)
- Freeze early convolutional layers (fine-tune later convolutional layers)
- Freeze none (fine-tune all layers)

### 3) Vision Transformer (ViT)

**Goal:** train a transformer-based image classifier for the same binary task.

Key implemented components:
- Scaled dot-product attention
- Multi-head attention forward pass
- Transformer encoder blocks
- ViT forward pass including patch projection and `[cls]` token usage

Train:
```bash
python train_vit.py
```

Test:
```bash
python test_vit.py
```

### 4) Challenge Model

**Goal:** design and train a custom architecture and training strategy, then generate predictions for a held-out challenge split.

Train:
```bash
python train_challenge.py
```

Generate Predictions:
```bash
python predict_challenge.py --uniqname=<your_uniqname>
```

This produces:
- <your_uniqname>.csv containing float-valued predictions for the challenge images

If an output format checker exists:
```bash
python test_output.py <your_uniqname>.csv
```

## Checkpointing and Early Stopping

Training scripts save model checkpoints periodically and use early stopping based on validation loss. This helps:
- prevent wasted training after convergence
- resume training after interruptions
- evaluate the best checkpoint rather than the final epoch

## Metrics

The project tracks common classification metrics:
- Accuracy
- AUROC (Area Under the ROC Curve)

Training scripts typically generate plots per epoch for:
- Loss
- Accuracy
- AUROC

## License / Notes

If you are publishing this publicly, ensure you have permission to distribute any included data. A safe default is to keep `data/` and `checkpoints/` out of version control via `.gitignore`.
