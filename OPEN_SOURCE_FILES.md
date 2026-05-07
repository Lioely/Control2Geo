# Minimal Open-Source File Set

This repository split is intended for model inference and benchmark testing only. Training scripts, training datasets, and training-only callbacks should stay private.

## Entry Points

- `scripts/usage_infer.py`: minimal single-image inference example for GitHub users.
- `scripts/test_model.py`: benchmark/evaluation entry point for configured test datasets.

## Model Configuration And Weights

- `config/model/test_bridge_control.yaml`: model architecture and checkpoint paths for inference.
- `config/data/test_data.yaml`: benchmark dataset list and preprocessing settings.
- `model.pth`: released model weights referenced by the model config.
- `checkpoint/empty_prompt768.npy`: empty text prompt embedding used by the released model.

If the released checkpoint is renamed, update `ckpt_path` in `config/model/test_bridge_control.yaml`.

## Model Code

- `model/utils.py`
- `model/diffusion_bridge/cldbm_point_latent_control.py`
- `model/diffusion_unet/control_net_latent_control.py`
- `model/diffusion_unet/openaimodel.py`
- `model/diffusion_unet/latentcoder/autoencoder.py`
- `model/diffusion_unet/latentcoder/autoencoder_model.py`
- `model/diffusion_unet/latentcoder/attention.py`

These files are the minimal model construction path used by `config/model/test_bridge_control.yaml`.

## Inference And Evaluation Utilities

- `utils/infer_utils.py`
- `utils/vis.py`
- `utils/geometry_torch.py`
- `utils/geometry_numpy.py`
- `utils/geometry_utils.py`
- `utils/metric.py`
- `utils/alignment.py`
- `utils/tools.py`

## Test Data Loading

- `data/dataloader/eval_dataloader.py`
- `data/dataset/eval/eval_dataset.py`
- `data/dataset/utils.py`
- `data/split/*.csv` for the datasets you want to support publicly, or a small sample CSV documenting the expected columns:
  - `image`
  - `depth`
  - `meta`
  - `dataset`
  - `control_geo`

Each `meta` JSON file must contain normalized camera intrinsics under the `intrinsics` key.

## Dependencies

Keep an inference/test oriented `requirements.txt`. The current required groups are:

- PyTorch stack: `torch`, `torchvision`, `xformers`
- Model/config: `omegaconf`, `pytorch_lightning`, `taming-transformers`, `einops`
- Data and visualization: `pillow`, `pandas`, `numpy`, `scipy`, `matplotlib`, `opencv-python`, `trimesh`
- Geometry: `utils3d`

## Keep Private

- `scripts/train_point_latent_control.py`
- `data/dataloader/train_dataloader.py`
- `data/dataloader/val_dataloader.py`
- `data/dataset/train/**`
- `config/model/train_*.yaml`
- `config/data/train_data.yaml`
- `config/data/val_data.yaml`
- `model/diffusion_bridge/image_logger*.py`
- `model/diffusion_bridge/metric_logger.py`
- notebooks, `.ipynb_checkpoints`, experiment outputs, and unused model variants.
