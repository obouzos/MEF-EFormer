# MEF-EFormer: Efficient Transformer for Multi-Exposure Image Fusion

Official PyTorch implementation of **MEF-EFormer**, an efficient transformer architecture for multi-exposure image fusion.

This repository includes the model implementation, training and inference scripts, loss function, dataloader, and trained weights for MEF-EFormer.

> **Paper:** The paper link will be added when available.

## Installation

Clone the repository:

```bash
git clone https://github.com/obouzos/MEF-EFormer.git
cd MEF-EFormer
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Inference

The inference script expects two spatially aligned RGB images representing the underexposed and overexposed inputs. Both images must have identical spatial dimensions.

### FP32 inference

FP32 is the default inference mode:

```bash
python inference.py --under "images/under.png" --over "images/over.png" --output "output/fused.png" --precision fp32
```

The `--precision fp32` argument may be omitted:

```bash
python inference.py --under "images/under.png" --over "images/over.png" --output "output/fused.png"
```

### FP16 inference

Native FP16 inference requires a CUDA-capable GPU:

```bash
python inference.py --under "images/under.png" --over "images/over.png" --output "output/fused_fp16.png" --precision fp16
```

In native FP16 mode, the model parameters, input tensors, and output tensor use `torch.float16`.

### Custom checkpoint

Use the `--weights` argument to load a custom checkpoint:

```bash
python inference.py --under "images/under.png" --over "images/over.png" --output "output/fused.png" --weights "checkpoints/MEF_EFormer.pth" --precision fp32
```

Display all available options:

```bash
python inference.py --help
```

The script automatically uses CUDA when available and otherwise runs on the CPU. Native FP16 inference is available only on CUDA.

## Training

Training uses FP16 automatic mixed precision on a CUDA-capable GPU. Model parameters and saved checkpoints remain in FP32.

The expected training dataset structure is:

```text
train/
├── under/
│   ├── input_1.png
│   └── ...
├── over/
│   ├── input_1.png
│   └── ...
└── targets/
    ├── input_1.png
    └── ...
```

The images in `under`, `over`, and `targets` must use matching filenames.

Run training with:

```bash
python train.py --inputs "path/to/train" --targets "path/to/train/targets"
```

By default, training uses:

```text
Epochs:                 100
Micro-batch size:       1
Gradient accumulation:  8 steps
Effective batch size:   8
Optimizer:              Adam
Adam betas:             (0.9, 0.999)
Initial learning rate:  5e-4
Minimum learning rate:  1e-5
Scheduler:              CosineAnnealingLR
Scheduler T_max:        100
Seed:                   1234
Precision:              FP16 automatic mixed precision
```

The final checkpoint is saved to:

```text
checkpoints/MEF_EFormer.pth
```

No validation, early stopping, or validation-based checkpoint selection is performed during training. The final-epoch checkpoint is used for post-training evaluation.

To change the number of epochs or checkpoint directory:

```bash
python train.py --inputs "path/to/train" --targets "path/to/train/targets" --epochs 100 --output-dir "checkpoints"
```

Display all available training options:

```bash
python train.py --help
```

## Citation

Citation information will be added when the paper becomes available.