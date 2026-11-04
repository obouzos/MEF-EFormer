# MEF-EFormer: Efficient Transformer for Multi-Exposure Image Fusion

Official PyTorch implementation of **MEF-EFormer**, an efficient transformer architecture for multi-exposure image fusion.

This repository includes the model implementation, inference script, and trained weights for MEF-EFormer.

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

The training code and instructions will be added in a future update.

## Citation

Citation information will be added when the paper becomes available.