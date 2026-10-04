# MEF-EFormer: Efficient Transformer for Multi-Exposure Image Fusion

Official PyTorch implementation of **MEF-EFormer**, an efficient transformer architecture for multi-exposure image fusion.

This repository includes the code and trained weights for the MEF-EFormer method.

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

The inference script expects two well-aligned multi-exposure RGB images with identical dimensions:

- an underexposed image
- an overexposed image

Run the included example:

```bash
python inference.py --under "images/under.png" --over "images/over.png" --output "output/fused.png"
```

The script automatically uses CUDA when available and otherwise runs on the CPU.

## Citation

Citation information will be added when the paper becomes available.
