# Vision Transformer (ViT) Implementation from Scratch

This repository contains a PyTorch implementation of the Vision Transformer (ViT) model as described in the paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929) by Dosovitskiy et al.

## Overview

Vision Transformers (ViT) apply the Transformer architecture, originally designed for natural language processing, to image classification tasks. The key idea is to split an image into fixed-size patches, linearly embed each patch, add position embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder.

## Architecture

The implementation includes the following components:

1. **Patch Embedding**: Splits the image into patches and projects them to the embedding dimension.
2. **Transformer Encoder**: A stack of Transformer blocks, each containing:
   - Multi-head Self-Attention (MSA)
   - Multi-Layer Perceptron (MLP)
   - Layer Normalization (LN)
   - Residual connections
3. **Classification Head**: A linear layer that maps the [CLS] token representation to class logits.

## Files

- `vision_transformer.py`: The main implementation of the Vision Transformer model.
- `example_usage.py`: Example script demonstrating how to use the model and visualize patches.

## Usage

```python
from vision_transformer import VisionTransformer

# Create a ViT model
model = VisionTransformer(
    img_size=224,        # Input image size
    patch_size=16,       # Patch size
    in_channels=3,       # Number of input channels (RGB)
    num_classes=1000,    # Number of classes
    embed_dim=768,       # Embedding dimension
    depth=12,            # Number of transformer blocks
    n_heads=12,          # Number of attention heads
    mlp_ratio=4.0,       # MLP hidden dim ratio
    qkv_bias=True        # Use bias for query, key, value projections
)

# Forward pass
import torch
x = torch.randn(1, 3, 224, 224)  # Batch of 1 image
logits = model(x)  # Shape: [1, num_classes]