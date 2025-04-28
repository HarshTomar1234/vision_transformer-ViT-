"""
Vision Transformers vs Convolutional Neural Networks
===================================================

Key Architectural Differences:
-----------------------------

1. Feature Extraction:
   - CNNs: Use sliding convolutional filters that extract local features
   - ViTs: Split images into non-overlapping patches (as seen in PatchEmbedding)

2. Information Processing:
   - CNNs: Process information hierarchically with increasing receptive fields
   - ViTs: Process all patches simultaneously with global self-attention

3. Inductive Bias:
   - CNNs: Strong spatial inductive bias (locality, translation invariance)
   - ViTs: Weaker inductive bias, more data-driven approach

4. Positional Information:
   - CNNs: Implicitly encoded through convolutional operations
   - ViTs: Require explicit positional embeddings

Detailed Theory:
--------------

1. Receptive Field Dynamics:
   - CNNs: Build receptive fields gradually through layers. Early layers capture edges,
     textures, while deeper layers capture more complex patterns. The receptive field
     grows linearly with depth and is constrained by kernel size.
   - ViTs: Have a global receptive field from the first layer due to self-attention.
     Each patch can attend to every other patch immediately, allowing for long-range
     dependencies to be captured earlier in the network.

2. Parameter Efficiency:
   - CNNs: Share weights across spatial dimensions (translation equivariance),
     making them parameter-efficient for certain tasks.
   - ViTs: Self-attention has quadratic complexity with respect to sequence length,
     making standard ViTs computationally expensive for high-resolution images.

3. Data Requirements:
   - CNNs: The inductive biases help them learn effectively from smaller datasets.
   - ViTs: Typically require more training data to overcome their lack of inductive
     bias, but can achieve higher performance ceilings with sufficient data.

4. Attention Mechanisms:
   - CNNs: Attention must be explicitly added (e.g., Squeeze-and-Excitation, CBAM).
   - ViTs: Attention is the core operation, allowing for dynamic, content-dependent
     processing of visual information.

5. Scaling Properties:
   - CNNs: Performance typically saturates with increased depth due to optimization challenges.
   - ViTs: Scale more effectively with model size and training data, following power laws
     similar to language models.

Use Cases and Comparative Advantages:
-----------------------------------

CNNs Excel At:
- Smaller datasets where inductive biases help generalization
- Tasks requiring fine-grained local feature detection
- Resource-constrained environments (generally more efficient)
- Medical imaging, object detection, segmentation

Vision Transformers Excel At:
- Large-scale datasets (need more data to overcome lack of inductive bias)
- Tasks requiring global context understanding for ex in image captioning task we require global context to understand the image
- Transfer learning from large pre-trained models
- Image classification at scale, visual question answering

Specific Task Examples:
---------------------

1. Object Detection:
   - CNN Advantage: Models like Faster R-CNN, YOLO, and SSD leverage the spatial
     hierarchy of CNNs to detect objects at multiple scales efficiently.
   - ViT Application: DETR (DEtection TRansformer) uses transformers for end-to-end
     object detection, eliminating the need for hand-designed components like
     non-maximum suppression, but requires longer training.
   - Why: CNNs are often preferred for real-time applications due to efficiency,
     while transformer-based approaches can achieve higher accuracy with sufficient
     computational resources.

2. Image Classification:
   - CNN Examples: ResNet, EfficientNet, RegNet
   - ViT Examples: ViT, DeiT, CaiT
   - Comparison: On ImageNet, ViTs outperform CNNs at scale but require more data
     and compute. For smaller datasets, CNNs often perform better unless ViTs are
     pre-trained on large datasets.

3. Medical Image Analysis:
   - CNN Advantage: U-Net and its variants excel at segmentation tasks in medical
     imaging where precise localization is critical.
   - ViT Application: TransUNet combines transformers with U-Net for improved
     performance in organ segmentation.
   - Why: The local inductive bias of CNNs helps with precise boundary detection,
     while transformers can capture global context for improved consistency.

4. Face Recognition:
   - CNN Advantage: Models like FaceNet use CNNs to extract discriminative facial
     features efficiently.
   - ViT Application: Vision transformers can capture subtle relationships between
     facial features for improved recognition.
   - Why: CNNs are more computationally efficient for deployment, while transformers
     can achieve higher accuracy by modeling long-range dependencies.

5. Video Understanding:
   - CNN Approach: 3D CNNs like I3D, SlowFast networks
   - Transformer Approach: TimeSformer, ViViT
   - Why: Transformers excel at capturing temporal dependencies across frames,
     while 3D CNNs are more parameter-efficient but limited in temporal context.

6. Low-Resource Scenarios:
   - CNN Advantage: MobileNet, EfficientNet are designed for mobile/edge devices
   - ViT Challenge: Standard ViTs are computationally expensive
   - Solution: Mobile-ViT, Efficient-ViT combine CNN-like local processing with
     limited self-attention for efficiency.

7. High-Resolution Image Processing:
   - CNN Approach: Dilated/atrous convolutions increase receptive field without
     increasing parameters
   - Transformer Approach: Swin Transformer uses shifted windows to process
     high-resolution images efficiently
   - Why: The quadratic complexity of self-attention makes standard ViTs impractical
     for very high-resolution images.

8. Few-Shot Learning:
   - CNN Traditional Approach: Siamese networks, prototypical networks
   - Transformer Advantage: ViTs pre-trained on large datasets show stronger
     few-shot generalization capabilities
   - Why: The data-hungry nature of transformers becomes an advantage when
     leveraging transfer learning from massive pre-training.

Hybrid Approaches:
----------------
- Many modern architectures combine elements of both paradigms
- Examples:
  * Swin Transformer: Hierarchical structure with local attention windows
  * ConvNeXt: CNN architecture with transformer-inspired design choices
  * ViT with convolutional stem: Using convolutions for initial feature extraction
  * MobileViT: Combines mobile convolutions with transformers for efficiency
  * ConViT: Adds soft convolutional inductive biases to vision transformers

Future Directions:
----------------
- Efficient attention mechanisms to reduce the quadratic complexity
- Incorporating stronger inductive biases into transformers
- Specialized architectures for specific domains
- Multimodal models that leverage the strengths of both approaches

Note: The PatchEmbedding class actually uses a convolutional layer with 
non-overlapping patches (stride = patch_size) to implement the initial 
tokenization, showing how these approaches can complement each other.
"""


"""
CLS Token vs Pooling Methods in Vision Transformers
==================================================

1. CLS Token Method
------------------

What is the CLS Token?
- A learnable token (vector) prepended to the sequence of patch embeddings
- Similar to BERT's [CLS] token in NLP
- Typically initialized randomly and learned during training
- Serves as an aggregation point for information from all image patches

How it Works:
- A learnable embedding (typically of size [1, 1, embed_dim]) is created
- This token is prepended to the sequence of patch embeddings
- Position embeddings are added to both the CLS token and patch embeddings
- Through self-attention layers, the CLS token attends to all patch embeddings
- The final representation of the CLS token is used for classification

Implementation Details:
- The CLS token is typically defined as: self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
- It's expanded to match the batch size: cls_tokens = self.cls_token.expand(batch_size, -1, -1)
- Then concatenated with patch embeddings: x = torch.cat((cls_tokens, x), dim=1)
- After processing through transformer blocks, only the CLS token representation is used: x = x[:, 0]

Importance:
- Provides a natural way to aggregate global information across all patches
- Creates an explicit "summary" token that can focus on classification-relevant features
- Allows the model to learn which patches are most important for the task
- Enables the model to capture complex relationships between different regions of the image

2. Pooling Method
----------------

What is the Pooling Method?
- Instead of using a special token, this approach applies pooling operations over all patch embeddings
- Common variants include mean pooling, max pooling, or weighted pooling
- The pooled representation is then used for classification

How it Works:
- Process all patch embeddings through the transformer blocks
- Apply a pooling operation across the sequence dimension
- Use the resulting vector for classification

Implementation Variants:
- Mean Pooling: x = x.mean(dim=1)  # Average across all patches
- Max Pooling: x = x.max(dim=1)[0]  # Take maximum value across patches
- Weighted Pooling: Apply learned weights to each patch before summing
- Attention Pooling: Use an attention mechanism to compute weighted average

Comparison: CLS Token vs Pooling
-------------------------------

CLS Token Advantages:
- More flexible: can learn complex relationships between patches
- More expressive: not limited to simple aggregation functions
- Follows the same paradigm as successful NLP models like BERT
- Can attend differently to different patches based on content
- Provides a dedicated "classification" representation

CLS Token Disadvantages:
- Adds an extra token, slightly increasing computation
- May require more training data to learn effective representations
- Can sometimes focus too much on certain patches, ignoring others

Pooling Advantages:
- Simpler implementation with fewer parameters
- Forces consideration of all patches
- Can be more stable in some training scenarios
- Often works well for simpler classification tasks
- No extra token needed, slightly reducing computation

Pooling Disadvantages:
- Less expressive than learned aggregation
- Fixed aggregation function may not be optimal for all tasks
- Cannot learn to prioritize patches differently for different classes

When to Use Each Method:
-----------------------

Prefer CLS Token when:
- Working with complex, diverse datasets
- Training on large amounts of data
- Need to capture intricate relationships between image regions
- Using transfer learning from large pre-trained models
- Implementing models that need to explain their decisions (attention maps from CLS token can be interpretable)

Prefer Pooling when:
- Working with smaller datasets
- Computational efficiency is critical
- The task requires equal consideration of all image regions
- Implementing lightweight models for edge devices
- The classification task is relatively simple

Research Findings:
----------------
- The original ViT paper used the CLS token approach
- Some studies (e.g., Touvron et al. in DeiT) found comparable performance between CLS tokens and mean pooling
- CLS tokens tend to perform better as model size and dataset size increase
- For smaller models or datasets, the difference is often negligible
- Some recent architectures use a hybrid approach: both CLS token and pooled representations

Implementation Note:
------------------
The current model.py does not include a CLS token implementation. To add it, you would need to:
1. Create a learnable CLS token parameter
2. Prepend it to patch embeddings
3. Adjust position embeddings to account for the extra token
4. Extract the CLS token representation for classification
"""


"""
Flash Attention in Vision Transformers
======================================

What is Flash Attention?
-----------------------
Flash Attention is an algorithm that optimizes the computation of attention mechanisms in transformer 
models. Introduced by Dao et al. in the paper "FlashAttention: Fast and Memory-Efficient Exact Attention 
with IO-Awareness," it addresses the two main bottlenecks in standard attention implementations:

1. Memory bottleneck: Standard attention stores the full attention matrix of size O(N²), where N is 
   the sequence length, which becomes prohibitively large for long sequences.
   
2. Memory bandwidth bottleneck: Standard attention implementations are memory-bound rather than 
   compute-bound, meaning they spend more time reading/writing to GPU high bandwidth memory (HBM) 
   than performing computations.

How Flash Attention Works:
-------------------------
1. Tiling: Breaks the input matrices into smaller blocks that fit in faster SRAM cache
2. Recomputation: Trades computation for memory by recomputing certain values instead of storing them
3. Parallel Softmax: Computes softmax in a numerically stable way without materializing the full attention matrix

Key Benefits:
-----------
1. Speed: Up to 7.6x faster than standard attention on GPUs
2. Memory Efficiency: Reduces memory usage from O(N²) to O(N)
3. Scalability: Enables processing of much longer sequences
4. Accuracy: Provides exact (not approximate) attention computation

Implementation in Vision Transformers:
------------------------------------
In the model.py file, Flash Attention is implemented by transposing the query, key, and value tensors 
to optimize the matrix multiplication pattern:

Standard attention computes:
1. Attention scores: Q × K^T
2. Apply softmax
3. Weighted sum: softmax(QK^T) × V

Flash attention reorganizes this computation to minimize memory transfers and maximize GPU utilization.
The transposition of tensors (q.transpose(2, 3), etc.) is part of this optimization.

Mathematical Formulation:
-----------------------
Standard Attention:
   Attention(Q, K, V) = softmax(QK^T/√d) × V

Flash Attention optimizes this by:
1. Partitioning Q, K, V into blocks
2. Computing partial attention for each block
3. Combining results with a memory-efficient algorithm

Code Implementation Details:
--------------------------
In the SelfAttentionEncoder class:

1. Standard attention path:
   - Compute attention scores: torch.matmul(q, k.transpose(-2, -1))
   - Apply softmax: F.softmax(attention, dim=-1)
   - Apply to values: torch.matmul(attention, v)

2. Flash attention path:
   - Transpose dimensions: q.transpose(2, 3), etc.
   - Compute attention with optimized memory access patterns
   - Avoid materializing the full attention matrix

Performance Comparison:
---------------------
For a ViT model with:
- Sequence length of 196 (14×14 patches from 224×224 image)
- 12 attention heads
- Batch size of 32

Memory usage:
- Standard attention: ~1.8GB
- Flash attention: ~0.9GB

Speed (forward + backward pass):
- Standard attention: ~45ms
- Flash attention: ~15ms

When to Use Flash Attention:
--------------------------
1. High-resolution images: When processing larger images with more patches
2. Longer sequences: When working with video or 3D data
3. Memory-constrained environments: When GPU memory is limited
4. Training efficiency: To reduce training time for large models
5. Larger batch sizes: To fit more examples in memory during training

Limitations:
-----------
1. Implementation complexity: More complex than standard attention
2. Hardware-specific optimizations: May require tuning for different GPU architectures
3. Not always beneficial for very short sequences: The overhead might not be worth it for small inputs

"""



"""
Positional Embeddings in Vision Transformers
======================================

## What are Positional Embeddings?

Positional embeddings are vectors added to token embeddings to provide information about the position of each token in a sequence. In Vision Transformers, they help the model understand the spatial relationships between image patches.

## Why are Positional Embeddings Required?

- **Self-attention is permutation invariant**: Without positional information, the transformer cannot distinguish between different arrangements of the same tokens
- **Spatial context matters**: In images, the relative positions of patches contain crucial information
- **Structure preservation**: They help maintain the 2D structure of the image after it's flattened into a sequence of patches

## How Positional Embeddings Work

1. Generate position-specific vectors for each position in the sequence
2. Add these vectors to the corresponding token embeddings
3. The combined embeddings are then processed by the transformer layers

## Types of Positional Embeddings

### 1. Fixed Positional Embeddings

#### Sinusoidal (Sine-Cosine) Embeddings
- **Used in**: "Attention Is All You Need" (Vaswani et al., 2017)
- **How it works**: Uses sine and cosine functions of different frequencies
- **Formula**: 
  - PE(pos, 2i) = sin(pos/10000^(2i/d_model))
  - PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
- **Advantages**:
  - Can extrapolate to sequence lengths not seen during training
  - No additional parameters to learn
  - Encodes relative positions implicitly

### 2. Learnable Positional Embeddings

#### Absolute Positional Embeddings
- **Used in**: BERT, ViT (Vision Transformer)
- **How it works**: Directly learn a unique embedding vector for each position
- **Implementation**: Simple lookup table of embeddings that are learned during training
- **Advantages**:
  - Can capture more complex position-dependent patterns
  - Adapts to the specific task and data

#### Relative Positional Embeddings
- **Used in**: Transformer-XL, Music Transformer
- **How it works**: Learn embeddings for relative positions between tokens
- **Implementation**: Encodes the distance between tokens rather than absolute positions
- **Advantages**:
  - Better generalization to longer sequences
  - More efficient for certain tasks like music generation

### 3. 2D Positional Embeddings (for images)

#### Learned 2D Positional Embeddings
- **Used in**: Original ViT (Vision Transformer)
- **How it works**: Learn separate embeddings for each (x,y) position in the 2D grid
- **Implementation**: Typically a parameter matrix of shape (H×W, embedding_dim)

#### Factorized 2D Positional Embeddings
- **Used in**: Some ViT variants
- **How it works**: Learn separate embeddings for x and y coordinates, then combine them
- **Advantages**: Reduces parameter count and may generalize better

## Implementation in Vision Transformers

For your Vision Transformer implementation, you'll need to add positional embeddings to your model. Here's how you would typically implement learnable positional embeddings:

```python
# In the VisionTransformer class:
self.pos_embedding = nn.Parameter(torch.zeros(1, self.patch_embedding.num_patches + 1, embed_dim))  # +1 for [CLS] token
self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
self.pos_dropout = nn.Dropout(pos_drop)
```

Then in the forward method, you would:
1. Create patch embeddings
2. Add class token
3. Add positional embeddings
4. Apply dropout

## When to Use Different Types

- **Sinusoidal**: When you need to generalize to longer sequences than seen during training
- **Learnable Absolute**: When sequence length is fixed and you want maximum flexibility
- **Relative**: When modeling relative positions is more important than absolute positions
- **2D**: Specifically for image-based transformers like ViT

For our Vision Transformer implementation, learnable absolute positional embeddings are most common and would be appropriate to use.

"""


"""
Weight Initialization in Vision Transformers
==============================================

## Why Initialize Weights?

1. **Faster Convergence**: Proper initialization helps models converge faster during training
2. **Avoid Vanishing/Exploding Gradients**: Good initialization prevents gradients from becoming too small or too large
3. **Better Performance**: Models with proper initialization often achieve better final performance
4. **Reproducibility**: Consistent initialization ensures reproducible results

## Truncated Normal Distribution

In this code, I'm using `nn.init.trunc_normal_` which initializes weights using a truncated normal distribution:

```python
module.cls_token.data = nn.init.trunc_normal_(module.cls_token.data, mean=0, std=0.02)
```

### What is Truncated Normal?

- **Normal Distribution**: A standard normal (Gaussian) distribution with a specified mean and standard deviation
- **Truncated**: Values are "cut off" and resampled if they are more than 2 standard deviations from the mean
- **Parameters**:
  - `mean=0`: Center of the distribution (0 in your case)
  - `std=0.02`: Standard deviation (controls the spread of values)

## Why Use Truncated Normal for Vision Transformers?

The original Vision Transformer paper (and many subsequent implementations) use truncated normal initialization with a small standard deviation (0.02) because:

1. **Stability**: It prevents extreme initial values that could cause training instability
2. **Empirical Success**: This initialization scheme has been shown to work well for transformer architectures
3. **Signal Propagation**: It helps maintain appropriate signal magnitude through the network

## Different Initialization for Different Layers

Our code initializes different types of layers differently:

1. **Positional Embeddings & CLS Token**: Truncated normal with std=0.02
2. **Linear & Conv2d Layers**: Truncated normal with std=0.02, biases set to zero
3. **LayerNorm**: Biases set to zero, weights set to 1.0

This is standard practice in transformer models, as different layer types benefit from different initialization schemes.

## Impact on Training

Proper initialization is particularly important for Vision Transformers because:

1. They are deep networks with many parameters
2. They use self-attention which can be sensitive to initialization
3. They typically require careful training regimes to achieve good performance      

"""



""" 
Data Augmentation Techniques in Vision Transformers
===================================================

Data augmentation is a critical component of training deep learning models, especially in computer vision tasks. It helps improve model generalization, robustness, and performance by introducing variations to the training data. In the context of Vision Transformers (ViT), data augmentation techniques can significantly enhance the model's ability to generalize to unseen data.

## Types of Data Augmentation in Vision Transformers:

### 1. Spatial Transforms:

- **Random Cropping**: Randomly selects a portion of the image to be used as input
- **Random Resizing**: Resizes the image to a random size
- **Random Horizontal/Vertical Flipping**: Flips the image horizontally or vertically

### 2. Color Transforms:

- **Random Brightness**: Adjusts the brightness of the image
- **Random Contrast**: Adjusts the contrast of the image
- **Random Hue**: Adjusts the hue of the image

### 3. Noise Injection:

- **Random Noise Addition**: Adds random noise to the image
- **Random Dropout**: Randomly drops out pixels from the image

### 4. Geometric Transforms:

- **Random Affine Transform**: Applies random affine transformations (translation, rotation, scaling) to the image 

### 5. Label Smoothing:

- **Label Smoothing**: Introduces noise into the labels during training, helping the model to be more robust to label errors 

### 6. Mixup:

- **Mixup**: Combines two images and their labels by linearly interpolating their features and labels.

## Why Data Augmentation is Important in Vision Transformers:

1. **Generalization**: Data augmentation helps the model to learn more robust features that generalize well to unseen data
2. **Robustness**: It helps the model to be more resilient to small variations in the input data
3. **Performance**: It can improve the model's performance on unseen data

"""


"""
Mixup Data Augmentation Technique
===================================

Mixup is a data augmentation technique that helps improve model generalization and performance, especially for image classification tasks. It was introduced in the paper "mixup: Beyond Empirical Risk Minimization" (2017).

Mixup is a simple yet effective data augmentation technique that creates new training samples by linearly interpolating between pairs of images and their labels. Let me implement the Mixup function for your utils.py file and explain how it works:

## How Mixup Works:

1. **Basic Concept**: Mixup creates virtual training examples by blending two images and their corresponding labels:
   - x̃ = λ·x_i + (1-λ)·x_j
   - ỹ = λ·y_i + (1-λ)·y_j
   - where λ is sampled from a Beta(α,α) distribution

2. **Implementation Details**:
   - The mixing coefficient λ is drawn from a Beta distribution with parameter α
   - Larger α values create more diverse mixtures (typically α=1.0 works well)
   - The same λ is used for both image and label mixing

3. **Loss Calculation**:
   - For one-hot encoded labels: directly use the mixed labels
   - For class indices: calculate the loss as λ·loss(x̃,y_i) + (1-λ)·loss(x̃,y_j)

## Benefits of Mixup:

- **Improved Generalization**: Helps models generalize better by creating smoother decision boundaries
- **Regularization**: Acts as a form of regularization, reducing overfitting
- **Robustness**: Makes models more robust to adversarial examples
- **Simplicity**: Very easy to implement compared to more complex augmentation techniques


"""


"""
CutMix Data Augmentation Technique
=====================================

CutMix is an advanced data augmentation technique that helps improve model generalization and performance, especially for image classification tasks. It was introduced in the paper "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features" (2019).

## How CutMix Works:

1. **Basic Concept**: CutMix takes two images and creates a new training sample by:
   - Cutting out a rectangular region from one image
   - Pasting this region onto another image
   - Adjusting the target labels proportionally based on the area of the cut region

2. **Mathematical Representation**:
   - For two images (x_A, y_A) and (x_B, y_B)
   - The new image is: x̃ = M ⊙ x_A + (1-M) ⊙ x_B
   - The new label is: ỹ = λ·y_A + (1-λ)·y_B
   - Where M is a binary mask and λ is the mixing ratio (typically the area proportion)

## Benefits of CutMix:

- Improves generalization by creating diverse training samples
- Helps models learn to focus on multiple regions of an image
- Reduces overfitting by introducing regularization
- Often outperforms other mixing strategies like Mixup and Cutout
- Particularly effective for Vision Transformers and CNNs

## Implementation in PyTorch:
Here's a simple example of how you might implement CutMix in PyTorch:

```python
class CutMixCollator:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, batch):
        batch = default_collate(batch)
        
        # Only apply CutMix with some probability (e.g., 0.5)
        if torch.rand(1).item() > 0.5:
            return batch
            
        # Get batch size and image dimensions
        batch_size, c, h, w = batch[0].shape
        
        # Sample mixing parameter from beta distribution
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        
        # Sample random indices for image B
        indices = torch.randperm(batch_size)
        
        # Generate random bounding box for CutMix
        cut_ratio = torch.sqrt(1. - lam)
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)
        
        # Random center position for the box
        cx = torch.randint(w, (1,)).item()
        cy = torch.randint(h, (1,)).item()
        
        # Calculate box boundaries
        bbx1 = max(0, cx - cut_w // 2)
        bby1 = max(0, cy - cut_h // 2)
        bbx2 = min(w, cx + cut_w // 2)
        bby2 = min(h, cy + cut_h // 2)
        
        # Apply CutMix
        batch[0][:, :, bby1:bby2, bbx1:bbx2] = batch[0][indices, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust labels
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        batch[1] = batch[1] * lam + batch[1][indices] * (1. - lam)
        
        return batch


To use CutMix in your training loop, you would apply it at the batch level using the DataLoader's collate_fn:

```python
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size,
    shuffle=True,
    collate_fn=CutMixCollator(alpha=1.0)
)
```

"""