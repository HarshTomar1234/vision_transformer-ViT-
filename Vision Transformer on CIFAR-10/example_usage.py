import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from vision_transformer import VisionTransformer, train_vit_on_cifar10, visualize_attention

def main():
    # Create a ViT model
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=10,  # CIFAR-10 has 10 classes
        embed_dim=384,   # Smaller embedding dimension
        depth=6,         # Fewer transformer blocks
        n_heads=6,       # Fewer attention heads
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.1
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Load a sample image from CIFAR-10 for demonstration
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load a single batch from CIFAR-10
    cifar_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    dataloader = DataLoader(cifar_dataset, batch_size=1, shuffle=True)
    sample_img, label = next(iter(dataloader))
    
    # Get class names
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print(f"Sample image class: {classes[label.item()]}")
    
    # Display the sample image
    img = sample_img[0].permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min())  # Normalize for display
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Sample Image: {classes[label.item()]}")
    plt.axis('off')
    
    # Forward pass through the model
    with torch.no_grad():
        outputs = model(sample_img)
        _, predicted = outputs.max(1)
    
    print(f"Model prediction: {classes[predicted.item()]}")
    
    # Visualize attention for this image
    visualize_attention(model, sample_img, head_idx=0, block_idx=0)
    
    # Print the option to train the model
    print("\nTo train the model on CIFAR-10, run:")
    print("train_vit_on_cifar10(model, epochs=5)")

if __name__ == "__main__":
    main()