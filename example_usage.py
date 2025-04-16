import torch
from vision_transformer import VisionTransformer
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def load_and_preprocess_image(image_path, img_size=224):
    """
    Load and preprocess an image for the Vision Transformer.
    
    Args:
        image_path: Path to the image file
        img_size: Size to resize the image to
        
    Returns:
        Preprocessed image tensor
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def visualize_patches(image_tensor, patch_size=16):
    """
    Visualize how an image is split into patches.
    
    Args:
        image_tensor: Input image tensor [1, C, H, W]
        patch_size: Size of each patch
    """
    # Convert to numpy for visualization
    img = image_tensor.squeeze(0).permute(1, 2, 0).numpy()
    
    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    img = np.clip(img, 0, 1)
    
    # Create a figure
    plt.figure(figsize=(10, 10))
    
    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    
    # Plot the image with patch grid
    plt.subplot(1, 2, 2)
    plt.imshow(img)
    
    # Add grid lines for patches
    h, w = img.shape[0], img.shape[1]
    for i in range(0, h, patch_size):
        plt.axhline(y=i, color='r', linestyle='-', alpha=0.3)
    for j in range(0, w, patch_size):
        plt.axvline(x=j, color='r', linestyle='-', alpha=0.3)
    
    plt.title(f"Image Split into {patch_size}x{patch_size} Patches")
    plt.tight_layout()
    plt.show()

def main():
    print("Loading pre-trained model...")
    
    # let's use a pre-trained model from torchvision
    model = models.vit_b_16(pretrained=True)
    model.eval()
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Load your own image
    image_path = "images/ai_learner.jpeg"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    image_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    
    # Visualize how the image is split into patches
    visualize_patches(image_tensor)
    
    # Forward pass
    with torch.no_grad():
        output = model(image_tensor)
    
    # Get ImageNet class labels
    try:
        # Try to load ImageNet class labels
        import json
        import requests
        
        # Download ImageNet class index
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        response = requests.get(url)
        imagenet_classes = [line.strip() for line in response.text.splitlines()]
        
        # Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Get top 5 predictions
        top5_prob, top5_idx = torch.topk(probabilities, 5)
        
        # Print results with class names
        print("\nTop predicted classes:")
        for i in range(5):
            print(f"{top5_prob[i].item()*100:.2f}% - {imagenet_classes[top5_idx[i]]}")
            
    except Exception as e:
        print(f"Could not load ImageNet classes: {e}")
        print("Top predicted classes (indices only):")
        print(output[0].topk(5))

if __name__ == "__main__":
    main()
