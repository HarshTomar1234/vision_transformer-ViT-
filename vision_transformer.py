import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import math


class PatchEmbedding(nn.Module):
    """
    Split the image into patches and embed them.
    
    Args:
        img_size (int): Size of the input image (assumed to be square)
        patch_size (int): Size of each patch (assumed to be square)
        in_channels (int): Number of input channels
        embed_dim (int): Embedding dimension
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Linear projection of flattened patches
        # This is equivalent to a Conv2d with kernel_size=patch_size and stride=patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, in_channels, img_size, img_size]
            
        Returns:
            Tensor of shape [batch_size, n_patches, embed_dim]
        """
        # x shape: [batch_size, in_channels, img_size, img_size]
        x = self.proj(x)  # [batch_size, embed_dim, img_size/patch_size, img_size/patch_size]
        x = x.flatten(2)  # [batch_size, embed_dim, n_patches]
        x = x.transpose(1, 2)  # [batch_size, n_patches, embed_dim]
        
        return x


class Attention(nn.Module):
    """
    Multi-head Self Attention mechanism.
    
    Args:
        dim (int): Input dimension
        n_heads (int): Number of attention heads
        qkv_bias (bool): If True, add a learnable bias to query, key, value
        attn_drop (float): Dropout rate for attention weights
        proj_drop (float): Dropout rate for projection
    """
    def __init__(self, dim, n_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5  # Scaling factor for dot product
        
        # Combined projection for query, key, value
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            
        Returns:
            Tensor of shape [batch_size, seq_len, dim]
        """
        batch_size, seq_len, dim = x.shape
        
        # Project input to query, key, value and reshape to multi-head format
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, n_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each has shape [batch_size, n_heads, seq_len, head_dim]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [batch_size, n_heads, seq_len, seq_len]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention weights to values
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron block.
    
    Args:
        in_features (int): Number of input features
        hidden_features (int): Number of hidden features
        out_features (int): Number of output features
        drop (float): Dropout rate
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()  # GELU activation as used in the original Transformer
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """
    Transformer block with attention and MLP.
    
    Args:
        dim (int): Input dimension
        n_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
        qkv_bias (bool): If True, add a learnable bias to query, key, value
        drop (float): Dropout rate
        attn_drop (float): Dropout rate for attention weights
    """
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        
        # Layer normalization before attention
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        # Multi-head self-attention
        self.attn = Attention(
            dim, n_heads=n_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop
        )
        
        # Layer normalization before MLP
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        # MLP block
        self.mlp = MLP(
            in_features=dim, 
            hidden_features=int(dim * mlp_ratio),
            drop=drop
        )
        
    def forward(self, x):
        # Apply attention with residual connection
        x = x + self.attn(self.norm1(x))
        # Apply MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer model.
    
    Args:
        img_size (int): Input image size
        patch_size (int): Patch size
        in_channels (int): Number of input channels
        num_classes (int): Number of classes for classification
        embed_dim (int): Embedding dimension
        depth (int): Number of transformer blocks
        n_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
        qkv_bias (bool): If True, add a learnable bias to query, key, value
        drop_rate (float): Dropout rate
        attn_drop_rate (float): Dropout rate for attention weights
    """
    def __init__(
        self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
        embed_dim=768, depth=12, n_heads=12, mlp_ratio=4.0, qkv_bias=True,
        drop_rate=0., attn_drop_rate=0.
    ):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size, patch_size=patch_size,
            in_channels=in_channels, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.n_patches
        
        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])
        
        # Layer normalization after transformer blocks
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize patch embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Apply to all modules
        self.apply(self._init_weights_recursive)
        
    def _init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward_features(self, x):
        # Convert image to patch embeddings
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        
        # Add class token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_token, x), dim=1)  # [B, num_patches + 1, embed_dim]
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Apply layer normalization
        x = self.norm(x)
        
        # Return class token
        return x[:, 0]
    
    def forward(self, x):
        # Get features from the class token
        x = self.forward_features(x)
        # Apply classification head
        x = self.head(x)
        return x


# Example usage and visualization functions
def visualize_attention(model, img, head_idx=0, block_idx=0):
    """
    Visualize attention maps for a specific head and block.
    
    Args:
        model: Vision Transformer model
        img: Input image tensor [1, C, H, W]
        head_idx: Index of attention head to visualize
        block_idx: Index of transformer block to visualize
    """
    model.eval()
    
    # Register hook to get attention maps
    attention_maps = []
    
    def hook_fn(module, input, output):
        # The Attention module in our implementation doesn't return attention weights
        # We need to compute them here
        batch_size, seq_len, dim = input[0].shape
        
        # Get qkv projections
        qkv = module.qkv(input[0]).reshape(batch_size, seq_len, 3, module.n_heads, module.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * module.scale
        attn = attn.softmax(dim=-1)
        
        attention_maps.append(attn.detach())
    
    # Register hook on the specified attention module
    hook = model.blocks[block_idx].attn.register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        _ = model(img)
    
    # Remove hook
    hook.remove()
    
    if not attention_maps:
        print("No attention maps were captured. Check if the model has the expected structure.")
        return
    
    # Get attention map for the specified head
    attn_map = attention_maps[0][0, head_idx]  # [seq_len, seq_len]
    
    # Visualize attention from cls token to patches
    cls_attn = attn_map[0, 1:]  # Attention from CLS to patches
    
    # Reshape to match image patches
    patch_size = model.patch_embed.patch_size
    num_patches_per_side = img.shape[-1] // patch_size
    cls_attn = cls_attn.reshape(num_patches_per_side, num_patches_per_side)
    
    # Visualize
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    # For random tensor, just show a blank image
    if isinstance(img, torch.Tensor) and img.shape[1] == 3:
        # Normalize for visualization
        img_vis = img[0].permute(1, 2, 0).cpu()
        img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min())
        plt.imshow(img_vis)
    else:
        plt.imshow(np.zeros((224, 224, 3)))
    plt.title("Input Image")
    
    plt.subplot(1, 2, 2)
    plt.imshow(cls_attn.cpu(), cmap='viridis')
    plt.title(f"Attention Map (Block {block_idx}, Head {head_idx})")
    
    plt.tight_layout()
    plt.savefig('attention_visualization.png')  # Save the figure
    plt.show()


def train_vit_on_cifar10(model, epochs=5, batch_size=64, lr=1e-4):
    """
    Train the Vision Transformer model on CIFAR-10 dataset.
    
    Args:
        model: Vision Transformer model
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
    """
    # Data transformations
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match ViT input size
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if i % 100 == 99:
                print(f'Epoch: {epoch+1}, Batch: {i+1}, Loss: {running_loss/100:.3f}, Acc: {100.*correct/total:.3f}%')
                running_loss = 0.0
        
        # Evaluate on test set
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        print(f'Epoch: {epoch+1}, Test Loss: {test_loss/len(test_loader):.3f}, Test Acc: {100.*correct/total:.3f}%')


# Example of creating and using the model
if __name__ == "__main__":
    # Creating a smaller ViT model for demonstration
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
    
    # Uncomment to train the model on CIFAR-10
    train_vit_on_cifar10(model, epochs=5)
    
    # Example of visualizing attention (requires a trained model)
    img = torch.randn(1, 3, 224, 224)  # Random image for demonstration
    visualize_attention(model, img, head_idx=0, block_idx=0)
