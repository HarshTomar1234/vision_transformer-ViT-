import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self,
                 img_size = 224,
                 patch_size = 16,
                 in_channels = 3,
                 embed_dim = 768,
                 bias = True):
        
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels = in_channels,
            out_channels = embed_dim,
            kernel_size = patch_size,
            stride = patch_size,  # it ensures that patches are not overlapping each other by keeping stride equal to patch_size
            bias = bias
        )
      
    def forward(self, x):
       
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x  # [batch_size, num_patches, embed_dim]


class SelfAttentionEncoder(nn.Module):
    def __init__(self,
                 embed_dim = 768,
                 num_heads = 12,
                 attention_dropout = 0.0,
                 projection_dropout = 0.0,
                 flash_attention = True):
                 
        super(SelfAttentionEncoder, self).__init__()
        self.num_heads = num_heads
        self.head_dim = int(embed_dim // num_heads)
        self.scale = self.head_dim ** -0.5 
        self.attention_dropout = attention_dropout
        self.flash_attention = flash_attention

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(projection_dropout)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        q = self.q(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        k = self.k(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        v = self.v(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]

        if self.flash_attention:
           x = F.scaled_dot_product_attention(q, k, v, dropout_p= self.attention_dropout if self.training else 0.0)

        else:
            attention = (q @ k.transpose(-2, -1)) * self.scale
            attention = attention.softmax(dim = -1)
            attention = self.attn_dropout(attention)
            x = (attention @ v)

        x = x.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)  
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


class MLP(nn.Module):
    def __init__(self,
                 in_features = 768,
                 mlp_ratio = 4,
                 mlp_p = 0):

        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, in_features * mlp_ratio)    
        self.act_f = nn.GELU()    
        self.dropout1 = nn.Dropout(mlp_p)
        self.fc2 = nn.Linear(in_features * mlp_ratio, in_features)
        self.dropout2 = nn.Dropout(mlp_p)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_f(x)
        x = self.dropout1(x)    

        x = self.fc2(x)
        x = self.dropout2(x)
        return x
        
class EncoderBlock(nn.Module):
    def __init__(self,
                 flash_attention = True,
                 embed_dim = 768,
                 num_heads = 12,
                 mlp_ratio = 4,
                 projection_dropout = 0,
                 attention_dropout = 0,
                 mlp_p = 0.0):

        super(EncoderBlock, self).__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttentionEncoder(embed_dim=embed_dim,
                                              num_heads=num_heads,
                                              attention_dropout=attention_dropout,
                                              projection_dropout=projection_dropout,
                                              flash_attention=flash_attention)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(in_features=embed_dim,
                       mlp_ratio=mlp_ratio,
                       mlp_p=mlp_p)

    def forward(self, x):   
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    A PyTorch implementation of the Vision Transformer (ViT) model. Implemented in the paper:
    'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale'
       "https://arxiv.org/abs/2010.11929v2"
    """
    def __init__(self,
                 img_size = 224,
                 patch_size = 16,
                 in_channels = 3,
                 num_classes = 1000,
                 flash_attention = True,
                 embed_dim = 768,
                 depth = 12,
                 num_heads = 12,
                 mlp_ratio = 4,
                 projection_dropout = 0,
                 attention_dropout = 0,
                 mlp_p = 0,
                 pos_drop = 0,
                 head_p = 0,
                 pooling = "cls",  # "cls" or "mean" or "max" or "avg"
                 custom_weight_init = True):

        super(VisionTransformer, self).__init__()

        self.pooling = pooling
        assert self.pooling in ["cls", "avg"]

        self.patch_embedding = PatchEmbedding(img_size=img_size,
                                              patch_size=patch_size,
                                              in_channels=in_channels,
                                              embed_dim=embed_dim)

        if pooling == "cls":
           num_tokens = self.patch_embedding.num_patches + 1   # cls token
        else:
           num_tokens = self.patch_embedding.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) 
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(pos_drop)   

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(flash_attention=flash_attention,
                         embed_dim=embed_dim,
                         num_heads=num_heads,
                         mlp_ratio=mlp_ratio,
                         projection_dropout=projection_dropout,
                         attention_dropout=attention_dropout,
                         mlp_p=mlp_p)
            for _ in range(depth)
            
        ]) 

        self.norm = nn.LayerNorm(embed_dim)
        self.head_drop = nn.Dropout(head_p)  
        self.head = nn.Linear(embed_dim, num_classes) 

        self.apply(self._init_weights) 

                 
    def cls_pos_embedding(self, x):
        if self.pooling == "cls":
           cls_token_expand = self.cls_token.expand(x.shape[0], -1, -1)  # [batch_size, 1, embed_dim]
           x = torch.cat((cls_token_expand, x), dim=1)

        x = x + self.pos_embedding
        x = self.pos_drop(x)
        return x


    def _init_weights(self, module: nn.Module):

        if isinstance(module, VisionTransformer):
            module.cls_token.data = nn.init.trunc_normal_(module.cls_token.data, mean= 0,  std=0.02)
            module.pos_embedding.data = nn.init.trunc_normal_(module.pos_embedding.data, mean=0,  std=0.02)

        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0,  std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

 
    def forward(self, x):

        x = self.patch_embedding(x)
        x = self.cls_pos_embedding(x)

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)

        x = self.norm(x)
        
        if self.pooling == "cls":
           x = x[:, 0]
        else:
           x = x.mean(dim=1)

        x = self.head_drop(x)
        x = self.head(x)
        return x


        
if __name__ == '__main__':
    
    rand = torch.randn(4, 3, 224, 224)
 
    vit = VisionTransformer(pooling = "avg")
    vit(rand)
    # print(vit)