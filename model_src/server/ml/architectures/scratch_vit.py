import logging
import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PatchEmbedding(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("Image size must be divisible by patch size.")
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Convolutional layer to create patches and project them
        # This is a common way to implement patch embedding efficiently.
        # It's equivalent to splitting, flattening, and then a linear layer.
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W) e.g., (B, 3, 224, 224)
        x = self.projection(x)  # (B, embed_dim, num_patches_h, num_patches_w) e.g. (B, 768, 14, 14)
        x = x.flatten(2)  # (B, embed_dim, num_patches) e.g. (B, 768, 196)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim) e.g. (B, 196, 768)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention Block"""

    def __init__(self, embed_dim: int, num_heads: int, attention_dropout: float = 0.0, projection_dropout: float = 0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("Embedding dimension must be divisible by number of heads.")
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5  # Scaling factor for dot product attention

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)  # Query, Key, Value projections
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape  # Batch, Num_Tokens (patches+CLS), Channels (embed_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv shape: (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Feed-Forward Network / MLP Block for Transformer"""

    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None,
                 activation: nn.Module = nn.GELU, dropout: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = activation()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """A single Transformer Encoder Block"""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 attention_dropout: float = 0.0, projection_dropout: float = 0.0, mlp_dropout: float = 0.0,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, attention_dropout, projection_dropout)
        self.norm2 = norm_layer(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(in_features=embed_dim, hidden_features=mlp_hidden_dim, dropout=mlp_dropout)
        # TODO Note: Some ViT implementations include dropout after attention/MLP (projection_dropout/mlp_dropout in our case)
        # and also a separate "drop_path" or "stochastic depth" for regularization, not included here for simplicity.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))  # Apply attention
        x = x + self.mlp(self.norm2(x))  # Apply MLP
        return x


class ScratchViT(nn.Module):
    def __init__(self,
                 num_classes: int,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embed_dim: int = 768,  # Corresponds to hidden_dim in torchvision ViT
                 depth: int = 12,  # Number of TransformerEncoderBlocks
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 attention_dropout: float = 0.0,
                 projection_dropout: float = 0.0,  # Dropout after attention projection and MLP output
                 mlp_dropout: float = 0.0,  # Dropout within MLP hidden layers
                 pos_embedding_type: str = 'learnable',  # 'learnable' or 'sinusoidal_2d'
                 head_hidden_dims: Optional[List[int]] = None,  # For custom MLP head
                 head_dropout_rate: float = 0.0  # Dropout for the final classification head
                 ):
        super().__init__()
        if num_classes <= 0: raise ValueError("Number of classes must be positive.")
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # 1. Patch Embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # 2. CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # (1, 1, embed_dim)
        torch.nn.init.normal_(self.cls_token, std=0.02)  # Initialize CLS token

        # 3. Positional Embeddings
        self.pos_embedding_type = pos_embedding_type
        # +1 for CLS token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        torch.nn.init.normal_(self.pos_embed, std=0.02)  # Initialize learnable pos_embed
        # Note: If using sinusoidal, it would be calculated, not learned, and might not be nn.Parameter

        # 4. Transformer Encoder
        self.encoder_blocks = nn.Sequential(
            *[TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,  # Passed to MHA and MLP
                mlp_dropout=mlp_dropout  # Passed to MLP hidden layer
            ) for _ in range(depth)]
        )

        # 5. Final LayerNorm after encoder (common practice)
        self.norm = nn.LayerNorm(embed_dim)

        # 6. Classification Head
        if head_hidden_dims and len(head_hidden_dims) > 0:
            head_layers: List[nn.Module] = []
            current_in_features = embed_dim
            for hidden_dim in head_hidden_dims:
                head_layers.append(nn.Linear(current_in_features, hidden_dim))
                head_layers.append(nn.ReLU(inplace=True))  # Or GELU
                if head_dropout_rate > 0:
                    head_layers.append(nn.Dropout(head_dropout_rate))
                current_in_features = hidden_dim
            head_layers.append(nn.Linear(current_in_features, num_classes))
            self.head = nn.Sequential(*head_layers)
        else:
            layers_for_simple_head: List[nn.Module] = []
            if head_dropout_rate > 0.0:
                layers_for_simple_head.append(nn.Dropout(head_dropout_rate))
            layers_for_simple_head.append(nn.Linear(embed_dim, num_classes))
            self.head = nn.Sequential(*layers_for_simple_head)

        self.apply(self._init_weights)  # Apply custom weight initialization

        logger.info(
            f"ViTFromScratch initialized: img_size={img_size}, patch_size={patch_size}, embed_dim={embed_dim}, depth={depth}, heads={num_heads}")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        """
        Interpolate positional encoding for different image sizes.
        Only implemented for 2D learnable positional embeddings.
        x: input tensor (B, num_patches + 1, embed_dim)
        w: new width of patch grid
        h: new height of patch grid
        """
        npatch = x.shape[1] - 1  # Number of patch tokens (excluding CLS)
        N = self.pos_embed.shape[1] - 1  # Number of patch tokens pos_embed was trained on
        if npatch == N and w == h:  # No interpolation needed
            return self.pos_embed

        logger.info(f"Interpolating positional embeddings from {N} to {npatch} patches.")
        class_pos_embed = self.pos_embed[:, 0]  # CLS token pos_embed
        patch_pos_embed = self.pos_embed[:, 1:]  # Patch token pos_embeds

        dim = x.shape[-1]  # embed_dim

        # Calculate original grid size (assuming square)
        w0 = h0 = int(math.sqrt(N))
        if w0 * h0 != N:  # Not a square grid, or something is wrong
            logger.warning(f"Positional embedding (N={N}) not from a square grid. Cannot interpolate simply.")
            return self.pos_embed  # Return original if not square

        # Reshape to 2D grid format
        patch_pos_embed = patch_pos_embed.reshape(1, w0, h0, dim).permute(0, 3, 1, 2)
        # Interpolate
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(w, h),
            mode='bicubic',
            align_corners=False,
        )
        # Reshape back to sequence and flatten
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        # Concatenate with CLS token's positional embedding
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape  # Batch, Channels, Height, Width

        # 1. Patch Embedding
        x_patched = self.patch_embed(x)  # (B, num_patches, embed_dim)
        num_current_patches_h = H // self.patch_embed.patch_size
        num_current_patches_w = W // self.patch_embed.patch_size

        # 2. Prepend CLS token
        # cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, embed_dim)
        # x_with_cls = torch.cat((cls_tokens, x_patched), dim=1) # (B, num_patches + 1, embed_dim)

        # Correct way to tile cls_token for batch
        cls_token = self.cls_token.expand(x_patched.shape[0], -1, -1)
        x_with_cls = torch.cat((cls_token, x_patched), dim=1)

        # 3. Add Positional Embeddings
        # Interpolate positional embeddings if image size/patch size leads to different num_patches
        if self.pos_embedding_type == 'learnable':
            # Check if current number of patches matches the trained pos_embed size
            # (num_patches_h * num_current_patches_w) is num_current_patches
            if (num_current_patches_h * num_current_patches_w) != (self.pos_embed.shape[1] - 1):
                pos_embed_interp = self.interpolate_pos_encoding(x_with_cls, num_current_patches_w,
                                                                 num_current_patches_h)
                x_final_embed = x_with_cls + pos_embed_interp
            else:
                x_final_embed = x_with_cls + self.pos_embed
        elif self.pos_embedding_type == 'sinusoidal_2d':
            # TODO: Implement 2D Sinusoidal Positional Encoding generation
            # This would depend on num_current_patches_h, num_current_patches_w, embed_dim
            raise NotImplementedError("Sinusoidal 2D positional encoding not yet implemented.")
        else:
            x_final_embed = x_with_cls  # No positional encoding or unhandled type

        # 4. Transformer Encoder
        x_encoded = self.encoder_blocks(x_final_embed)

        # 5. Final LayerNorm
        x_normed = self.norm(x_encoded)

        # 6. Get CLS token output for classification
        cls_token_output = x_normed[:, 0]  # (B, embed_dim)

        # 7. Classification Head
        logits = self.head(cls_token_output)  # (B, num_classes)
        return logits
