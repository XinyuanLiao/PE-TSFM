import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from .tsfm_config import TSFMConfig
from typing import Optional
from dataclasses import dataclass


@dataclass
class TSFMModelOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class TSFMforPretrainingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    prediction_output: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None

@dataclass
class TSFMforClassificationOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    prediction_logits: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None


class MultiheadAttention(nn.Module):
    def __init__(self, config: TSFMConfig):
        super(MultiheadAttention, self).__init__()
        self.num_heads = config.dim // config.dim_per_heads
        self.head_dim = config.dim_per_heads
        self.qkv = nn.Linear(config.dim, config.dim * 3)
        self.proj = nn.Linear(config.dim, config.dim)
        self.dropout = config.dropout

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0
        )
        attn = attn.transpose(1, 2).contiguous().view(B, L, D)
        attn = self.proj(attn)

        return attn


class Patchifier(nn.Module):
    def __init__(self, config: TSFMConfig):
        super(Patchifier, self).__init__()
        self.patch_size = config.patch_size

    def forward(self, x):
        B, C, L = x.shape
        assert L % self.patch_size == 0, "Sequence length must be divisible by the patch size."
        num_patches = L // self.patch_size
        x = x.reshape(B, C, num_patches, self.patch_size)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, config: TSFMConfig):
        super().__init__()
        self.dim = config.dim

        position = torch.arange(10000).unsqueeze(1)  # (10000, 1)
        div_term = torch.exp(torch.arange(0, self.dim, 2) * (-math.log(10000.0) / self.dim))  # (dim//2,)
        pe = torch.zeros(10000, self.dim)  # (10000, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
        self.register_buffer('pe', pe)

    def forward(self, x):
        B, C, N, D = x.shape
        assert D == self.dim, f"Expected input dimension {self.dim}, but got {D}"
        pos_encoding = self.pe[:N, :].unsqueeze(0).unsqueeze(0)  # (1, 1, N, D)
        return x + pos_encoding
    
class Embedding(nn.Module):
    def __init__(self, config: TSFMConfig):
        super(Embedding, self).__init__()
        self.config = config
        if config.shared_embedding:
            self.input_embedding = nn.Linear(config.patch_size, config.dim)
        else:
            self.input_embedding = nn.ModuleList()
            for _ in range(config.input_channels):
                self.input_embedding.append(nn.Linear(config.patch_size, config.dim))
            
    def forward(self, x):
        if self.config.shared_embedding:
            embeddings = self.input_embedding(x)  # (B, C, N, D)
        else:
            embeddings = [self.input_embedding[i](x[:, i, :, :]) for i in range(self.config.input_channels)]
            embeddings = torch.stack(embeddings, dim=1)
        return embeddings  # (B, C, N, D)
    
    
class TSFMEncoderLayer(nn.Module):
    def __init__(self, config: TSFMConfig):
        super(TSFMEncoderLayer, self).__init__()
        self.norm1 = nn.RMSNorm([config.dim], eps=config.norm_eps)
        self.attn = MultiheadAttention(config)
        
        self.norm2 = nn.RMSNorm([config.dim], eps=config.norm_eps) # normalization for channel attention

        self.norm3 = nn.RMSNorm([config.dim], eps=config.norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(config.dim, int(config.dim * config.ffn_ratio)),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(int(config.dim * config.ffn_ratio), config.dim),
        )

    def forward(self, x):
        B, C, N, D = x.shape

        # Pre norm and time-attention
        x = x.reshape(B*C, N, D)
        x = x + self.attn(self.norm1(x))
        x = x.reshape(B, C, N, D)

        # Pre norm and channel-attention
        x = x.transpose(2, 1).reshape(B*N, C, D)  # (B*N, C, D)
        x = x + self.attn(self.norm2(x))
        x = x.reshape(B, N, C, D).transpose(1, 2)  # (B, C, N, D)
    
        x = x.reshape(B*C, N, D)
        x = x + self.ffn(self.norm3(x))
        return x.reshape(B, C, N, D)  # (B, C, N, D)
        
    
class TSFMPretrainedModel(PreTrainedModel):
    config_class = TSFMConfig
    base_model_prefix = "tsfm"
    def _init_weights(self, module: nn.Module):
        """
        Initialize weights
        """
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                module.bias.data.zero_()
    
class TSFMEncoder(TSFMPretrainedModel):
    config_class = TSFMConfig

    def __init__(self, config: TSFMConfig):
        super(TSFMEncoder, self).__init__(config)
        self.config = config
        self.layers = nn.ModuleList([
            TSFMEncoderLayer(config) for _ in range(config.num_layers)
        ])

        self.post_init()

    def forward(self, x):
        """
        x: (B, C, N, D)
        """
        for layer in self.layers:
            x = layer(x) # (B, C, N, D)

        return TSFMModelOutput(
            last_hidden_state=x
        )

class PretrainHead(nn.Module):
    def __init__(self, config: TSFMConfig):
        super(PretrainHead, self).__init__()
        self.dropout = nn.Dropout(config.head_dropout)
        self.head = nn.Linear(config.dim, config.patch_size)

    def forward(self, x):
        # input shape: (B, C, N, D)
        embedding = self.head(self.dropout(x))
        return embedding

class ClassificationHead(nn.Module):
    def __init__(self, config: TSFMConfig):
        super(ClassificationHead, self).__init__()
        self.dropout = nn.Dropout(config.head_dropout)
        self.head = nn.Linear(config.dim * config.input_channels, config.num_classes)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # input shape: (B, C, N, D)
        x_avg = x.mean(dim=2)  # (B, C, D)
        logits = self.head(self.dropout(self.flatten(x_avg)))
        return logits

class TSFMforPretraining(TSFMPretrainedModel):
    config_class = TSFMConfig

    def __init__(self, config: TSFMConfig):
        super(TSFMforPretraining, self).__init__(config)
        self.config = config
        self.patching = Patchifier(config)
        self.embedder = Embedding(config)
        self.positional_encoding = PositionalEncoding(config)
        self.encoder = TSFMEncoder(config)
        
        self.mask_token = nn.Parameter(torch.zeros(config.patch_size), requires_grad=config.mask_token_trainable)
        self.head = PretrainHead(config)
        self.patch_size = config.patch_size

        self.post_init()

    def _normalize(self, inputs):
        means = inputs.mean(dim=2, keepdim=True)
        stds = inputs.std(dim=2, keepdim=True)
        return (inputs - means) / (stds + self.config.norm_eps)
    
    def _random_masking(self, input_tensor, mask_ratio):
        B, C, N, P = input_tensor.shape
        device = input_tensor.device

        num_mask = int(N * mask_ratio)
        if num_mask <= 0 or num_mask >= N:
            return input_tensor, torch.zeros(B, C, N, device=device)

        noise = torch.rand(B, C, N, device=device)
        _, mask_idx = torch.topk(noise, num_mask, dim=-1)

        patch_mask = torch.zeros(B, C, N, dtype=torch.bool, device=device)
        patch_mask.scatter_(-1, mask_idx, True)

        mask_vec = self.mask_token.expand(B, C, N, P)
        masked_tensor = torch.where(
            patch_mask.unsqueeze(-1),
            mask_vec,
            input_tensor
        )

        return masked_tensor, patch_mask

    def forward(self, x):
        # x: (B, C, L)
        x = self._normalize(x)
        x = self.patching(x)  # (B, C, N, P)

        masked_x, patch_mask= self._random_masking(x, self.config.mask_ratio)

        x_embed = self.embedder(masked_x)  # (B, C, N, D)
        x_embed = self.positional_encoding(x_embed)

        model_output = self.encoder(x_embed)
        embedding = model_output.last_hidden_state
        x_hat = self.head(embedding)
        loss = F.mse_loss(x_hat * patch_mask.unsqueeze(-1), x * patch_mask.unsqueeze(-1), reduction='sum') / (patch_mask.sum() * self.patch_size + self.config.norm_eps)

        return TSFMforPretrainingOutput(
            loss=loss,
            prediction_output=x_hat,
            last_hidden_state=model_output.last_hidden_state
        )

class TSFMforClassification(TSFMPretrainedModel):
    config_class = TSFMConfig

    def __init__(self, config: TSFMConfig):
        super(TSFMforClassification, self).__init__(config)
        self.config = config
        self.patching = Patchifier(config)
        self.embedder = Embedding(config)
        self.positional_encoding = PositionalEncoding(config)
        self.encoder = TSFMEncoder(config)
        self.head = ClassificationHead(config)
        
        self.post_init()

    def _normalize(self, inputs):
        means = inputs.mean(dim=2, keepdim=True)
        stds = inputs.std(dim=2, keepdim=True)
        return (inputs - means) / (stds + self.config.norm_eps)

    def forward(self, x, labels=None):
        x = self._normalize(x)
        x = self.patching(x)
        x = self.embedder(x)
        x = self.positional_encoding(x)
        model_output = self.encoder(x)
        embedding = model_output.last_hidden_state
        logits = self.head(embedding)

        return TSFMforClassificationOutput(
            loss=None if labels is None else F.cross_entropy(logits, labels, label_smoothing=self.config.label_smoothing),
            prediction_logits=logits,
            last_hidden_state=model_output.last_hidden_state
        )
