import numpy as np
import torch
import torch.nn as nn
import signatory

class SignatureBlock(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, depth=2):
        super(SignatureBlock, self).__init__()
        print('Signature-based block used!')
        self.depth = depth
        self.max_features = 1024  # Limiter à une taille raisonnable
        self.in_features = min((in_channels * (in_channels**self.depth - 1)) // (in_channels - 1), self.max_features)
        self.out_channels = out_channels
        self.linear = nn.Linear(self.in_features, self.out_channels)

    def forward(self, q, k, v, mask):
        # q.size() = [B, L, H, E]
        B, L, H, E = q.shape
        # Reshape to apply signature transformation
        x = q.permute(0, 2, 3, 1).contiguous().view(B * H, E, L)  # shape = [B*H, E, L]
        
        # Compute the signature up to specified depth
        x_signature = signatory.signature(x, self.depth)  # shape = [B*H, terms]
        
        # Apply a linear transformation to map the signature back to desired dimension
        x_transformed = self.linear(x_signature)  # shape = [B*H, out_channels]

        # Reshape back to the original dimensions
        x_transformed = x_transformed.view(B, H, E, -1).permute(0, 3, 1, 2)  # shape = [B, L, H, E]
        return (x_transformed, None)

class SignatureCrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, depth=2, activation='tanh'):
        super(SignatureCrossAttention, self).__init__()
        print('Signature-based cross attention used!')
        self.activation = activation
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear_q = nn.Linear((in_channels * (in_channels**self.depth - 1)) // (in_channels - 1), out_channels)
        self.linear_kv = nn.Linear((in_channels * (in_channels**self.depth - 1)) // (in_channels - 1), out_channels)

    def forward(self, q, k, v, mask):
        # q, k, v size = [B, L, H, E]
        B, L, H, E = q.shape
        xq = q.permute(0, 2, 3, 1).contiguous().view(B * H, E, L)  # [B*H, E, L]
        xk = k.permute(0, 2, 3, 1).contiguous().view(B * H, E, L)
        xv = v.permute(0, 2, 3, 1).contiguous().view(B * H, E, L)

        # Compute signature for q, k, v
        xq_signature = signatory.signature(xq, self.depth)
        xk_signature = signatory.signature(xk, self.depth)
        xv_signature = signatory.signature(xv, self.depth)

        # Linear transformation
        q_transformed = self.linear_q(xq_signature)  # [B*H, out_channels]
        kv_transformed = self.linear_kv(xk_signature * xv_signature)  # Simple multiplication for attention

        # Reshape and apply attention mechanism
        q_transformed = q_transformed.view(B, H, E, -1).permute(0, 3, 1, 2)  # [B, L, H, E]
        kv_transformed = kv_transformed.view(B, H, E, -1).permute(0, 3, 1, 2)  # [B, L, H, E]

        # Apply the attention mechanism in time domain (simple dot product)
        attn_output = torch.einsum('blhe,blhe->blhe', q_transformed, kv_transformed)
        return (attn_output, None)
