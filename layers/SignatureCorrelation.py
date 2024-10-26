import torch
import torch.nn as nn
import signatory
import torch.nn.functional as F


# Lead-lag transformation function
def lead_lag_transform(x):
    # x shape: [batch_size, seq_len, channels]
    lead = x.unsqueeze(2)  # [batch_size, seq_len, 1, channels]
    lag = x.unsqueeze(2)  # [batch_size, seq_len, 1, channels]
    lag = torch.cat([lag[:, 1:], lag[:, -1:]], dim=1)  # Shifted sequence to create lag effect
    lead_lag = torch.cat([lead, lag], dim=2)  # [batch_size, seq_len, 2, channels]
    return lead_lag.reshape(x.shape[0], x.shape[1] * 2, x.shape[2])


class Augment(nn.Module):
    def __init__(self,
                 in_channels: int,  # Correspond à d_model, ici 8
                 layer_sizes: tuple,  # Par exemple (64, 128)
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: bool = True,
                 activation: callable = F.relu,
                 include_original: bool = True,
                 include_time: bool = True):
        super(Augment, self).__init__()

        self.include_original = include_original
        self.include_time = include_time

        # Convolutional layers for augmentation
        self.convs = nn.ModuleList()
        if layer_sizes:
            # Première convolution : prend en entrée les `in_channels` (ici 8)
            self.convs.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=layer_sizes[0],  # Première couche : par exemple, 64
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias
            ))

            # Couches de convolutions suivantes
            last_layer_channels = layer_sizes[0]
            for augment_channel in layer_sizes[1:]:
                self.convs.append(nn.Conv1d(
                    in_channels=last_layer_channels,
                    out_channels=augment_channel,
                    kernel_size=1,  # Kernel size 1 pour transformation simple
                    bias=bias
                ))
                last_layer_channels = augment_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward operation.

        Arguments:
            x (torch.Tensor): The path to augment. Shape: [batch_size, seq_len, channels].
        Returns:
            The augmented path.
        """
        if len(x.shape) != 3:
            raise RuntimeError('Argument x should have three dimensions, (batch, seq_len, channels). '
                               f'Given shape {x.shape}.')

        # Longueur tronquée pour la convolution
        len_truncated = x.size(1) - self.convs[0].kernel_size[0] + 1
        pieces = []

        # Inclure les données d'origine
        if self.include_original:
            truncated_x = x.narrow(1, self.convs[0].kernel_size[0] - 1, len_truncated)
            pieces.append(truncated_x)

        # Inclure le temps
        if self.include_time:
            time = torch.linspace(0, 1, len_truncated, dtype=torch.float, device=x.device)
            time = time.unsqueeze(1).expand(x.size(0), len_truncated, 1)
            pieces.append(time)

        # Appliquer les convolutions pour l'augmentation
        if self.convs:
            augmented_x = self.convs[0](x.transpose(1, 2))  # [batch_size, in_channels, seq_len] pour Conv1d
            for conv in self.convs[1:]:
                augmented_x = self.activation(conv(augmented_x))
            augmented_x = augmented_x.transpose(1, 2)  # [batch_size, seq_len, out_channels]
            pieces.append(augmented_x)

        # Concaténer le long de l'axe des canaux
        return torch.cat(pieces, dim=2)



class SignatureBlock(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, embed_dim, depth=4):
        super(SignatureBlock, self).__init__()
        print(f'Signature block used with depth={depth}')

        # Augment the input data
        self.augment = Augment(in_channels=embed_dim,  # Adjusted to match E (embedding dimension)
                               layer_sizes=(64, 128),
                               kernel_size=3,
                               include_original=True,
                               include_time=True)
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_len = seq_len

        # Define a linear layer to project signature output back to the desired dimensions
        augmented_channels = embed_dim + 1 + 128  # original + time + augment_channels (last layer)
        self.linear = nn.Linear(signatory.signature_channels(augmented_channels * 2, depth), out_channels)

    def forward(self, q, k, v, mask):
        B, L, H, E = q.shape
        q = q.permute(0, 2, 1, 3).reshape(B * H, L, E)  # Reshape for processing: [B*H, L, E]
        print(f'q shape after reshape: {q.shape}')

        # Apply augmentation
        q_augmented = self.augment(q)
        print(f'q_augmented shape: {q_augmented.shape}')

        # Apply lead-lag transformation
        q_lead_lag = lead_lag_transform(q_augmented)
        print(f'q_lead_lag shape: {q_lead_lag.shape}')

        # Compute signature
        q_signature = signatory.signature(q_lead_lag, depth=self.depth)
        print(f'q_signature shape: {q_signature.shape}')

        # Apply linear transformation to match the expected output dimensions
        q_transformed = self.linear(q_signature)
        print(f'q_transformed shape: {q_transformed.shape}')

        # Reshape back to [B, H, L, E]
        q_transformed = q_transformed.view(B, H, self.seq_len, -1)
        return (q_transformed, None)


class SignatureCrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, embed_dim, depth=4, activation='tanh'):
        super(SignatureCrossAttention, self).__init__()
        print(f'Signature-based cross attention used with depth={depth}')

        self.augment_q = Augment(in_channels=embed_dim,  # Adjusted to match E (embedding dimension)
                                 layer_sizes=(64, 128),
                                 kernel_size=3)
        self.augment_k = Augment(in_channels=embed_dim,  # Adjusted to match E (embedding dimension)
                                 layer_sizes=(64, 128),
                                 kernel_size=3)
        self.depth = depth
        self.q_linear = nn.Linear(signatory.signature_channels(192 * 2, depth), out_channels)
        self.k_linear = nn.Linear(signatory.signature_channels(192 * 2, depth), out_channels)
        self.activation = activation

    def forward(self, q, k, v, mask):
        B, L, H, E = q.shape
        q = q.permute(0, 2, 1, 3).reshape(B * H, L, E)
        k = k.permute(0, 2, 1, 3).reshape(B * H, L, E)

        # Apply augmentation and lead-lag transformation
        q_augmented = lead_lag_transform(self.augment_q(q))
        k_augmented = lead_lag_transform(self.augment_k(k))
        print(f'q_augmented shape: {q_augmented.shape}, k_augmented shape: {k_augmented.shape}')

        # Compute signatures
        q_signature = signatory.signature(q_augmented, depth=self.depth)
        k_signature = signatory.signature(k_augmented, depth=self.depth)
        print(f'q_signature shape: {q_signature.shape}, k_signature shape: {k_signature.shape}')

        # Linear transformations
        q_proj = self.q_linear(q_signature)
        k_proj = self.k_linear(k_signature)
        print(f'q_proj shape: {q_proj.shape}, k_proj shape: {k_proj.shape}')

        # Compute attention
        attention = torch.einsum('bxe,bye->bxy', q_proj, k_proj)
        if self.activation == 'tanh':
            attention = torch.tanh(attention)
        elif self.activation == 'softmax':
            attention = torch.softmax(attention, dim=-1)
        print(f'Attention shape: {attention.shape}')

        # Reshape back and apply to v
        attention = attention.view(B, H, q_proj.size(0), k_proj.size(0))
        out = torch.einsum('bhxy,bhye->bhxe', attention, v)
        return (out, None)
