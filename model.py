import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, input_dim=512, model_dim=512, num_heads=8, num_layers=6, output_dim=512):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.output_proj = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        # Ensure shape [seq_len, batch_size, feature_dim] for PyTorch Transformer
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.input_proj(x)
        x = self.transformer(x)
        x = self.output_proj(x)
        return x

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size=1000, input_dim=512):
        super().__init__()
        self.x = torch.randn(size, input_dim)
        self.y = torch.randn(size, input_dim)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
