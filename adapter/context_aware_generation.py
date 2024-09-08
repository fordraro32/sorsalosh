import torch
import torch.nn as nn

class ContextAwareGeneration(nn.Module):
    def __init__(self, config):
        super(ContextAwareGeneration, self).__init__()
        self.max_sequence_length = config.max_sequence_length
        self.lstm = nn.LSTM(config.input_dim, config.input_dim // 2, num_layers=2, bidirectional=True, batch_first=True)
        self.attention = nn.MultiheadAttention(config.input_dim, num_heads=8)

    def forward(self, x):
        # LSTM for capturing long-range dependencies
        lstm_out, _ = self.lstm(x)
        
        # Self-attention for context awareness
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        return attn_out
