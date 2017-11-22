import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class CnnTextClassifier(nn.Module):
    def __init__(self, vocab_size, window_sizes=(3, 4, 5)):
        super(CnnTextClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, config.emb_size)

        self.convs = nn.ModuleList([
            nn.Conv2d(1, config.num_filters, [window_size, config.emb_size], padding=(window_size - 1, 0))
            for window_size in window_sizes
        ])

        self.fc = nn.Linear(config.num_filters * len(window_sizes), config.num_classes)

    def forward(self, x):
        x = self.embedding(x)           # [B, T, E]

        # Apply a convolution + max pool layer for each window size
        x = torch.unsqueeze(x, 1)       # [B, C, T, E] Add a channel dim.
        xs = []
        for conv in self.convs:
            x2 = F.relu(conv(x))        # [B, F, T, 1]
            x2 = torch.squeeze(x2, -1)  # [B, F, T]
            x2 = F.max_pool1d(x2, x2.size(2))  # [B, F, 1]
            xs.append(x2)
        x = torch.cat(xs, 2)            # [B, F, window]

        # FC
        x = x.view(x.size(0), -1)       # [B, F * window]
        logits = self.fc(x)             # [B, class]

        # Prediction
        probs = F.softmax(logits)       # [B, class]
        classes = torch.max(probs, 1)   # [B]

        return probs, classes
