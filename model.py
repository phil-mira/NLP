import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import numpy as np


class DevelopmentalConstraints:
    """Implements constraints that mirror human language acquisition stages"""

    def __init__(self):
        self.current_stage = 0
        self.stages = {
            0: "phonological",  # 0-6 months: focus on sound patterns
            1: "holophrastic",  # 6-18 months: single word utterances
            2: "telegraphic",  # 18-36 months: two-word combinations
            3: "syntactic"     # 36+ months: complex sentence structures
        }

    def get_attention_mask(self, sequence_length, stage=None):
        """Creates attention masks based on developmental stage"""
        if stage is None:
            stage = self.current_stage

        mask = torch.ones(sequence_length, sequence_length)

        if stage == 0:  # Phonological: attend only to local sound patterns
            bandwidth = 3
            for i in range(sequence_length):
                start = max(0, i - bandwidth)
                end = min(sequence_length, i + bandwidth + 1)
                mask[i] = 0
                mask[i, start:end] = 1

        elif stage == 1:  # Holophrastic: focus on individual words
            mask = torch.eye(sequence_length)

        elif stage == 2:  # Telegraphic: attend to pairs of words
            bandwidth = 2
            for i in range(sequence_length):
                start = max(0, i - bandwidth)
                end = min(sequence_length, i + bandwidth + 1)
                mask[i] = 0
                mask[i, start:end] = 1

        return mask


class LanguageAcquisitionModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_layers=4, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.developmental_constraints = DevelopmentalConstraints()

        # Position encoding
        self.register_buffer(
            "position_encoding",
            self._create_position_encoding(1024, hidden_size)
        )

        # Stage-specific processing layers
        self.phonological_conv = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=3, padding=1
        )

        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads)
            for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def _create_position_encoding(self, max_len, hidden_size):
        """Creates sinusoidal position encodings"""
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2) *
            (-math.log(10000.0) / hidden_size)
        )
        pos_encoding = torch.zeros(max_len, hidden_size)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding

    def forward(self, x):
        B, L = x.shape

        # Embedding + positional encoding
        x = self.embedding(x) + self.position_encoding[:L]

        # Apply stage-specific processing
        stage = self.developmental_constraints.current_stage

        if stage == 0:  # Phonological stage
            x = x.transpose(1, 2)
            x = self.phonological_conv(x)
            x = x.transpose(1, 2)

        # Get attention mask based on developmental stage
        attention_mask = self.developmental_constraints.get_attention_mask(L)
        attention_mask = attention_mask.to(x.device)

        # Process through transformer layers with constrained attention
        for layer in self.transformer_layers:
            x = layer(x, attention_mask)

        return self.output_layer(x)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, x, attention_mask=None):
        # Self-attention with mask
        attended = self.attention(
            x, x, x,
            attn_mask=attention_mask,
            need_weights=False
        )[0]
        x = self.norm1(x + attended)

        # Feed-forward
        ff = self.feed_forward(x)
        return self.norm2(x + ff)


class LanguageAcquisitionDataset(Dataset):
    """Dataset that implements curriculum learning based on developmental stages"""

    def __init__(self, texts, tokenizer, max_length=512):
        self.encoded_texts = []
        self.stage_appropriate_texts = defaultdict(list)

        for text in texts:
            encoded = tokenizer.encode(
                text, max_length=max_length, truncation=True)
            self.encoded_texts.append(encoded)

            # Categorize text by appropriate developmental stage
            if len(encoded) <= 3:  # Short sequences for phonological stage
                self.stage_appropriate_texts[0].append(encoded)
            if len(encoded) <= 5:  # Single word utterances for holophrastic
                self.stage_appropriate_texts[1].append(encoded)
            if len(encoded) <= 10:  # Short phrases for telegraphic
                self.stage_appropriate_texts[2].append(encoded)
            self.stage_appropriate_texts[3].append(
                encoded)  # All texts for syntactic

        self.current_stage = 0
        self.tokenizer = tokenizer
        self.max_length = max_length

    def set_stage(self, stage):
        self.current_stage = stage

    def __len__(self):
        return len(self.stage_appropriate_texts[self.current_stage])

    def __getitem__(self, idx):
        text = self.stage_appropriate_texts[self.current_stage][idx]

        # Create input and target sequences for self-supervised learning
        input_seq = text[:-1]
        target_seq = text[1:]

        # Pad sequences
        input_seq = F.pad(
            torch.tensor(input_seq),
            (0, self.max_length - len(input_seq)),
            value=self.tokenizer.pad_token_id
        )
        target_seq = F.pad(
            torch.tensor(target_seq),
            (0, self.max_length - len(target_seq)),
            value=self.tokenizer.pad_token_id
        )

        return input_seq, target_seq


def train_epoch(model, dataloader, optimizer, device):
    """Training loop for one epoch"""
    model.train()
    total_loss = 0

    for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)

        optimizer.zero_grad()
        output = model(input_seq)

        loss = F.cross_entropy(
            output.view(-1, output.size(-1)),
            target_seq.view(-1),
            ignore_index=tokenizer.pad_token_id
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
