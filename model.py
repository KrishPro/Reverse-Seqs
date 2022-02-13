"""
Written by KrishPro @ KP
"""

import torch.nn as nn
import torch
import math


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, dropout: float) -> None:
        super().__init__()
        
        self.emb_size = emb_size
        self.embedding_layer = nn.Embedding(vocab_size, self.emb_size)
        self.positional_encoding = PositionalEncoding(self.emb_size, dropout)
    
    def forward(self, indices: torch.Tensor):
        assert indices.dtype == torch.long, f"Indices to embedding layer must be of dtype torch.long, Currently dtype is {indices.dtype}"
        embeddings = self.embedding_layer(indices) * math.sqrt(self.emb_size)
        return self.positional_encoding(embeddings)

class Transformer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, nhead: int, num_encoder_layers: int, num_decoder_layers: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.embedding_layer = EmbeddingLayer(vocab_size, d_model, dropout)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=dropout)
        self.classifier = nn.Linear(d_model, vocab_size)

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def forward(self, src: torch.Tensor, tar: torch.Tensor):
        T, _ = tar.shape

        tar_mask: torch.Tensor = self.generate_square_subsequent_mask(T).to(tar.device)

        src = self.embedding_layer(src)
        tar = self.embedding_layer(tar)

        transformer_out: torch.Tensor = self.transformer(src, tar, tgt_mask=tar_mask)

        classifier_out: torch.Tensor = self.classifier(transformer_out)
        return classifier_out
