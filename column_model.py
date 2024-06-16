import torch
import torch.nn as nn
# import nn.transformer encoder
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import numpy as np
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(1000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)
        self.register_buffer('pe', self.encoding)

    def forward(self, x):
        pe = self.pe[:, :x.size(1)].to(x.device)
        return x + pe




class ColumnModel(nn.Module):
    def __init__(self, max_question_length = 100, max_columns = 64):
        super(ColumnModel, self).__init__()
        self.max_question_length = max_question_length
        self.max_columns = max_columns
        self.embedding_dim = 300
        self.positional_encoding = PositionalEncoding(self.embedding_dim)
        self.layer = TransformerEncoderLayer(d_model=self.embedding_dim, nhead=2, batch_first=True)
        self.multihead_attn = TransformerEncoder(self.layer, num_layers=1)
        self.fc = nn.Linear(self.embedding_dim, 1)

    def forward(self, question_embedding , columns_embedding):
        question_embedding = self.positional_encoding(question_embedding)
        separator = torch.zeros((question_embedding.shape[0], 1, self.embedding_dim)).to(question_embedding.device)
        concatenated = torch.cat((question_embedding, separator, columns_embedding), dim=1).to(question_embedding.device)
        # pass the concatenated embeddings through the transformer encoder
        out = self.multihead_attn(concatenated)
        # print('out:', out.shape)
        columns_output = out[:, 101:, :]
        # print('columns_output:', columns_output.shape)
        columns_output = self.fc(columns_output)
        columns_output = columns_output.squeeze(2)
        # print('columns_output:', columns_output.shape)
        return columns_output
