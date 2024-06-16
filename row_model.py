import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
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


class RowModel(nn.Module):
    def __init__(self, d_model=300):
        super(RowModel, self).__init__()
        self.max_question_length = 100
        self.max_columns = 64
        self.d_model = d_model
        self.positional_encoding = PositionalEncoding(d_model)
        self.layer = TransformerEncoderLayer(d_model=self.d_model, nhead=2, batch_first=True)
        self.multihead_attn = TransformerEncoder(self.layer, num_layers=1)
        self.fc = nn.Linear((self.max_question_length + 3*self.max_columns + 1)*self.d_model, 1)


    def forward(self, question_embedding, rows_embedding):
        """
            question_embedding -> (batch_size, max_question_length, d_model)
            rows_embedding -> (batch_size, (1 + number_of_negative_samples), max_columns, d_model)

            input to transformer encoder:
            question1 [sep] row[1]1
            question1 [sep] row[1]2
            question1 [sep] row[1]3
            ............
            question1 [sep] row[1](1+neg_samples)
            question2 [sep] row[2]1
            .....

            question(batch_size) [sep] row[batch_size](1+neg_samples)

            where [sep] is a separator token
            returns:
            rows_output -> (batch_size, (1 + number_of_negative_samples))

        """
        question_embedding = self.positional_encoding(question_embedding)
        separator = torch.zeros((question_embedding.shape[0],rows_embedding.shape[1], 1, self.d_model)).to(question_embedding.device)
        question_embedding = question_embedding.unsqueeze(1)
        question_embedding = question_embedding.repeat(1, rows_embedding.shape[1], 1, 1)
        concatenated = torch.cat((question_embedding, separator, rows_embedding), dim=2).to(question_embedding.device)
        concatenated = concatenated.view(-1, concatenated.shape[2], concatenated.shape[3]) # [batch_size * (1 + number_of_negative_samples), max_columns, d_model]
        out = self.multihead_attn(concatenated)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        out = torch.sigmoid(out)
        out = out.view(-1)

        return out




