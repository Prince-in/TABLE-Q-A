from torch.utils.data import DataLoader, Dataset
import numpy as np
from nltk.tokenize import word_tokenize
import nltk

import random
import joblib
# use word2vec model
from gensim.models import word2vec
import sys
from torch.nn.utils.rnn import pad_sequence

class RowDataset(Dataset):
    def __init__(self, data , word2vec_model , negative_samples = 2, max_question_length = 100, max_columns = 64):
        self.data = data
        self.word2vec_model = word2vec_model
        self.negative_samples = negative_samples
        self.word2vec_dim = word2vec_model.vector_size
        self.max_question_length = max_question_length
        self.max_columns = max_columns

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        d = self.data[idx]
        question = d['question']
        table = d['table']
        rows = table['rows']
        cols = table['cols']
        types = table['types']
        label_row = d['label_row']
        question_embedding = self.get_question_embedding(question, self.word2vec_model)
        pad = word_tokenize('pad')
        types = types + pad * (self.max_columns - len(types))
        types_embedding = []
        for t in types:
            t_embedding = self.get_embedding(t, self.word2vec_model)
            types_embedding.append(t_embedding)
        types_embedding = np.array(types_embedding)
        cols_embedding = [] 
        cols = cols + pad * (self.max_columns - len(cols))
        for c in cols:
            c_embedding = self.get_embedding(c, self.word2vec_model)
            cols_embedding.append(c_embedding)
        cols_embedding = np.array(cols_embedding)
        # get the embeddings for the rows
        # pick a random row from the label_row
        random_row = random.choice(label_row)
        # get the embeddings for the rows
        row_embedding = [self.get_row_embedding(rows[random_row], types_embedding , cols_embedding, self.word2vec_model)]
        number_of_negative_samples = self.negative_samples
        if len(label_row) == len(rows):
            r2 = np.array(row_embedding)
            r2 = np.repeat(r2, number_of_negative_samples + 1, axis=0)
            r3 = np.array([1])
            r3 = np.repeat(r3, number_of_negative_samples + 1, axis=0)
            return question_embedding, r2, r3
        cnt = 0
        row_indices_not_in_label_row = [i for i in range(len(rows)) if i not in label_row]
        while cnt < number_of_negative_samples:
            random_row = random.choice(row_indices_not_in_label_row)
            row_embedding.append(self.get_row_embedding(rows[random_row], types_embedding, cols_embedding, self.word2vec_model))
            cnt += 1
        row_embedding = np.array(row_embedding)
        labels = [1] + [0]*number_of_negative_samples
        labels = np.array(labels)
        return question_embedding, row_embedding, labels

    def get_question_embedding(self, tokens, model):
        pad = word_tokenize('pad')
        question = tokens + pad * (self.max_question_length - len(tokens))
        embeddings = []
        for token in question:
            if token in model:
                embeddings.append(model[token])
            else:
                embeddings.append(np.zeros(self.word2vec_dim))
        return np.array(embeddings)

    def get_embedding(self, tokens, model):
        embeddings = []
        for token in tokens:
            if token in model:
                embeddings.append(model[token])
            else:
                embeddings.append(np.zeros(self.word2vec_dim))
        return np.mean(embeddings, axis=0)
    def get_row_embedding(self, row, types_embedding, cols_embedding, model):
        pad = word_tokenize('pad')
        row = row + pad * (self.max_columns - len(row))
        row_embedding = []
        for cell in row:
            cell_embedding = self.get_embedding(cell, model)
            row_embedding.append(cell_embedding)
        return np.concatenate([row_embedding, types_embedding, cols_embedding], axis=0)



class ColumnDataset(Dataset):
    def __init__(self, data, word2vec_model, max_question_length = 100, max_columns = 64):
        self.data = data
        self.word2vec_model = word2vec_model
        self.vector_size = word2vec_model.vector_size
        self.max_question_length = max_question_length
        self.max_columns = max_columns
        self.word2vec_dim = word2vec_model.vector_size


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question'] # list of tokens 
        table = item['table'] 
        cols = table['cols'] # list of list of tokens
        label_col = item['label_col'] # list of list of tokens
        # see if the question is asking for a column

        # target = index of the column in the cols list
        # target = cols.index(label_col[0])
        target = np.zeros(self.max_columns)
        target[cols.index(label_col[0])] = 1

        question_embedding = self.get_question_embedding(question, self.word2vec_model)
        column_embeddings = self.get_column_embeddings(cols)
        return question_embedding, column_embeddings, target
    
    def get_question_embedding(self, tokens, model):
        pad = word_tokenize('pad')
        question = tokens + pad * (self.max_question_length - len(tokens))
        embeddings = []
        for token in question:
            if token in model:
                embeddings.append(model[token])
            else:
                embeddings.append(np.zeros(self.word2vec_dim))
        return np.array(embeddings)
    
    def avg_embedding(self, words):
        embeddings = []
        for word in words:
            if word in self.word2vec_model:
                embeddings.append(self.word2vec_model[word])
            else:
                embeddings.append(np.zeros(self.vector_size))
        return np.max(embeddings, axis=0)

    def get_column_embeddings(self, columns):
        pad = word_tokenize('pad')
        columns = columns + pad * (self.max_columns - len(columns))
        embeddings = []
        for col in columns:
            embeddings.append(self.avg_embedding(col))
        return np.array(embeddings)
    