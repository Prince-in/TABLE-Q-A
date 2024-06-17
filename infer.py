import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
from nltk.tokenize import word_tokenize
from row_model import RowModel
from column_model import ColumnModel
import warnings
from copy import deepcopy
import gensim.downloader as api
import sys



# Ignore all warnings
warnings.filterwarnings("ignore")



# load the models
row_model= RowModel()
col_model = ColumnModel()

col_model.load_state_dict(torch.load('models/column_model.pth', map_location=torch.device('cpu')))
row_model.load_state_dict(torch.load('models/row_model.pth', map_location=torch.device('cpu')))


# like predict on val_file and write predictions
# path = '/Users/bhaveshgurnani/Desktop/COL772-NLP/assignments/COL772-2302-A2/data/A2_sample_test_input.jsonl'
# path is a command line argument given to the script
path = sys.argv[1]
pred_path = sys.argv[2]

# load the data
def load_data(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def infer_preprocess(data):
    # create a copy of data
    data_copy = deepcopy(data)
    for d in data_copy:
        question = d['question']
        question = question.lower()
        question = word_tokenize(question)
        d['question'] = question # d['question'] is a list of tokens
        table = d['table']
        rows = table['rows']
        cols = table['cols']
        for r in rows:
            for i in range(len(r)):
                r[i] = r[i].lower()
                r[i] = word_tokenize(r[i]) # each cell is a list of tokens
        for i in range(len(cols)):
            cols[i] = cols[i].lower()
            cols[i] = word_tokenize(cols[i]) # each column is a list of tokens
        d['table'] = table
        types = table['types']
        for i in range(len(types)):
            types[i] = types[i].lower()
            types[i] = word_tokenize(types[i]) # each type is a list of tokens
        table['types'] = types
        caption = table['caption']
        caption = caption.lower()
        caption = word_tokenize(caption) # caption is a list of tokens
        table['caption'] = caption
    return data_copy



class inferDataset(Dataset):
    def __init__(self, data , word2vec_model , negative_samples = 2, max_question_length = 100, max_columns = 64):
        self.data = data
        self.word2vec_model = word2vec_model
        self.negative_samples = negative_samples
        self.word2vec_dim = word2vec_model.vector_size
        self.max_question_length = max_question_length
        self.max_columns = max_columns
        self.vector_size = word2vec_model.vector_size

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        d = self.data[idx]
        question = d['question']
        table = d['table']
        rows = table['rows']
        cols = table['cols']
        types = table['types']
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
        row_embedding = []
        for i in range(len(rows)):
            row_embedding.append(self.get_row_embedding(rows[i], types_embedding , cols_embedding, self.word2vec_model))
        row_embedding = np.array(row_embedding)
        column_embeddings = self.get_column_embeddings(cols)
        return question_embedding, row_embedding, column_embeddings

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



data = load_data(path)
preprocess_data = infer_preprocess(data)
print('Loading pretrained word2vec model...')
word2vec = api.load("word2vec-google-news-300")
print('Word2vec Loaded')


dataset = inferDataset(preprocess_data, word2vec)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


predictions = []

device = torch.device("cpu")
print('Device:', device)
row_model.eval()
col_model.eval()
for i, batch in enumerate(dataloader):
    batch = [b.to(device).float() for b in batch]
    question, rows, columns = batch
    pad_vector = columns[0, 63, :]
    row_output = row_model(question, rows)
    col_output = col_model(question, columns)
    mask = torch.all(columns == pad_vector, dim=2)
    mask = mask.to(device)
    outputs = col_output.masked_fill(mask, -1) 
    col_idx = torch.argmax(outputs, dim=1)
    # row_indices = (row_output > 0.35).float()
    col_idx = torch.randint(0, len(data[i]['table']['cols']), (1,)) if col_idx >= len(data[i]['table']['cols']) else col_idx
    max_val = torch.max(row_output)
    row_indices = (row_output >= 0.9*max_val).float()
    label_col = data[i]['table']['cols'][col_idx]
    label_row = []
    for j in range(len(row_indices)):
        if row_indices[j] == 1:
            label_row.append(j)
    label_cell = []
    for r in label_row:
        label_cell.append([r, label_col])
    predictions.append({'label_col': [label_col], 'label_cell': label_cell, 'label_row': label_row, 'qid': data[i]['qid']})

print("Writing Predictions to file")
with open(pred_path, 'w') as f:
    for p in predictions:
        f.write(json.dumps(p))
        f.write('\n') 
