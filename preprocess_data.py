import json
import joblib
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import sys

# load the data
train_path = sys.argv[1]
val_path = sys.argv[2]
def load_data(path):
    with open(path) as f:
        data = f.read().strip().split('\n')
    data = [json.loads(d) for d in data]
    return data

# preprocess the data, convert to lowercase, etc.
def preprocess(data):
    for d in data:
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
        label_col = d['label_col']
        label_row = d['label_row']
        for i in range(len(label_col)):
            label_col[i] = label_col[i].lower()
            label_col[i] = word_tokenize(label_col[i])
        d['label_col'] = label_col
        d['label_row'] = label_row
        types = table['types']
        for i in range(len(types)):
            types[i] = types[i].lower()
            types[i] = word_tokenize(types[i]) # each type is a list of tokens
        table['types'] = types
        caption = table['caption']
        caption = caption.lower()
        caption = word_tokenize(caption) # caption is a list of tokens
        table['caption'] = caption

    return data



print('Preprocessing data... in preprocess_data.py')
train_data = load_data(train_path)
val_data = load_data(val_path)
data = train_data + val_data
data = preprocess(data)
joblib.dump(data, 'preprocessed_data.pkl')
print('Preprocessed data saved in preprocessed_data.pkl')

