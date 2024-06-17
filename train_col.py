import torch
import torch.optim as optim
import torch.nn as nn
from column_model import ColumnModel
from data_loader import ColumnDataset
from torch.utils.data import DataLoader
import joblib
import os
import gensim.downloader as api




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)


# load the data
data = joblib.load('preprocessed_data.pkl')
print('Data length:', len(data))

# load word2vec model
print('Loading pretrained word2vec model...')
# word2vec = api.load('fasttext-wiki-news-subwords-300')
word2vec = api.load('word2vec-google-news-300')
print('Word2vec Loaded')

# create the dataset
print('Creating dataset...')
dataset = ColumnDataset(data, word2vec)

# create the dataloader
print('Creating dataloader...')
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

print('Dataloader length:', len(dataloader))

# create the model
print('Creating the model...')
model = ColumnModel(word2vec).to(device)
print('Model:', model)

# task classification , softmax used in last layer in model

# create the loss function
print('Creating the loss function...')
criterion = nn.CrossEntropyLoss(ignore_index=-1)

print('Creating the optimizer...')
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# if model exists, load it
if os.path.exists('models/column_model.pth'):
    print('Loading the pretrained column model...')
    model.load_state_dict(torch.load('models/column_model.pth'))
    print('Model Loaded')


# train the model
print('Training the model...')
for epoch in range(50):  # loop over the dataset multiple times
    running_loss = 0.0
    correct = 0
    total = 0
    for i, batch in enumerate(dataloader):
        batch = [b.to(device).float() for b in batch]
        question, cols, labels = batch
        pad_vector = cols[0, 63, :]
        optimizer.zero_grad()
        outputs = model(question, cols)
        mask = torch.all(cols == pad_vector, dim=2)
        mask = mask.to(device)
        # apply the mask
        outputs = outputs.masked_fill(mask, -1) # set the masked values to -1
        loss = criterion(outputs, labels)
        # get the accuracy
        preds = torch.argmax(outputs, dim=1)
        labels = torch.argmax(labels, dim=1)
        correct += torch.sum(preds == labels).item()
        total += labels.shape[0]
        loss.backward()
        optimizer.step()
        # if i % 100 == 0:
        #     print('Epoch:', epoch, 'Batch:', i, 'Loss:', loss.item())
        running_loss += loss.item()


    print('Epoch:', epoch, 'Loss:', running_loss / len(dataloader))
    print('Accuracy:', 100 * correct / total)
    # save the model
    print('Saving the model...')
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), 'models/column_model.pth')
    print(f"COLUMN Model Saved {epoch}")
    