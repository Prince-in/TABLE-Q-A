import torch
import torch.optim as optim
import torch.nn as nn
from row_model import RowModel
from data_loader import RowDataset
from torch.utils.data import DataLoader
import joblib
import os
import gensim.downloader as api
import time



print("Entered in Train_row.py")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)

# load the data
data = joblib.load('preprocessed_data.pkl')
print('Data length:', len(data))

# load word2vec model
print('Loading pretrained word2vec model...')
word2vec = api.load("word2vec-google-news-300")
print("Word2vec Loaded")

# create the dataset
print('Creating dataset...')
dataset = RowDataset(data, word2vec)

# create the dataloader
print('Creating dataloader...')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
print('Dataloader length:', len(dataloader))

print('Creating the model...')
model = RowModel().to(device)
print('Model:', model)
# create the model

# task classification , softmax used in last layer in model

# create the loss function
print('Creating the loss function...')
criterion = nn.BCELoss() # binary cross entropy loss

print('Creating the optimizer...')
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# if model exists, load it
if os.path.exists('models/row_model.pth'):
    print('Loading the pretrained row model...')
    model.load_state_dict(torch.load('models/row_model.pth'))
    print('Model Loaded')


print('Training the model...')
for epoch in range(70):  # loop over the dataset multiple times
    running_loss = 0.0
    correct = 0
    total = 0
    st = time.time()
    for i, batch in enumerate(dataloader):
        print(i)
        batch = [b.to(device).float() for b in batch]
        question, rows, labels = batch
        optimizer.zero_grad()
        outputs = model(question, rows)# [batch_size * (1 + number_of_negative_samples)]
        labels = labels.view(-1).float()
        loss = criterion(outputs, labels)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.shape[0]
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('Epoch:', epoch, 'Loss:', running_loss / len(dataloader), 'Time:', (time.time() - st) / 60 , 'minutes')
    print('Accuracy:', 100 * correct / total)
    print('Saving the model...')
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), 'models/row_model.pth')
    print(f"Model saved {epoch}")
    
    