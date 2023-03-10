# -*- coding: utf-8 -*-
"""Skipgram_pytorch.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15PDkoJpTy611D59y53KebSFR2-crRCTF
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nltk.corpus import stopwords
import re, argparse

torch.manual_seed(1)
import nltk
nltk.download('stopwords')

parser.add_argument("--data_root", help="Root path (.txt) of the dataset", required=True, type=str)
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
parser.add_argument('--lr', help='Learning rate for the model', default=0.001, type=float)
parser.add_argument('--iters', help='Number of Iterations ', default=100, type=int)
args = parser.parse_args()

checkpoint_dir=args.checkpoint_dir
lr=args.lr
num_epochs=args.iters

data_file=open(args.data_root, 'r')
#data_file = open("/content/drive/MyDrive/CS772 data/final_data (1).txt", 'r')
raw_text = data_file.readline()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# cleaning the dataset
stop_words = set(stopwords.words("english"))
sentences = re.sub('[^A-Za-z0-9]+', ' ', raw_text) # remove special characters
pattern = r'[0-9]'
sentences= re.sub(pattern, '', sentences)
sentences = re.sub(r'(?:^| )\w(?:$| )', ' ', sentences).strip() # remove 1 letter words
sentences = sentences.lower().split() # lower all characters
sentences = [word for word in sentences if word.lower() not in stop_words]
print("Preprocessed the dataset")

vocab = set(sentences)

vocab=list(vocab)

vocab = set(sentences)
vocab_size = len(vocab)

import random

# Define the vocabulary and the corresponding index
vocab = sentences
word2index = {word: i for i, word in enumerate(vocab)}
index2word = {i: word for i, word in enumerate(vocab)}

# Generate the training data
data = []
window_size = 2
for i, word in enumerate(vocab):
    for j in range(i-window_size, i+window_size+1):
        if i != j and j >= 0 and j < len(vocab):
            data.append((word2index[word], word2index[vocab[j]]))

# Shuffle the data
#random.shuffle(data)

# Define the number of epochs
num_epochs = 5

# Define the size of the vocabulary
vocab_size = len(vocab)

data = sorted(data, key=lambda x: x[0])



#checkpoint_dir = "/content/drive/MyDrive/CS772 data"

from os.path import join
def save_checkpoint(model, optimizer, checkpoint_dir, epoch):
    checkpoint_path = join(checkpoint_dir, "checkpoint_step{:09d}.pth".format(epoch))
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

import torch
import torch.nn as nn
import torch.optim as optim

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = self.linear(embeds)
        log_probs = nn.LogSoftmax(dim=1)(out)
        return log_probs

def train(data, model, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        total_loss = 0
        for context, target in data:
            context_var = torch.tensor([context], dtype=torch.long)
            target_var = torch.tensor([target], dtype=torch.long)
            
            model.zero_grad()
            log_probs = model(context_var)
            loss = criterion(log_probs, target_var)

            loss.backward()

            optimizer.step()
        
            total_loss += loss.item()
        print("Epoch %d: Loss = %.4f" % (epoch+1, total_loss))
        if epoch % 3 == 0:
          save_checkpoint(model, optimizer, checkpoint_dir, epoch)
        
# Define the model and optimizer
embedding_dim = 10
model = SkipGramModel(vocab_size, embedding_dim)
optimizer = optim.SGD(model.parameters(), lr)
criterion = nn.NLLLoss()

# Train the model
train(data, model, optimizer, criterion, num_epochs)
word_vectors = model.embeddings.weight.data

