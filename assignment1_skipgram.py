# -*- coding: utf-8 -*-
"""nlp_ass_pytorch.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hQ0d9YZdTqp9FYYDm4MaPYE3kj7pKtsS

Importing all necessary libraries
"""

import nltk
from nltk.corpus import gutenberg
nltk.download('gutenberg')
nltk.download('punkt')
import regex as re
from nltk.corpus import stopwords
nltk.download('stopwords')
import string
import torch
torch.manual_seed(10)
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn import decomposition
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (10,8)
import torch.nn.functional as F

"""Loading web scraped data"""

import json
with open(r'/content/drive/MyDrive/final_sentences (1).json') as f:
  final_sentences = json.load(f)

"""Preprocessing data (Stop word removal,punctuation removal,lower case)"""

def preprocessing(corpus):
    stop_words = set(stopwords.words('english'))   
    training_data = []
    for sentence in corpus:
        x = []
        for word in sentence:
            word = word.strip(string.punctuation)
            word = word.strip()
            word = re.sub(r"(http[s]?\://\S+)|([\[\(].*[\)\]])|([#@]\S+)|\n|([0-9])|(\,+)|(\'+)|(\"+)|([^\w\s])", '', word.lower())
           
            if word.lower() not in stop_words and word != '':
                word = word.lower()
                x.append(word)
        training_data.append(x)
    print(training_data)
    return training_data

"""Converting data to matrix for training"""

def prepare_data_for_training(sentences):
    data = {}
    for sentence in sentences:
        for word in sentence:
            if word not in data:
                data[word] = 1
            else:
                data[word] += 1
    V = len(data)
    data = sorted(list(data.keys()))
    vocab = {}
    for i in range(len(data)):
        vocab[data[i]] = i
      
    N = sum([len(sentence) for sentence in sentences])
    X_train = torch.zeros((N, V), dtype=torch.float)
    y_train = torch.zeros((N, V), dtype=torch.float)
    k = 0
    for sentence in sentences:
        for i in range(len(sentence)):
            X_train[k, vocab[sentence[i]]] = 1
            for j in range(i-2,i+2):
                if i!=j and j>=0 and j<len(sentence):
                    y_train[k, vocab[sentence[j]]] += 1
            k += 1
    
    
    
    return X_train,y_train,V,data

"""Calling preprocessing function"""

training = preprocessing(final_sentences[0:200])

"""Calling prepare data for training function"""

X_train,Y_train,V,data = prepare_data_for_training(training)

from google.colab import drive
drive.mount('/content/drive')

"""Defining all variable needed for training"""

vocab_size = V
embedding_dims = 5
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
initrange = 0.5 / embedding_dims
num_epochs = 1
learning_rate = 0.001
lr_decay = 0.99

"""Initializing our weight matrix"""

W1 = Variable(torch.randn(vocab_size, embedding_dims, device=device).uniform_(-initrange, initrange).float()).to(device) # shape V*H
W2 = Variable(torch.randn(embedding_dims, vocab_size, device=device).uniform_(-initrange, initrange).float()).to(device)

"""Defining forward,backward propgation function and training function and performing training"""

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
for epo in range(num_epochs):
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    def forward(X):
        h = torch.matmul(W1.T,X).reshape(embedding_dims, 1)
        u = torch.matmul(W2.T,h)
        y = F.softmax(u, dim=0)
        return y,h,u

    def backpropagate(X, t,h,u,p):
        e = t - torch.tensor(p).reshape(len(t),1)
        dLdW1 = torch.matmul(e,h.T)
        dLdW2 = (torch.matmul(X.reshape(vocab_size,1), torch.matmul(W1.T, e).T)).T
        W1.data = W1.data - learning_rate * dLdW1
        W2.data = W2.data - learning_rate * dLdW2

    for j in range(len(X_train)):
        y_result,h,u = forward(X_train[j])
        p = Y_train[j]
        backpropagate(X_train[j], y_result,h,u,p)
        C = torch.tensor(0).to(device)
        loss = torch.zeros((1,1)).to(device)
        #for m in range(len(Y_train[j])):
            #if Y_train[j][m]:
                #loss += -1 * u[m][0]
                #C += 1
        u = u.clamp(min=1e-6)

        loss += torch.log(torch.sum(torch.exp(u)))
    
    if epo%10 == 0:
        learning_rate *= lr_decay
  
    if epo%2 == 0:
        print(f'Epoch {epo}, loss = {loss}')
    if epo == num_epochs - 1:
        output = {}
        for i in range(vocab_size):
            word = data[i]
            n = [0 for j in range(vocab_size)]
            n[i] = 1
            n = torch.tensor(n,dtype=torch.float)
            
            W1 = W1.cpu()
            prediction = torch.matmul(W1.T,n.T).reshape(1,embedding_dims)
            prediction_numpy = prediction.detach().numpy()
            output[word] = prediction_numpy
        with open(r'/content/output.json','w') as f:
            json.dump(output, f, cls=NumpyEncoder)

"""Defining function to check accuracy on validation data"""

words_list = []
from sklearn.metrics.pairwise import cosine_similarity

with open('/content/drive/MyDrive/analogy1.txt', 'r') as f:
    for line in f:
        line = line.strip() 
        words = line.split(' ') 
        words_list.append(words)



words_list[0]
def predict():
    with open(r'/content/output.json') as f:
        output = json.load(f)
    correct = 0
    for sentence in words_list:
        
        word1 = sentence[0]
        word2 = sentence[1]
        word3 = sentence[2]
        word4 = sentence[3]
        if word1 in data and word2 in data and word3 in data:
        
            
        
            cos_sim = {}
            final_output = (np.array(output[word1]).reshape(embedding_dims,1) - np.array(output[word2]).reshape(embedding_dims,1) + np.array(output[word3]).reshape(embedding_dims,1)).reshape(1,embedding_dims) 
            for i in range(vocab_size):
                word = data[i]
                sim = cosine_similarity(final_output,np.array(output[word]))
                cos_sim[word] = sim
              
            max_keys = [key for key, value in cos_sim.items() if value == max(cos_sim.values())]
            
            max_value = max_keys[0]
            print(max_value)
            print(word4)
            if str(max_value) == str(word4):
                correct = correct + 1
            else:
                continue
        else:
            continue
    return correct/len(sentence)
predict()



