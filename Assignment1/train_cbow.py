# Implementing CBOW model for the exercise given by a tutorial in pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from models import CBOW as CBOW
from os.path import join
import re, argparse
import wandb

torch.manual_seed(1)
device = torch.device("cuda")

parser = argparse.ArgumentParser(description='Code to train the CBOW model for Analogy task')

parser.add_argument("--data_root", help="Root path (.txt) of the dataset", required=True, type=str)
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
parser.add_argument('--lr', help='Learning rate for the model', default=0.001, type=float)
parser.add_argument('--iters', help='Number of Iterations ', default=100, type=int)

args = parser.parse_args()

# loading the data
data_file = open(args.data_root, 'r')
raw_text = data_file.readline()

data = []
context_size = 2 # {w_i-2 ... w_i ... w_i+2}
embedding_dim = 10

# cleaning the dataset
sentences = re.sub('[^A-Za-z0-9]+', ' ', raw_text) # remove special characters
sentences = re.sub(r'(?:^| )\w(?:$| )', ' ', sentences).strip() # remove 1 letter words
sentences = sentences.lower().split() # lower all characters
print("Preprocessed the dataset")


def make_context_vector(context, word_to_idx):
    idxs = [word_to_idx[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

def save_checkpoint(model, optimizer, checkpoint_dir, epoch):
    checkpoint_path = join(checkpoint_dir, "checkpoint_step{:09d}.pth".format(epoch))
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


vocab = set(sentences)
vocab_size = len(vocab)

word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}

# initializing the wandb logs
wandb.init(
    # Set the project where this run will be logged
    project="Cbow training",
    # Track hyperparameters and run metadata
    config={
    "learning_rate": args.lr,
    "Embedding_dim": embedding_dim,
    "Vocab size": vocab_size,
    "architecture": "CBOW (128)",
    "dataset": "Guttenberg + Kushal(final)",
})

# preparing the dataset
for i in range(2, len(sentences) - 2):
    context = [sentences[i-2], sentences[i-1], sentences[i+1], sentences[i+2]]
    target = sentences[i]
    data.append((context, target))
print("Prepared dataset")

# initializing the model and optimizer
model = CBOW(vocab_size, embedding_dim)
optimizer = optim.SGD(model.parameters(), lr=args.lr)
# optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                        #    lr=args.lr)


losses = []
loss_function = nn.NLLLoss()

print("Start Training")
for epoch in range(args.iters):
    total_loss = 0
    for context, target in data:
        context_vector = make_context_vector(context, word_to_idx)
        
        # Remember PyTorch accumulates gradients; zero them out
        model.zero_grad()
        
        nll_prob = model(context_vector)
        # print(type(nll_prob))
        loss = loss_function(nll_prob, Variable(torch.tensor([word_to_idx[target]])))
        
        # backpropagation
        loss.backward()
        # update the parameters
        optimizer.step() 
        
        total_loss += loss.item()
        print(loss.item())
        wandb.log({"Item Loss": loss.item()})

    if epoch % 30 == 0:
        save_checkpoint(model, optimizer, args.checkpoint_dir, epoch)

        
    losses.append(total_loss)
    wandb.log({"Train Loss": total_loss})
    print(total_loss)




#get everything on GPU
#write a batched script