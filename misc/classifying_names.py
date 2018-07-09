import numpy as np
import unicodedata
import string
from utils import *
from text_io import *
from LSTMNameClassifier import Model
import torch
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
def random_choice(x):
    return x[random.randint(0, len(x)-1)]
# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

# Get all names
all_filenames = find_files("./data/names/*.txt")
for filename in all_filenames:
    category = get_basename(filename)
    lines = readlines(filename)

    all_categories.append(category)
    category_lines[category] = lines

n_category = len(all_categories)
n_hidden = 1024
n_iter = 100000
log_step = 1000

model = Model(n_letters, n_hidden, n_category) # The input is going to be the size of the embedding which is 56, the output is the number of category (classification)

optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.NLLLoss()
# Create dataloader .... sort of
def get_line():
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    
    category = torch.tensor([all_categories.index(category)]*len(line), dtype=torch.long)
    line = line_to_tensor(line)
    return category, line

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


# criterion = nn.NLLLoss()
for itr in range(n_iter):
    category, line = get_line()
    pred = model(line)
    loss = criterion(pred, category)

    model.zero_grad()
    loss.backward()
    optimizer.step()

    model.hidden = model.init_hidden()
    if itr % log_step == 0:
        print(itr, "Loss", loss.data.numpy().squeeze())
        # Compute for batch accuracy
        print("Match", categoryFromOutput(pred)[1] == category.data.numpy()[0])
