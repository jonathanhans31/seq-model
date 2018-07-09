# Dependencies
from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch
import torch.nn as nn

# Global variable declarations
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Utils functions

# Get all files with pattern
def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1,n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line),1,len(all_letters))
    for i, letter in enumerate(line):
        tensor[i][0][letterToIndex(letter)] = 1
    return tensor

# END UTILS


# Building the categories and names 
category_lines = {}
all_categories = []
for filename in findFiles("data/names/*.txt"):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines
n_categories = len(all_categories)


### MODEL ###
class Model(nn.Module):
    """docstring for Model"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.i2h = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.i2o = nn.Linear(input_dim + hidden_dim, output_dim)
    def forward(self, input, hidden):
        combined = torch.cat([input, hidden], dim=1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        return output, hidden
    def init_hidden():
        # this function is critical to reset the weights for each sequence
        return torch.zeros(1, self.hidden_dim)
### END MODEL ###

n_hidden = 128
net = Model(n_letters, n_hidden, n_categories)

### Test ###
inp = lineToTensor('Albert')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = net(inp[0], hidden)
print(output.size())

top_n, top_i = output.topk(1)
category_i = top_i[0].item()
print(top_n)
print(top_i)
print(top_i[0].item())
    # return all_categories[category_i], category_i

### End Test ###


        