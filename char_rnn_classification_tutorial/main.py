import numpy as np
import os
import time
import torch
import torch.optim as optim
from utils import *
from model import *
# Dataset preparation
categories = []
category_names = {}

list_files = findFiles("./data/names/*.txt")
for f in list_files:
    category = os.path.basename(f)[:-4] # Get only the filename
    categories.append(category)
    category_names[category] = readLines(f)    

def random_choice(pool):
    return pool[np.random.randint(0, len(pool))]

def get_random_sample():
    category = random_choice(categories)
    name = random_choice(category_names[category])

    # Convert the category and name into tensors
    var_category = torch.tensor([categories.index(category)], dtype=torch.long)
    var_name = lineToTensor(name)

    return var_category.cuda(), var_name.cuda(), category, name

def output_to_category(output):
    idx = torch.argmax(output)
    return categories[idx.cpu().data.numpy()]
# Example of usage "var_category, var_name, category, name = get_random_sample()"

n_hidden = 32
n_category = len(categories)

net = Model(n_letters, n_hidden, n_category)
net.cuda()

optimizer = optim.SGD(net.parameters(), lr=0.005)
loss_fn = nn.NLLLoss()

num_step = 100000
log_step = 1000
for step in range(1, num_step):
    var_category, var_name, category, name = get_random_sample() # Get data
    optimizer.zero_grad()
    net.hidden = net.init_hidden()
    output = net(var_name)
    # print(output[-1].size(), var_category.size())
    loss = loss_fn(output[-1].unsqueeze(0), var_category)
    loss.backward()
    optimizer.step()
    if step % log_step == 0:
        print("Logging",step, "Loss:",loss.cpu().data,"Sample output", name, "---", output_to_category(output[-1]), "|",category)
    
    


# General Components of Training Neural Network
# 1. Batch feeding (DONE)
# 2. Model definition (DONE)
# 3. Optimizer (DONE)
# 4. Loss functions (DONE)
# 5. Sampling
# 6. Saving models
# 7. Logging