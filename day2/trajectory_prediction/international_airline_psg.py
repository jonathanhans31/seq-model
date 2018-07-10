import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.optim as optim
import torch.nn as nn
from model import *
from tqdm import tqdm
def to_numpy(x):
    return x.cpu().data.numpy().squeeze()
# Load the dataset and try to visualize the data
dataset = pd.read_csv("international-airline-passengers.csv",usecols=[1],skipfooter=3).values.astype("float32")
print("Dataset shape",dataset.shape)


# Steps Checklist
# 1. Load and preprocess the model

# 1A Scale the dataset
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)

# 1B Split to training and testing
N = dataset.shape[0]
train = dataset[:int(N*0.6)]
test = dataset[int(N*0.6):]
print("Train and test", train.shape, test.shape)
train_x = train[:-2]
train_y = train[1:]
test_x = test[:-2]
test_y = test[1:]

# 2. Batch feeding
WINDOW_SIZE = 7

# A function to feed the batch during training
def get_batch(x, y, window_size, idx = -1):
    N = len(x)
    if idx < 0:
        idx = np.random.randint(0, N-window_size)
    _x = np.expand_dims(x[idx:idx+window_size],-1)
    _y = y[idx:idx+window_size]
    var_x = torch.from_numpy(_x)
    var_y = torch.from_numpy(_y)
    return var_x.cuda(), var_y.cuda(), _x, _y

var_x, var_y, x, y = get_batch(train_x, train_y, WINDOW_SIZE,0)
print("\nSample Data")
print("==============")
for i in range(WINDOW_SIZE):
    print(x[i].squeeze(), y[i].squeeze())


# 3. Build the model (LSTM)
n_input = 1
n_hidden = 64
n_output = 1

net = Model(n_input, n_hidden, n_output)
net.cuda()
num_params = 0
for p in net.parameters():
    num_params += p.numel()
print(net)
print(num_params)
net.hidden = net.init_hidden()
# Test inference
output = net(var_x)
print("\nInference Test")
print("==============")
print("Pred-Size", output.size())
print("Label-Size", var_y.size())
experiment_name = "1 Layer LSTM 64 hidden"
def train():
    # 4. Set optimizer and loss function
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    n_steps = 10000
    n_train = len(train_x)-WINDOW_SIZE
    log_step = 1000
    losses = []
    
    print("Begin training", experiment_name)
    for _, step in enumerate(tqdm(range(n_steps))):
        
        var_x, var_y, x, y = get_batch(train_x, train_y, WINDOW_SIZE, step% n_train)

        net.hidden = net.init_hidden()
        output = net(var_x)
        # loss = loss_fn(output[-1], var_y[-1])
        loss = loss_fn(output, var_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_step == 0:
            val_loss = loss.cpu().data.numpy().squeeze()
            losses.append(val_loss)
            print("\nStep",step,"Loss",val_loss)
            print("Pred",list(to_numpy(output)))
            print("Label",list(to_numpy(var_y)))

    print("Done Training")
    plt.plot(losses)
    plt.show()
    print("Saving models...")
    torch.save(net.state_dict(), experiment_name+".pth")

def test():
    net.load_state_dict(torch.load(experiment_name+".pth"))
    

    pred = []
    n_test = len(test_x)
    for step in range(n_test-WINDOW_SIZE):
        var_x, var_y, x, y = get_batch(test_x, test_y, WINDOW_SIZE, step)
        net.hidden = net.init_hidden()
        output = net(var_x)
        tmp = output.cpu().data.numpy().squeeze()
        if step == 0 :
            pred += list(tmp)
        else:
            pred.append(tmp[-1])
    
    pred = np.reshape(pred,[-1,1])
    # pred = scaler.inverse_transform(pred)
    # plot_y = scaler.inverse_transform(test_y)
    plt.plot(pred)
    plt.plot(test_y,"-")
    plt.show()

    pred = []
    n_test = len(train_x)
    for step in range(n_test-WINDOW_SIZE):
        var_x, var_y, x, y = get_batch(train_x, train_y, WINDOW_SIZE, step)
        net.hidden = net.init_hidden()
        output = net(var_x)
        tmp = output.cpu().data.numpy().squeeze()
        if step == 0 :
            pred += list(tmp)
        else:
            pred.append(tmp[-1])
    
    pred = np.reshape(pred,[-1,1])
    # pred = scaler.inverse_transform(pred)
    # plot_y = scaler.inverse_transform(train_y)
    plt.plot(pred)
    plt.plot(train_y,"-")
    plt.show()
train()
test()

