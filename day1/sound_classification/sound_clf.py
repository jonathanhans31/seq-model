
# coding: utf-8

# In[1]:

import torch
import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os
import pandas as pd
import glob
import numpy as np

dataset_path = "D:/datasets/sound_classification/"
ipd.Audio(dataset_path+"Train/2022.wav")


# In[2]:


data, sampling_rate = librosa.load(dataset_path+"/Train/2022.wav")


# In[3]:


plt.figure(figsize=(12, 4))
librosa.display.waveplot(data, sampling_rate)


# In[4]:


### Steps to complete
# 1. Load audio files
# 2. Extract features from audio
# 3. Convert the data to pass it in our deep learning model
# 4. Run a deep learning model and get results


# In[5]:


# Look into train.csv and test.csv
df = pd.read_csv(dataset_path+"train.csv")
df.head()


# In[6]:


def parser(row):
    # function to load files and extract features
    file_name = os.path.join(os.path.abspath(dataset_path), 'Train', str(row.ID) + '.wav')

    # handle exception to check if there isn't a file which is corrupted
    try:
        # here kaiser_fast is a technique used for faster extraction
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        # we extract mfcc feature from data
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
    except Exception as e:
        print("Error encountered while parsing file: ", file)
        return None, None
 
    feature = mfccs
    label = row.Class

    return [feature, label]
# train_df = df[:100].copy() # A subset of the data for testing the code only
train_df = df.copy()


# In[7]:


# train_df = train_df.apply(parser, axis=1)


# In[8]:


X = [item[0] for item in train_df]
Y = [item[1] for item in train_df]


# In[15]:


X = np.load("Features.npy")
Y = np.load("Labels.npy")


# In[16]:


# from sklearn.preprocessing import LabelEncoder

# lb = LabelEncoder()
# def to_categorical(data):
#     H = data.max()
#     one_hots = np.eye(H)
#     return [one_hots[i-1] for i in data]
# Y = to_categorical(lb.fit_transform(Y))


# In[17]:


from model import Model
input_dim = 40 # Number of mfcc feature
hidden_dim = 1024
output_dim = len(Y[0]) # Number of class
print(input_dim, hidden_dim, output_dim)

net = Model(input_dim, hidden_dim, output_dim)
net.cuda()


# In[18]:


np.save("Features.npy", np.array(X))
np.save("Labels.npy", np.array(Y))


# In[19]:


def get_data():
    idx = np.random.randint(0, len(X))
    feature = X[idx]
    label = Y[idx]
    var_feature = torch.tensor(feature.astype("float32")).cuda()
    var_label = torch.tensor(np.argmax(label), dtype=torch.long).cuda()
    return var_feature.unsqueeze(0), var_label.unsqueeze(0), feature, np.argmax(label)
var_feature, var_label ,feature ,label = get_data()


# In[20]:


import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F

optimizer = optim.SGD(net.parameters(), lr = 0.0005)
criterion = nn.CrossEntropyLoss()


# In[21]:


n_steps = 500000
log_step = 1000
N_samples = 1000
samples = []
for i in range(N_samples):
    samples.append(get_data())
def compute_accuracy(net):
    correct = 0
    for i in range(N_samples):
        var_feature, var_label ,feature ,label = samples[i]
        output = net(var_feature)
        correct += int(output.cpu().data.numpy().squeeze().argmax() == label)
    return correct / N_samples

for step in range(n_steps):
    var_feature, var_label ,feature ,label = get_data()
    output = net(var_feature)

    loss = criterion(output, var_label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % log_step == 0:
        print("Step", step, "Accuracy", compute_accuracy(net),"Loss", loss.cpu().data.numpy().squeeze(),"Current batch",int(output.cpu().data.numpy().squeeze().argmax() == label))
        

