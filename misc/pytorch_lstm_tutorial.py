import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from LSTMTagger import *
# torch.manual_seed(1)


# lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3

## First Way
# inputs = [torch.randn(1, 3) for _ in range(5)] 	# Make a sequence of length 5

# hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3)) # Initialize hidden state
# for i in inputs:
# 	inp = i.view(1,1,-1)
# 	print("Input size",inp.size())
# 	out, hidden = lstm(inp, hidden)
# 	print("Output size", out.size())

### Input.size() => (A,B,C)
## A = Number of Sequence
## B = Number of Batch
## C = Number of Feature

# inputs = [torch.randn(1, 3)] * 5 	# Make a sequence of length 5
# inputs = torch.cat(inputs).view(len(inputs), 1, -1)
# hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
# out, hidden = lstm(inputs, hidden)

# print("Sizes", inputs.size(), out.size())
# print(out.shape)
# print(hidden)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

#def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
model = Model(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss() # Negative Log Likelyhood Loss
optimizer = optim.SGD(model.parameters(), lr=0.1)

fixed_sample = prepare_sequence(training_data[0][0], word_to_ix)
def show_sample(model):
    with torch.no_grad():
        tag_scores = model(fixed_sample) 
        print(tag_scores) # Print out the tag scores for each word 

show_sample(model)
# Define the number of epochs to train the model
num_epochs = 10000
n_data = len(training_data)
#model.train()
for e in range(num_epochs):
    for step, (sentence, tags) in enumerate(training_data):
        # Get the data ready to be fed into the network
        data = prepare_sequence(sentence, word_to_ix)
        label_tag = prepare_sequence(tags, tag_to_ix)
        
        # Do a forward pass on the model
        pred_tag = model(data)
        
        # Compute for the loss        
        loss = loss_function(pred_tag, label_tag) # Compute for the Negative Log Likelyhood between the predicted and the label tag

        model.zero_grad()   # Zero out the gradients from previous forward pass
        loss.backward()     # Do a backward pass on the loss
        optimizer.step()    # Update
        
        model.hidden = model.init_hidden() # Refresh the hidden state for each sample
        
        
        log = "Epoch {}, Step {}, Loss {}".format(e, step, loss.data.numpy().squeeze())
        print(log)
        
# After training the model show the scores
show_sample(model) # Show how the scores 
    

        
        
        
        
        

    


