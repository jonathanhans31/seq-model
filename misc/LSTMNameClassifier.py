import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class Model(nn.Module):
    """docstring for Model"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim

        ## The LSTM is going to take in input from the one hot encoding
        self.lstm = nn.LSTM(input_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, output_dim) 
        self.hidden = self.init_hidden()
    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))
    def forward(self, x):
        N_sequence = len(x)

        inputs = x.view(N_sequence, 1, -1)

        # Extract the feature in the sequence into lstm_out
        lstm_out, self.hidden = self.lstm(inputs, self.hidden)

        # Add a Linear Layer 
        tag_space = self.hidden2tag(lstm_out.view(N_sequence, -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
