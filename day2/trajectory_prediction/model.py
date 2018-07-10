import torch
import torch.nn as nn
from torch.autograd import Variable as Variable
import torch.nn.functional as F
class Model(nn.Module):
    """docstring for Model"""
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers = 1):
        super(Model, self).__init__()
        self.input_dim = input_dim # Number of feature as input (in this case it's going to be n_letters)
        self.hidden_dim = hidden_dim # Any arbitrary number
        self.output_dim = output_dim # Number of class (in this case n_categories)
        self.n_layers = n_layers
        # # Use an LSTM cell for the RNN 
        # layers.append(nn.LSTM(hidden_dim, output_dim))
        self.main = nn.LSTM(input_dim, hidden_dim, num_layers = self.n_layers)
        layers = []
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim, output_dim))
        # layers.append(nn.ReLU(inplace=True))
        self.regressor = nn.Sequential(*layers)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.n_layers, 1, self.hidden_dim).cuda(),
                torch.zeros(self.n_layers, 1, self.hidden_dim).cuda())
    def forward(self, x):
        N_sequence = len(x)
        output, hidden = self.main(x, self.hidden)
        y = self.regressor(output.view(N_sequence, -1))
        return y        

        