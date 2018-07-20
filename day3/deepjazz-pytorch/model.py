import torch
import torch.nn as nn
from torch.autograd import Variable as Variable
import torch.nn.functional as F
class Model(nn.Module):
    """docstring for Model"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Model, self).__init__()
        self.input_dim = input_dim # Number of feature as input (in this case it's going to be n_letters)
        self.hidden_dim = hidden_dim # Any arbitrary number
        self.output_dim = output_dim # Number of class (in this case n_categories)

        # Use an LSTM cell for the RNN 
        self.n_layer = 2
        self.main = nn.LSTM(input_dim, hidden_dim, num_layers =  self.n_layer, dropout=1)      # The LSTM is to deal with the sequence and learn the features
        self.clf = nn.Linear(hidden_dim, output_dim)    # Classify the name based on the feature extracted from the text
        self.hidden = self.init_hidden()

    def init_hidden(self, use_cuda=False):
        if use_cuda:
            # The axes semantics are (num_layers, minibatch_size, hidden_dim)
            return (torch.zeros( self.n_layer, 1, self.hidden_dim).cuda(),
                    torch.zeros( self.n_layer, 1, self.hidden_dim).cuda())
        return (torch.zeros( self.n_layer, 1, self.hidden_dim),
                    torch.zeros( self.n_layer, 1, self.hidden_dim))
    def forward(self, x):

        # return self.main(x)
        N_sequence = len(x)
        x = x.view(N_sequence, 1, -1) # Make sure the input data format is correct

        # # Extract the feature in the sequence into lstm_out
        # # Feed the sentence and the initial
        lstm_out, self.hidden = self.main(x, self.hidden)

        pred = self.clf(lstm_out.view(N_sequence, -1))
        return pred, lstm_out

class FFDiscriminator(nn.Module):
    """docstring for Model"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFDiscriminator, self).__init__()
        self.input_dim = input_dim # Number of feature as input (in this case it's going to be n_letters)
        self.hidden_dim = hidden_dim # Any arbitrary number
        self.output_dim = output_dim # Number of class (in this case n_categories)

        layers = []
        layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class SeqDiscriminator(nn.Module):
    """docstring for Model"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SeqDiscriminator, self).__init__()
        self.input_dim = input_dim # Number of feature as input (in this case it's going to be n_letters)
        self.hidden_dim = hidden_dim # Any arbitrary number
        self.output_dim = output_dim # Number of class (in this case n_categories)

        # Use an LSTM cell for the RNN 
        self.main = nn.LSTM(input_dim, hidden_dim)      # The LSTM is to deal with the sequence and learn the features
        self.clf = nn.Linear(hidden_dim, output_dim)    # Classify the name based on the feature extracted from the text
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))
    def forward(self, x):
        # return self.main(x)
        N_sequence = len(x)
        x = x.view(N_sequence, 1, -1) # Make sure the input data format is correct

        # # Extract the feature in the sequence into lstm_out
        # # Feed the sentence and the initial
        lstm_out, self.hidden = self.main(x, self.hidden)
        pred = self.clf(lstm_out.view(N_sequence, -1))
        return pred
