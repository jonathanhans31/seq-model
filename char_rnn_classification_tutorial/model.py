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
        self.main = nn.LSTM(input_dim, hidden_dim)      # The LSTM is to deal with the sequence and learn the features
        self.clf = nn.Linear(hidden_dim, output_dim)    # Classify the name based on the feature extracted from the text
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim).cuda(),
                torch.zeros(1, 1, self.hidden_dim).cuda())
    def forward(self, x):
        N_sequence = len(x)
        x = x.view(N_sequence, 1, -1) # Make sure the input data format is correct

        # Extract the feature in the sequence into lstm_out
        # Feed the sentence and the initial
        lstm_out, self.hidden = self.main(x, self.hidden)

        # Return the scores
        return F.log_softmax(self.clf(lstm_out.view(N_sequence, -1)), dim=1)

        