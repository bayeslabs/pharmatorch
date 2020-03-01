import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


class charRNN(nn.Module):
    """
    This is a basic CharRNN model for smiles
    Contains an encoder and decoder
    """

    @classmethod
    def help(cls):
        print(
            "Intializers needed \n 1) vocabulary of smiles \n 2) dictionarary of config params [hidden_nodes,num_layers,dropout,type_of_RNN] applies same for encoder and decoder")

    def __init__(self,vocabulary,config):
        super(charRNN,self).__init__()
        self.vocabulary = vocabulary
        self.hidden_size = config['hidden_nodes']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.vocab_size = self.input_size = self.output_size = len(vocabulary)
        self.embedding_layer = nn.Embedding(self.vocab_size, self.vocab_size, padding_idx=config["pad"])
        self.linear_layer = nn.Linear(self.hidden_size, self.output_size)
        if config['rnn'] == 'LSTM':
            self.rnn_layer = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout, batch_first=True)
        elif config['rnn'] == 'GRU':
            self.rnn_layer = nn.GRU(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout, batch_first=True)
    def forward(self, x,lengths,hiddens =None):
        x = self.embedding_layer(x)
        x = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True)
        x, hiddens = self.rnn_layer(x, hiddens)
        x, _ = rnn_utils.pad_packed_sequence(x, batch_first=True)
        x = self.linear_layer(x)
        return x, lengths, hiddens
