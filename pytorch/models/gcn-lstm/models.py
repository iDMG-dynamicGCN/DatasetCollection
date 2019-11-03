import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight) #?
        output = torch.spmm(adj, support) #?
        return output + self.bias


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        # self.dropout = dropout

    def forward(self, x, adj):
        x = torch.sigmoid(self.gc1(x, adj)) # ?x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return torch.sigmoid(x)   #?

class LSTM(nn.Module):   #layers=2?  和时序展开区分开
    def __init__(self, lstm_in, lstm_hidden, lstm_out, layers):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=lstm_in, hidden_size=lstm_hidden, num_layers=layers)
        self.linear = nn.Linear(lstm_hidden, lstm_out)

    def forward(self, x):
        out, hidden = self.lstm(x)   #?x怎么对应GCN   out hidden
        last_hidden_out = out[-1]   #?last_hidden_out = out[:, -1, :].squeeze()
        return self.linear(last_hidden_out)