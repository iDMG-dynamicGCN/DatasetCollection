import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import read_data
from models import GCN, LSTM

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--name-pre', type=str, default='top-500-author-idx/top-500-author-idx',
                    help='Prefix name of the data file.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--node-num', type=int, default=500,
                    help='Number of nodes.')
parser.add_argument('--window-size', type=int, default=5,
                    help='Window size of the history network snapshot to be considered.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--gcn-hidden', type=int, default=16,
                    help='Number of hidden units of GCN.')   #?
parser.add_argument('--gcn-output', type=int, default=16,
                    help='Number of output of GCN.')   #?
parser.add_argument('--lstm-layers', type=int, default=2)   
parser.add_argument('--lstm-hidden', type=int, default=32)   #?
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Load training data
gcn_input_adjs = []
print('load trainging data--------------\n')
for k in range(0, args.window_size+1):
        adj = read_data(args.name_pre, (k+2010), args.node_num)
        gcn_input_adjs.append(adj)
        if (k+2010) == (args.window_size+2010):
            print('current network snapshot #%d'%(k+2010))
gnd = read_data(args.name_pre, (args.window_size+1+2010), args.node_num)
print('next network snapshot #%d'%((args.window_size+1+2010)))

# Model and optimizer
features = np.eye(args.node_num, args.node_num)
features = torch.FloatTensor(features)
gcn_net = GCN(nfeat=features.shape[1],
            nhid=args.gcn_hidden,
            nout=args.gcn_output)    #?out  针对每一个节点而言  先不要dropout
lstm_net = LSTM(lstm_in=args.gcn_output, lstm_hidden=args.lstm_hidden, lstm_out=args.node_num, layers=args.lstm_layers)
parameters = list(gcn_net.parameters()) + list(lstm_net.parameters())
optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
criterion = torch.nn.MSELoss()  #?

def train(epoch):
    gcn_net.train()
    lstm_net.train()
    optimizer.zero_grad()
    sequence = torch.stack([gcn_net(features, gcn_input_adjs[i]) for i in range(args.window_size+1)], 0)#?dim应该确定为多少 点序列 or 图序列   [6,500]  [500,500]
    out = lstm_net(sequence)   #?维数与gnd不匹配  .size()
    loss_train = criterion(out, gnd)
    loss_train.backward()
    optimizer.step()

    if epoch%100==0:
            print('Train #%d, Loss: %f'%(epoch, loss_train.item()))

def test():
    gcn_net.eval()
    lstm_net.eval()
    sequence = torch.stack([gcn_net(features, gcn_input_adjs[i]) for i in range(1, args.window_size+2)], 0)#?dim应该确定为多少 点序列 or 图序列
    out = lstm_net(sequence)
    loss_test = criterion(out, gnd) 
    print('Test, Loss: %f'%(loss_test.item()))

# Train model
for epoch in range(args.epochs):
    train(epoch)

# Testing
gcn_input_adjs.append(gnd)
print('load testing data--------------\n')
print('current network snapshot #%d'%(args.window_size+1+2010))
gnd = read_data(args.name_pre, (args.window_size+2+2010), args.node_num)
print('next network snapshot #%d'%((args.window_size+2+2010)))
test()