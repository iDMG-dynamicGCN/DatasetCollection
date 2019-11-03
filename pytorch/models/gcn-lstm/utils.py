import numpy as np
import scipy.sparse as sp
import torch

def normalize(mx):  #?DAD因子
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def read_data(name_pre, time_index, node_num):
    print('Read network snapshot #%d'%(time_index))
    curAdj = np.mat(np.zeros((node_num, node_num)))

    f = open('%s-%d.txt'%(name_pre, time_index))
    line = f.readline()
    while line:
        seq = line.split()
        src = int(seq[0]) 
        tar = int(seq[1]) 
        curAdj[src, tar] = 1
        curAdj[tar, src] = 1
        line = f.readline()
    
    f.close()
    curAdj_1 = curAdj + np.eye(node_num, node_num)
    curAdj_2 = normalize(curAdj_1)
    curAdj = torch.FloatTensor(curAdj)
    curAdj_3 = torch.FloatTensor(curAdj_2)
    return curAdj, curAdj_3
