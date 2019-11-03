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

def t_p_dis(out, gnd):
    tp, tn, fp, fn = 0, 0, 0, 0
    total_num = gnd.shape[1]
    for i in range(gnd.shape[0]):
        idx_gnd = torch.nonzero(gnd[i]) #?
        non_zero_num = idx_gnd.shape[0]
    # 每一个点对应的向量有多少个非零元素
        idx_out = out[i].topk(non_zero_num, 0, True, True)[1]   #?
        idx_gnd = idx_gnd.squeeze()
        tp_i = 0
        for j in idx_out:
            if j in idx_gnd:
                tp_i += 1
        tp += tp_i
        fp += (non_zero_num - tp_i)
        fn += (non_zero_num - tp_i)
        tn += total_num - tp_i - 2*(non_zero_num - tp_i)
    return tp, tn, fp, fn