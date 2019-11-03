import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

# 读取特定时间片下的网络快照的函数
# 数据文件名称前缀+时间片索引+动态网络中节点总数-->指定时间片的邻接矩阵
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
    return curAdj

# 初始化权重函数
# 权重矩阵的行数+权重矩阵的列数=初始化后的权重矩阵
def var_init(m, n):
    init_range = np.sqrt(6.0 / (m+n))
    initial = tf.random_uniform([m, n], minval=-init_range, maxval=init_range, dtype=tf.float64)
    return tf.Variable(initial)

# 计算某个网络快照GCN因子的函数
# 特定网络快照的邻接矩阵-->对应的GCN因子
def get_gcn_fact(adj):
    adj_ = adj + np.eye(node_num, node_num)
    row_sum = np.array(adj_.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.mat(np.diag(d_inv_sqrt))
    gcn_fact = d_mat_inv_sqrt*adj_*d_mat_inv_sqrt 

    return gcn_fact

# 计算多个网络快照经GCN和LSTM的输出
# 节点特征矩阵/单位矩阵+多个快照的GCN因子-->gcn_lstm模型输出及LSTM参数
def gcn_lstm(gcn_fact_phs):
    gcn_outputs = [] 
    for i in range(window_size+1):
        feature = node_feature
        gcn_fact = gcn_fact_phs[i]
        gcn_wei = gcn_weis[i]

        gcn_conv = tf.matmul(gcn_fact, feature)
        gcn_output = tf.sigmoid(tf.matmul(gcn_conv, gcn_wei))   # shape=(38, 1)
        gcn_output = tf.reshape(gcn_output, [1, node_num*hid_num0])
        gcn_outputs.append(gcn_output)
    LSTM_cells = rnn.MultiRNNCell([rnn.BasicLSTMCell(node_num*hid_num0)])
    with tf.variable_scope("gcn_lstm") as gcn_lstm:
        LSTM_outputs, states = rnn.static_rnn(LSTM_cells, gcn_outputs, dtype=tf.float64)
        LSTM_params = [var for var in tf.global_variables() if var.name.startswith(gcn_lstm.name)]
    
    output = tf.nn.sigmoid(tf.matmul(LSTM_outputs[-1], output_wei) + output_bias)

    return output, LSTM_params

def accuracy():

    return 

name_pre = "top-500-author-idx/top-500-author-idx"
node_num = 500
time_num = 11 # 时间片总数
window_size = 8   # 考虑的历史网络快照的窗口大小
epoch_num = 1000

gcn_weis = []
hid_num0 = 1
node_feature = np.eye(node_num, node_num)
for i in range(window_size+1):
    gcn_weis.append(tf.Variable(var_init(node_num, hid_num0)))
output_wei = tf.Variable(var_init(node_num*hid_num0, node_num*node_num), dtype=tf.float64)
output_bias = tf.Variable(tf.zeros(shape=[node_num*node_num], dtype=tf.float64), dtype=tf.float64)
output_params = [output_wei, output_bias]

# 定义TF占位符
gcn_fact_phs = [] # GCN因子的占位符列表
for i in range(window_size+1):
    gcn_fact_phs.append(tf.placeholder(tf.float64, shape=[node_num, node_num]))
gnd_ph = tf.placeholder(tf.float64, shape=(1, node_num*node_num)) # Placeholder of the ground-truth 标准答案的占位符

# 构建GCN-LSTM
output, LSTM_params = gcn_lstm(gcn_fact_phs)

# 定义训练过程的损失函数
loss = tf.reduce_sum(tf.square(gnd_ph - output))

# 定义训练的优化器
opt = tf.train.RMSPropOptimizer(learning_rate=0.005).minimize(loss, var_list=(gcn_weis+LSTM_params+output_params))

# 运行神经网络
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for t in range(window_size, time_num-2):
    gcn_facts = []
    for k in range(t-window_size, t+1):
        adj = read_data(name_pre, (k+2010), node_num)
        if (k+2010) == (t+2010):
            print('current network snapshot #%d'%(k+2010))


        gcn_fact = get_gcn_fact(adj)
        gcn_facts.append(gcn_fact)
    
    gnd = np.reshape(read_data(name_pre, (t+1+2010), node_num), (1, node_num*node_num))
    print('next network snapshot #%d'%((t+1+2010)))

    loss_list = []
    for epoch in range(epoch_num):
        ph_dict = dict(zip(gcn_fact_phs, gcn_facts))
        ph_dict.update({gnd_ph: gnd})
        _, g_loss, g_output = sess.run([opt, loss, output], feed_dict=ph_dict)
        loss_list.append(g_loss)
        if epoch%100==0:
            print('Train #%d, Loss: %f'%(epoch, g_loss))
        if epoch>500 and loss_list[epoch]>loss_list[epoch-1] and loss_list[epoch-1]>loss_list[epoch-2]:
            break
