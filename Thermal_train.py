#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import pearsonr
#from dataset import Dataset, collate_fn
SEED = 42
from scipy.stats import pearsonr




Atomic_wild=np.load('Wild_Atomic_Coordinate.npy',allow_pickle=True)
Atomic=np.load('Mutant_Atomic_Coordinate.npy',allow_pickle=True)
sequence_features_wild=np.load('Wild_Node.npy',allow_pickle=True)
sequence_features=np.load('Mutant_Node.npy',allow_pickle=True)
sequence_graphs_wild=np.load('Wild_Edge_Contact.npy',allow_pickle=True)
sequence_graphs=np.load('Mutant_Edge_Contact.npy',allow_pickle=True)
EdgeDW=np.load('Wild_Edge_Distance.npy',allow_pickle=True)
EdgeDM=np.load('Mutant_Edge_Distance.npy',allow_pickle=True)
Dihedral = np.load('Dih.npy',allow_pickle=True)
Dihedral_wild = np.load('Dih_w.npy',allow_pickle=True)
labels=np.load('dgg.npy',allow_pickle=True)
sequence_names=np.load('Seq_Name.npy',allow_pickle=True)


# In[4]:


Model_Path = './Model/'
Result_Path = './Result/'


# In[5]:


Dihedral_wild = np.array(Dihedral_wild)
Dihedral = np.array(Dihedral)
Atomic_wild = np.array(Atomic_wild)
Atomic = np.array(Atomic)
sequence_features =np.array(sequence_features)
sequence_features_wild =np.array(sequence_features_wild)


# In[6]:


Atomic_wild.astype(np.float64)
Dihedral.astype(np.float64)
Dihedral_wild.astype(np.float64)
Atomic.astype(np.float64)
sequence_features.astype(np.float64)
sequence_features_wild.astype(np.float64)


# In[7]:


A = []
Aw = []

# Iterate over the range
for i in range(len(Dihedral)):
    #print(f"Shapes before concatenation: Atomic[{i}]: {Atomic[i].shape}, Dihedral[{i}]: {Dihedral[i].shape}")
    # Concatenate and append to A
    A.append(np.concatenate(( Atomic[i],Dihedral[i]), axis=1))
    
    #print(f"Shape after concatenation: A[{i}]: {A[i].shape}")

    # Repeat the process for Aw
    #print(f"Shapes before concatenation: Atomic_wild[{i}]: {Atomic_wild[i].shape}, Dihedral_wild[{i}]: {Dihedral_wild[i].shape}")
    Aw.append(np.concatenate(( Atomic_wild[i],Dihedral_wild[i]), axis=1))
    
    #print(f"Shape after concatenation: Aw[{i}]: {Aw[i].shape}")


# In[8]:


import pandas as pd
zipped = list(zip(sequence_names,labels))
ds = pd.DataFrame(zipped, columns=['names','stability'])


# In[9]:


df=pd.read_csv('intro.csv')


# In[10]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# fit using the train set
#scaler.fit(M)
# transform the test test
#X_scaled = scaler.transform(a)


# In[11]:


WG = dict(zip(sequence_names, sequence_graphs_wild))
MG = dict(zip(sequence_names, sequence_graphs))
WF=dict(zip(sequence_names, sequence_features_wild))
MF=dict(zip(sequence_names, sequence_features))
WS=dict(zip(sequence_names, Aw))
MS=dict(zip(sequence_names, A))
zipped = list(zip(sequence_names, labels))
ds = pd.DataFrame(zipped, columns=['names', 'stability'])


# In[12]:


def NodeM(name):
    return MF[name]
def NodeW(name):
    return WF[name]
def GraphM(name):
    return MG[name]
def GraphW(name):
    return WG[name]
def StM(name):
    return MS[name]
def StW(name):
    return WS[name]
def DM(name):
    return MD[name]
def DW(name):
    return WD[name]


# In[13]:


names = ds['names'].values.tolist()


# In[29]:


from torch.utils.data.sampler import Sampler
  
class Dataset(Dataset):

    def __init__(self, dataframe):
        self.names = dataframe['names'].values.tolist()
        self.labels = dataframe['stability'].values.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        
        sequence_name = self.names[index]
        label = self.labels[index]
        
        sequence_feature = NodeM(sequence_name)
        scaler.fit(sequence_feature)
        sequence_feature = scaler.transform(sequence_feature)
        # L * L
        sequence_graph = GraphM(sequence_name)
        
        A= StM(sequence_name)
        scaler.fit(A)
        A = scaler.transform(A)

        # L * 91
        sequence_feature_wild = NodeW(sequence_name)
        scaler.fit(sequence_feature)
        # L * L
        sequence_feature_wild = scaler.transform(sequence_feature_wild)

        sequence_graph_wild =  GraphW(sequence_name)
        
        Aw=  StW(sequence_name)
        scaler.fit(Aw)
        Aw = scaler.transform(Aw)

        sample = {'sequence_feature': sequence_feature,\
                  'sequence_feature_wild': sequence_feature_wild,\
                  'sequence_graph': sequence_graph, \
                  'sequence_graph_wild': sequence_graph_wild, \
                  'A': A,\
                  'Aw': Aw,\
                  'label': label, \
                  'sequence_name': sequence_name, \
                  }
        return sample



def collate_fn(batch):
    max_natoms_m = 11
    max_natoms_w = 11
    sequence_feature = np.zeros((len(batch), max_natoms_m, 71))
    sequence_feature_wild = np.zeros((len(batch), max_natoms_w, 71))
    sequence_graph = np.zeros((len(batch), max_natoms_m, max_natoms_m))
    sequence_graph_wild = np.zeros((len(batch), max_natoms_w, max_natoms_w))
    A = np.zeros((len(batch), max_natoms_m, 43))
    Aw = np.zeros((len(batch), max_natoms_w,43))

    sequence_names = [] 
    labels=[]   
    for i in range(len(batch)):
        natom1 = len(batch[i]['sequence_feature'])
        natom2 = len(batch[i]['sequence_feature_wild'])
        natom3 = len(batch[i]['A'])
        natom4 = len(batch[i]['Aw'])
        sequence_feature[i,:natom1] = batch[i]['sequence_feature']
        sequence_feature_wild[i,:natom2] = batch[i]['sequence_feature_wild']
        sequence_graph[i,:natom1,:natom1] = batch[i]['sequence_graph']
        sequence_graph_wild[i,:natom2,:natom2] = batch[i]['sequence_graph_wild']
        A[i,:natom1,:43] = batch[i]['A']
        Aw[i,:natom2,:43] = batch[i]['Aw']
        sequence_names.append(batch[i]['sequence_name'])
        labels.append(batch[i]['label'])
        labels= np.asarray(labels)
    sequence_feature= torch.from_numpy(sequence_feature).float()
    sequence_feature_wild = torch.from_numpy(sequence_feature_wild).float()
    sequence_graph = torch.from_numpy(sequence_graph).float()
    sequence_graph_wild = torch.from_numpy(sequence_graph_wild).float()
    A = torch.from_numpy(A).float()
    Aw = torch.from_numpy(Aw).float()
    labels= torch.from_numpy(labels).float()

    return sequence_feature, sequence_feature_wild,sequence_graph , sequence_graph_wild,A,Aw, labels, sequence_names


# In[30]:


# Model parameters
NUMBER_EPOCHS = 1000
LEARNING_RATE = 5E-4
WEIGHT_DECAY = 1E-7
BATCH_SIZE = 8
NUM_CLASSES = 1

# GCN parameters
GCN_FEATURE_DIM = 71
GCN_HIDDEN_DIM1 = 256
GCN_HIDDEN_DIM2 = 128
# Increased hidden dimension
GCN_OUTPUT_DIM = 64
DROPOUT_RATE = 0.5 
GCN_HIDDEN_DIM = 256
# Attention parameters
DENSE_DIM = 16
ATTENTION_HEADS = 4
GAT_FEATURE_DIM = 71
GAT_HIDDEN_DIM = 256
GAT_OUTPUT_DIM = 64
NUM_HEADS = 4
DROPOUT_RATE = 0.6


# In[31]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.5):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.batch_norm = nn.BatchNorm1d(out_features)

    def forward(self, x, adj):
        x = self.linear(x)
        x = torch.matmul(adj, x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

class ComplexGraphNeuralNetwork(nn.Module):
    def __init__(self, input_features, hidden_features, output_features, num_layers=2, dropout_rate=0.5):
        super(ComplexGraphNeuralNetwork, self).__init__()

        self.conv1 = GraphConvolution(input_features, hidden_features, dropout_rate)
        self.convs = nn.ModuleList([GraphConvolution(hidden_features, hidden_features, dropout_rate) for _ in range(num_layers - 1)])
        self.conv_final = GraphConvolution(hidden_features, output_features, dropout_rate)

        self.fc1 = nn.Linear(hidden_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, 11 * output_features)  # Adjusted for the desired output size
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, adj):
        x = self.conv1(x, adj)
        for conv in self.convs:
            x = conv(x, adj)
        x = self.conv_final(x, adj)

        # Global pooling (you can use other pooling strategies based on your task)
        x = torch.mean(x, dim=0)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        # Reshape to the desired output size (11 x output_features)
        x = x.view(11, -1)

        return x

# Example usage
# Assuming you have node features (x) and adjacency matrix (adj)
input_features = 71
hidden_features = 64
output_features = 64


# In[32]:


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = input @ self.weight    # X * W
        output = adj @ support           # A * X * W
        if self.bias is not None:        # A * X * W + b
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# In[33]:


class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(GCN_FEATURE_DIM, GCN_HIDDEN_DIM1)
        self.ln1 = nn.LayerNorm(GCN_HIDDEN_DIM1)
        self.gc2 = GraphConvolution(GCN_HIDDEN_DIM1, GCN_HIDDEN_DIM2)  # Additional hidden layer
        self.ln2 = nn.LayerNorm(GCN_HIDDEN_DIM2)
        self.gc3 = GraphConvolution(GCN_HIDDEN_DIM2, GCN_OUTPUT_DIM)
        self.ln3 = nn.LayerNorm(GCN_OUTPUT_DIM)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, adj):  
        x = self.gc1(x, adj)
        x = self.relu1(self.ln1(x))
        x = self.gc2(x, adj)  # Additional hidden layer
        x = self.relu2(self.ln2(x))
        x = self.gc3(x, adj)
        output = self.relu3(self.ln3(x))
        return output


# In[44]:


class GCN1(nn.Module):
    def __init__(self):
        super(GCN1, self).__init__()
        self.gc1 = GraphConvolution(43, GCN_HIDDEN_DIM)
        self.ln1 = nn.LayerNorm(GCN_HIDDEN_DIM)
        self.gc2 = GraphConvolution(GCN_HIDDEN_DIM, GCN_OUTPUT_DIM)
        self.ln2 = nn.LayerNorm(GCN_OUTPUT_DIM)
        self.relu1 = nn.LeakyReLU(0.2,inplace=True)
        self.relu2 = nn.LeakyReLU(0.2,inplace=True)

    def forward(self, x, adj):  			# x.shape = (seq_len, GCN_FEATURE_DIM); adj.shape = (seq_len, seq_len)
        x = self.gc1(x, adj)  				# x.shape = (seq_len, GCN_HIDDEN_DIM)
        x = self.relu1(self.ln1(x))
        x = self.gc2(x, adj)
        output = self.relu2(self.ln2(x))	# output.shape = (seq_len, GCN_OUTPUT_DIM)
        return output


# In[45]:


class Attention(nn.Module):
    def __init__(self, input_dim, dense_dim, n_heads):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.dense_dim = dense_dim
        self.n_heads = n_heads
        self.fc1 = nn.Linear(self.input_dim, self.dense_dim)
        self.fc2 = nn.Linear(self.dense_dim, self.n_heads)

    def softmax(self, input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = torch.softmax(input_2d, dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def forward(self, input):  				# input.shape = (1, seq_len, input_dim)
        x = torch.tanh(self.fc1(input))  	# x.shape = (1, seq_len, dense_dim)
        x = self.fc2(x)  					# x.shape = (1, seq_len, attention_hops)
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)  		# attention.shape = (1, attention_hops, seq_len)
        return attention


# In[46]:


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()

        # Calculate the number of input channels for the 1x1 convolution dynamically
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.in_channels = in_channels if stride == 1 else out_channels

        self.conv1 = nn.Conv1d(self.in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Adjust input channels dynamically based on the output channels of the previous layer
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1x1(x)  # 1x1 convolution to adjust input channels
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(identity)
        out = self.relu(out)
        return out


# In[47]:


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.a = nn.Parameter(torch.FloatTensor(2 * out_features, 1))

        nn.init.xavier_uniform_(self.W.data)
        nn.init.xavier_uniform_(self.a.data)

    def forward(self, x, adj):
        h = torch.matmul(x, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(2), negative_slope=self.alpha)
        
        # Apply mask to the adjacency matrix
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        return F.elu(h_prime)

class ModelS(nn.Module):
    def __init__(self):
        super(ModelS, self).__init__()
        self.att = GraphAttentionLayer(71, 64, dropout=0.2)
        self.out_att = GraphAttentionLayer(64, 64, dropout=0.2)
        self.attention = Attention(64, DENSE_DIM, ATTENTION_HEADS)

        #self.fc_final = nn.Linear(hidden_size, num_classes)
        self.fc_final = nn.Linear(72, NUM_CLASSES)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    def forward(self, x_W, adj_W, x_M, adj_M, A_W, A_M):
        A_W = A_W.float()
        A_M = A_M.float()
        x_W = x_W.float()
        x_M = x_M.float()


        #for att in self.attentions:
        x_M = F.elu(self.att(x_M, adj_M))
        x_W = F.elu(self.att(x_W, adj_W))

        x_M = F.elu(self.out_att(x_M, adj_M))
        x_W = F.elu(self.out_att(x_W, adj_W)) 
        A_M=torch.cat((x_M,A_M),1)
        A_W=torch.cat((x_W,A_W),1)
        A_W = A_W.unsqueeze(0).float() 
        A_M = A_M.unsqueeze(0).float()
        tot = A_M-A_W
        #print(tot.shape)
       # att1 = self.attention(A_M)  # att.shape = (1, ATTENTION_HEADS, seq_len)
       # node_feature_embedding_avg = torch.bmm(att1, A_M.transpose(1, 2)).mean(dim=1)
        #att2 = self.attention(A_W)  # att.shape = (1, ATTENTION_HEADS, seq_len)
        #node_feature_embedding_avg1 = torch.bmm(att2, A_W.transpose(1, 2)).mean(dim=1)
        #node_feature_embedding_avg2 = node_feature_embedding_avg - node_feature_embedding_avg1
        output = torch.sigmoid(self.fc_final(tot.mean(dim=1)))  # output.shape = (1, NUM_CLASSES)
        return output.squeeze(0)


# In[48]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSAGELayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphSAGELayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.W.data)

    def forward(self, x, adj):
        h = torch.matmul(x, self.W)
        N = h.size(0)

        aggregate_neighborhood = torch.matmul(adj, h)
        h_concat = torch.cat([h, aggregate_neighborhood], dim=1)

        return F.elu(h_concat)
    
class CNNLayer(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CNNLayer, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(output_channels, output_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(output_channels, output_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(output_channels, output_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(output_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return x

class ModelS(nn.Module):
    def __init__(self):
        super(ModelS, self).__init__()
        self.sage1 = GraphSAGELayer(114, 32)
        self.sage2 = GraphSAGELayer(64 , 64)
        self.resnet_layer = ResNetBlock(128, 128)  # Adjust num_blocks as needed
        self.fc_final = nn.Linear(128, 1)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def forward(self, x_W, adj_W, x_M, adj_M, A_W, A_M):
        A_W = A_W.float()
        A_M = A_M.float()
        x_W = A_W.float()
        x_M = A_M.float()
        A_M = F.elu(self.sage1(A_M, adj_M))
        A_W = F.elu(self.sage1(A_W, adj_W))

        A_M = F.elu(self.sage2(A_M, adj_M))
        A_W = F.elu(self.sage2(A_W, adj_W))

        A_W = A_W.unsqueeze(0).float()
        A_M = A_M.unsqueeze(0).float()
        tot = A_M - A_W
        tot = tot.permute(0, 2, 1)  # Reshape for 1D convolution
        tot = self.resnet_layer(tot)

        # Pooling along the sequence dimension
        tot = F.avg_pool1d(tot, kernel_size=tot.size(2))

        # Reshape for fully connected layer
        tot = tot.view(tot.size(0), -1)
        output = torch.sigmoid(self.fc_final(tot))  # output.shape = (1, NUM_CLASSES)
        return output.squeeze(0)


# In[49]:


import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
       # self.GNN = ComplexGraphNeuralNetwork(input_features, hidden_features, output_features, num_layers=3)
        self.gcn1 = GCN1()
        self.gcn = GCN()
       # self.resnet = ResBlock(GCN_OUTPUT_DIM,64,64)
        self.attention = Attention(64, DENSE_DIM, ATTENTION_HEADS)
        self.fc_final = nn.Linear(64, NUM_CLASSES)

        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def forward(self, x_W, adj_W,x_M,adj_M,A_W,A_M):  
        x_W = x_W.float()
        A_W = A_W.float()
        #x_W=torch.cat((x_W,A_W),1)
       # print(x_W.shape)
        x_W = self.gcn(x_W, adj_W)
        A_W = self.gcn1(A_W,adj_W)
       # print(x_W.shape)# x.shape = (seq_len, GAT_OUTPUT_DIM)
        x_M = x_M.float()
        A_M = A_M.float()
        #x_M=torch.cat((x_M,A_M),1)        
        x_M = self.gcn(x_M, adj_M)  
        A_M = self.gcn1(A_M,adj_M)
        #A_M = self.gcn1(A_M,adj_M)
       # print(A_M.shape,x_M.shape)
        #x_M=torch.cat((x_M,A_M),1)
        #x_W=torch.cat((x_W,A_W),1)
        x_W = x_W + A_W
        x_M = x_M + A_M
        #print(x_M.shape)
        #x_M = x_M.sum(1)
        #x_W = x_W.sum(1)
       # print(x_M.shape)
        #tot = x_W-x_M
        x_W = x_W.unsqueeze(0).float() 
        x_M = x_M.unsqueeze(0).float()
        #print(tot.shape)
        #tot = x_W-x_M
        #print(tot.shape)
        att1 = self.attention(x_M) # att.shape = (1, ATTENTION_HEADS, seq_len)
        #print(att.shape)
        #print(tot.shape)
       # assert tot.size(2) == att.size(2),
        node_feature_embedding =  att1 @ x_M 
        node_feature_embedding_avg = torch.sum(node_feature_embedding,1) / self.attention.n_heads 
        att2 = self.attention(x_W) # att.shape = (1, ATTENTION_HEADS, seq_len)
        #print(att.shape)
        #print(tot.shape)
       # assert tot.size(2) == att.size(2),
        node_feature_embedding1 =  att2 @ x_W 
        node_feature_embedding_avg1 = torch.sum(node_feature_embedding1,
                                               1) / self.attention.n_heads 
        node_feature_embedding_avg2 = node_feature_embedding_avg1 - node_feature_embedding_avg
        output =  torch.sigmoid(self.fc_final(node_feature_embedding_avg2))  	# output.shape = (1, NUM_CLASSES)
        return output.squeeze(0)


# In[50]:


def train_one_epoch(model, data_loader, epoch):

    epoch_loss_train = 0.0
    n_batches = 0
    for data in tqdm(data_loader):
        model.optimizer.zero_grad()
        sequence_feature, sequence_feature_wild,sequence_graph , sequence_graph_wild,A,Aw, labels, sequence_names = data

        sequence_feature = torch.squeeze(sequence_feature)
        sequence_graph = torch.squeeze(sequence_graph)
        A = torch.squeeze(A)
        sequence_feature_wild = torch.squeeze(sequence_feature_wild)
        sequence_graph_wild = torch.squeeze(sequence_graph_wild)
        Aw = torch.squeeze(Aw)
        if torch.cuda.is_available():
            features = Variable(sequence_feature.cuda())
            graphs = Variable(sequence_graph.cuda())
            features_wild = Variable(sequence_feature_wild.cuda())
            graphs_wild = Variable(sequence_graph_wild.cuda())
            A = Variable(A.cuda())
            Aw = Variable(Aw.cuda())
            y_true = Variable(labels.cuda())
        else:
            features = Variable(sequence_feature)
            graphs = Variable(sequence_graph)
            features_wild = Variable(sequence_feature_wild)
            graphs_wild = Variable(sequence_graph_wild)
            A = Variable(A)
            Aw = Variable(Aw)
            y_true = Variable(labels)

        y_pred = model(features_wild, graphs_wild,features, graphs,Aw,A)
        y_true = y_true.float()

        # calculate loss
        loss = model.criterion(y_pred, y_true)
        #l2_lambda = 0.001
        #l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        #loss = loss + l2_lambda * l2_norm
        #print(loss)

        # backward gradient
        loss.backward()

        # update all parameters
        model.optimizer.step()

        epoch_loss_train += loss.item()
        n_batches += 1

    epoch_loss_train_avg = epoch_loss_train / n_batches
    return epoch_loss_train_avg


# In[51]:


def evaluate(model, data_loader):
    model.eval()

    epoch_loss = 0.0
    n_batches = 0
    valid_pred = []
    valid_true = []
    valid_name = []

    for data in tqdm(data_loader):
        with torch.no_grad():
            sequence_feature, sequence_feature_wild,sequence_graph , sequence_graph_wild,A,Aw, labels, sequence_names = data

            sequence_feature = torch.squeeze(sequence_feature)
            sequence_graph = torch.squeeze(sequence_graph)
            A= torch.squeeze(A)
            sequence_feature_wild = torch.squeeze(sequence_feature_wild)
            sequence_graph_wild = torch.squeeze(sequence_graph_wild)
            Aw = torch.squeeze(Aw)
            if torch.cuda.is_available():
                features = Variable(sequence_feature.cuda())
                graphs = Variable(sequence_graph.cuda())
                features_wild = Variable(sequence_feature_wild.cuda())
                graphs_wild = Variable(sequence_graph_wild.cuda())
                A = Variable(A.cuda())
                Aw = Variable(Aw.cuda())
                y_true = Variable(labels.cuda())
            else:
                features = Variable(sequence_feature)
                graphs = Variable(sequence_graph)
                features_wild = Variable(sequence_feature_wild)
                graphs_wild = Variable(sequence_graph_wild)
                A = Variable(A)
                Aw = Variable(Aw)
                y_true = Variable(labels)

            y_pred = model(features_wild, graphs_wild,features, graphs,Aw,A)
            y_true = y_true.float()

            loss = model.criterion(y_pred, y_true)
            #l2_lambda = 0.001
            #l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            #loss = loss + l2_lambda * l2_norm
            #print(loss)
                
            y_pred = y_pred.cpu().detach().numpy().tolist()
            y_true = y_true.cpu().detach().numpy().tolist()
            valid_pred.extend(y_pred)
            valid_true.extend(y_true)
            valid_name.extend(sequence_names)

            epoch_loss += loss.item()
            n_batches += 1
    epoch_loss_avg = epoch_loss / n_batches

    return epoch_loss_avg, valid_true, valid_pred, valid_name


# In[52]:


def train(model, dataframe,valid_dataframe,fold=0):
    train_loader = DataLoader(dataset=Dataset(df) ,batch_size=1, shuffle=True, num_workers=0,collate_fn=collate_fn)
    valid_loader = DataLoader(dataset=Dataset(df) ,batch_size=1, shuffle=True, num_workers=0,collate_fn=collate_fn)

    train_losses = []
    train_pearson = []
    train_r2 = []

    valid_losses = []
    valid_pearson = []
    valid_r2 = []

    best_val_loss = 1000
    best_epoch = 0

    for epoch in range(NUMBER_EPOCHS):
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        model.train()

        epoch_loss_train_avg = train_one_epoch(model, train_loader, epoch + 1)
        #print("========== Evaluate Train set ==========")
        #_, train_true, train_pred, _ = evaluate(model, train_loader)
       # print(train_pred)
        #result_train = analysis(train_true, train_pred)
        #print("Train loss: ", np.sqrt(epoch_loss_train_avg))
        #print("Train pearson:", result_train['pearson'])
        #print("Train r2:", result_train['r2'])

        #train_losses.append(np.sqrt(epoch_loss_train_avg))
        #train_pearson.append(result_train['pearson'])
        #train_r2.append(result_train['r2'])

        print("========== Evaluate Valid set ==========")
        epoch_loss_valid_avg, valid_true, valid_pred, valid_name = evaluate(model, valid_loader)
        result_valid = analysis(valid_true, valid_pred)
        print("Valid loss: ", np.sqrt(epoch_loss_valid_avg))
        print("Valid pearson:", result_valid['pearson'])
        print("Valid r2:", result_valid['r2'])


        valid_losses.append(np.sqrt(epoch_loss_valid_avg))
        valid_pearson.append(result_valid['pearson'])
        valid_r2.append(result_valid['r2'])

        if best_val_loss > epoch_loss_valid_avg:
            best_val_loss = epoch_loss_valid_avg
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(Model_Path, 'Fold' + str(fold) + '_best_model.pkl'))
            valid_detail_dataframe = pd.DataFrame({'names': valid_name, 'stability': valid_true, 'prediction': valid_pred})
            valid_detail_dataframe.sort_values(by=['names'], inplace=True)
            valid_detail_dataframe.to_csv(Result_Path + 'Fold' + str(fold) + "_valid_detail.csv", header=True, sep=',')
 
    result_all = {
        'Train_loss': train_losses,
        'Train_pearson': train_pearson,
        'Train_r2': train_r2,
        'Valid_loss': valid_losses,
        'Valid_pearson': valid_pearson,
        'Valid_r2': valid_r2,
        'Best_epoch': [best_epoch for _ in range(len(train_losses))]
    }
    result = pd.DataFrame(result_all)
    print("Fold", str(fold), "Best epoch at", str(best_epoch))
    result.to_csv('result.csv')

def analysis(y_true, y_pred):

    # continous evaluate
    pearson = np.corrcoef(y_true, y_pred)[0,1]
    r2 = metrics.r2_score(y_true, y_pred)


    result = {
        'pearson': pearson,
        'r2': r2
    }
    return result
def cross_validation(all_dataframe,fold_number=10):
    print("split_seed: ", SEED)
    sequence_names = all_dataframe['names'].values
    sequence_labels = all_dataframe['stability'].values
    kfold = KFold(n_splits=fold_number, shuffle=True)
    fold = 0

    for train_index, valid_index in kfold.split(sequence_names, sequence_labels):
        print("\n========== Fold " + str(fold + 1) + " ==========")
        train_dataframe = all_dataframe.iloc[train_index, :]
        valid_dataframe = all_dataframe.iloc[valid_index, :]
        print("Training on", str(train_dataframe.shape[0]), "examples, Validation on", str(valid_dataframe.shape[0]),
              "examples")
        model = Model()
        if torch.cuda.is_available():
            model.cuda()

        train(model, train_dataframe, valid_dataframe, fold + 1)
        fold += 1


# In[53]:



# In[33]:


def test(test_dataframe):
    test_loader = DataLoader(dataset=Dataset(ds) ,batch_size=1, shuffle=True, num_workers=0,collate_fn=collate_fn)
    test_result = {}

    for model_name in sorted(os.listdir(Model_Path)):
        print(model_name)
        model = Model()
        #if torch.cuda.is_available():
        #    model.cuda()
        model.load_state_dict(torch.load(Model_Path + model_name,map_location='cpu'))

        epoch_loss_test_avg, test_true, test_pred, test_name = evaluate(model, test_loader)
        result_test = analysis(test_true, test_pred)
        print("\n========== Evaluate Test set ==========")
        print("Test loss: ", np.sqrt(epoch_loss_test_avg))
        print("Test pearson:", result_test['pearson'])
        print("Test r2:", result_test['r2'])

        test_result[model_name] = [
            np.sqrt(epoch_loss_test_avg),
            result_test['pearson'],
            result_test['r2'],
        ]

        test_detail_dataframe = pd.DataFrame({'names': test_name, 'stability': test_true, 'prediction': test_pred})
        test_detail_dataframe.sort_values(by=['names'], inplace=True)
        test_detail_dataframe.to_csv(Result_Path + model_name + "_test_detail.csv", header=True, sep=',')

    test_result_dataframe = pd.DataFrame.from_dict(test_result, orient='index',
                                                   columns=['loss', 'pearson', 'r2', 'precision'])
    test_result_dataframe.to_csv(Result_Path + "test_result.csv", index=True, header=True, sep=',')


def analysis(y_true, y_pred):

    # continous evaluate
    pearson = pearsonr(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)


    result = {
        'pearson': pearson,
        'r2': r2
    }
    return result

if __name__ == "__main__":
    train_dataframe = df
    cross_validation(train_dataframe,fold_number=5) 
# In[35]:


test(df)


# In[ ]:




