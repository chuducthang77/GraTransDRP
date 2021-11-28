import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# GCN-CNN based model

class AE_NN(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=1024, dropout=0.2):

        super(AE_NN, self).__init__()

        # cell line feature
        self.encoder_1 = nn.Linear(17737, 1024)
        self.batch_norm1 = nn.BatchNorm1d(num_features=1024)
        self.encoder_3 = nn.Linear(1024,756)
        self.batch_norm2 = nn.BatchNorm1d(num_features=756)

        self.decoder_3 = nn.Linear(756,1024)
        self.batch_norm3 = nn.BatchNorm1d(num_features=1024)
        self.decoder_1 = nn.Linear(1024, 17737)
        self.batch_norm4 = nn.BatchNorm1d(num_features=17737)
        
        
        # activation and regularization
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):

        # protein input feed-forward:

        encode = self.encoder_1(data)
        encode = self.relu(self.batch_norm1(encode))
        encode = self.encoder_3(encode)
        encode = self.sigmoid(self.batch_norm2(encode))


        decode = self.decoder_3(encode)
        decode = self.relu(self.batch_norm3(decode))
        decode = self.decoder_1(decode)
        decode = self.sigmoid(self.batch_norm4(decode))

        # 1d conv layers

        return encode, decode
