import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# GCN-CNN based model

class GAT_GCN_Transformer_meth_ge(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(GAT_GCN_Transformer_meth_ge, self).__init__()

        self.n_output = n_output
        self.encoder_layer_1 = nn.TransformerEncoderLayer(d_model=num_features_xd, nhead=1, dropout=0.5)
        self.ugformer_layer_1 = nn.TransformerEncoder(self.encoder_layer_1, 1)
        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.encoder_layer_2 = nn.TransformerEncoderLayer(d_model=num_features_xd*10, nhead=1, dropout=0.5)
        self.ugformer_layer_2 = nn.TransformerEncoder(self.encoder_layer_2, 1)
        self.conv2 = GCNConv(num_features_xd*10, num_features_xd*10)
        self.fc_g1 = torch.nn.Linear(num_features_xd*10*2, 1500)
        self.fc_g2 = torch.nn.Linear(1500, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # cell line mut feature
        # self.conv_xt_mut_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8)
        # self.pool_xt_mut_1 = nn.MaxPool1d(3)
        # self.conv_xt_mut_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters*2, kernel_size=8)
        # self.pool_xt_mut_2 = nn.MaxPool1d(3)
        # self.conv_xt_mut_3 = nn.Conv1d(in_channels=n_filters*2, out_channels=n_filters*4, kernel_size=8)
        # self.pool_xt_mut_3 = nn.MaxPool1d(3)
        # self.fc1_xt_mut = nn.Linear(2944, output_dim)

        # cell line meth feature
        self.conv_xt_meth_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8)
        self.pool_xt_meth_1 = nn.MaxPool1d(3)
        self.conv_xt_meth_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters*2, kernel_size=8)
        self.pool_xt_meth_2 = nn.MaxPool1d(3)
        self.conv_xt_meth_3 = nn.Conv1d(in_channels=n_filters*2, out_channels=n_filters*4, kernel_size=8)
        self.pool_xt_meth_3 = nn.MaxPool1d(3)
        self.fc1_xt_meth = nn.Linear(1280, output_dim)

        # cell line ge feature
        self.conv_xt_ge_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8)
        self.pool_xt_ge_1 = nn.MaxPool1d(3)
        self.conv_xt_ge_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=8)
        self.pool_xt_ge_2 = nn.MaxPool1d(3)
        self.conv_xt_ge_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=8)
        self.pool_xt_ge_3 = nn.MaxPool1d(3)
        self.fc1_xt_ge = nn.Linear(4224, output_dim)

        # combined layers
        self.fc1 = nn.Linear(3*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.unsqueeze(x, 1)
        x = self.ugformer_layer_1(x)
        x = torch.squeeze(x,1)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = torch.unsqueeze(x, 1)
        x = self.ugformer_layer_2(x)
        x = torch.squeeze(x,1)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)

        # target_mut input feed-forward:
        # target_mut = data.target_mut
        # target_mut = target_mut[:,None,:]
        # conv_xt_mut = self.conv_xt_mut_1(target_mut)
        # conv_xt_mut = F.relu(conv_xt_mut)
        # conv_xt_mut = self.pool_xt_mut_1(conv_xt_mut)
        # conv_xt_mut = self.conv_xt_mut_2(conv_xt_mut)
        # conv_xt_mut = F.relu(conv_xt_mut)
        # conv_xt_mut = self.pool_xt_mut_2(conv_xt_mut)
        # conv_xt_mut = self.conv_xt_mut_3(conv_xt_mut)
        # conv_xt_mut = F.relu(conv_xt_mut)
        # conv_xt_mut = self.pool_xt_mut_3(conv_xt_mut)
        # xt_mut = conv_xt_mut.view(-1, conv_xt_mut.shape[1] * conv_xt_mut.shape[2])
        # xt_mut = self.fc1_xt_mut(xt_mut)

        target_meth = data.target_meth
        target_meth = target_meth[:,None,:]
        conv_xt_meth = self.conv_xt_meth_1(target_meth)
        conv_xt_meth = F.relu(conv_xt_meth)
        conv_xt_meth = self.pool_xt_meth_1(conv_xt_meth)
        conv_xt_meth = self.conv_xt_meth_2(conv_xt_meth)
        conv_xt_meth = F.relu(conv_xt_meth)
        conv_xt_meth = self.pool_xt_meth_2(conv_xt_meth)
        conv_xt_meth = self.conv_xt_meth_3(conv_xt_meth)
        conv_xt_meth = F.relu(conv_xt_meth)
        conv_xt_meth = self.pool_xt_meth_3(conv_xt_meth)
        xt_meth = conv_xt_meth.view(-1, conv_xt_meth.shape[1] * conv_xt_meth.shape[2])
        xt_meth = self.fc1_xt_meth(xt_meth)

        target_ge = data.target_ge
        target_ge = target_ge[:,None,:]
        conv_xt_ge = self.conv_xt_ge_1(target_ge)
        conv_xt_ge = F.relu(conv_xt_ge)
        conv_xt_ge = self.pool_xt_ge_1(conv_xt_ge)
        conv_xt_ge = self.conv_xt_ge_2(conv_xt_ge)
        conv_xt_ge = F.relu(conv_xt_ge)
        conv_xt_ge = self.pool_xt_ge_2(conv_xt_ge)
        conv_xt_ge = self.conv_xt_ge_3(conv_xt_ge)
        conv_xt_ge = F.relu(conv_xt_ge)
        conv_xt_ge = self.pool_xt_ge_3(conv_xt_ge)
        xt_ge = conv_xt_ge.view(-1, conv_xt_ge.shape[1] * conv_xt_ge.shape[2])
        xt_ge = self.fc1_xt_ge(xt_ge)

        # concat
        xc = torch.cat((x, xt_meth, xt_ge), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        out = nn.Sigmoid()(out)
        return out, x
