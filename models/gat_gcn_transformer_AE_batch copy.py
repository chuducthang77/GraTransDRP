import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# GCN-CNN based model

class GAT_GCN_Transformer_AE_batch(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(GAT_GCN_Transformer_AE_batch, self).__init__()

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

        # cell line feature
        self.encoder_1 = nn.Conv1d(in_channels=1, out_channels = n_filters, kernel_size = 8)
        self.pool_1 = nn.MaxPool1d(8)
        self.encoder_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters*2, kernel_size=8)
        self.pool_2 = nn.MaxPool1d(7)
        self.encoder_3 = nn.Conv1d(in_channels=n_filters*2, out_channels=n_filters*4, kernel_size=8)
        self.decoder_3 = nn.ConvTranspose1d(in_channels=n_filters*4, out_channels=n_filters*2, kernel_size=8)
        self.unpool_2 = nn.Upsample(scale_factor = 7)
        self.decoder_2 = nn.ConvTranspose1d(in_channels=n_filters*2, out_channels=n_filters, kernel_size=8)
        self.unpool_1 = nn.Upsample(scale_factor = 8)
        self.decoder_1 = nn.ConvTranspose1d(in_channels=n_filters, out_channels=1, kernel_size=8)

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels = n_filters, kernel_size = 8),
            nn.MaxPool1d(8),
            nn.BatchNorm1d(n_filters),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=n_filters, out_channels=n_filters*2, kernel_size=8),
            nn.MaxPool1d(7),
            nn.BatchNorm1d(n_filters*2),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=n_filters*2, out_channels=n_filters*4, kernel_size=8),
        )
        # self.fc_mu = nn.Linear(640, output_dim)
        # self.fc_var = nn.Linear(640, output_dim)
        # self.decoder_input = nn.Linear(output_dim, 640)
        self.encoder_linear = nn.Linear(640, output_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=n_filters*4, out_channels=n_filters*2, kernel_size=8),
            nn.BatchNorm1d(n_filters*2),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=7),
            nn.ConvTranspose1d(in_channels=n_filters*2, out_channels=n_filters, kernel_size=8),
            nn.BatchNorm1d(n_filters),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=8),
            nn.ConvTranspose1d(in_channels=n_filters, out_channels=1, kernel_size=8),
        )

        # combined layers
        self.fc1 = nn.Linear(2*output_dim, 1024)
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

        # protein input feed-forward:
        target = data.target_mut[:,None,:]
        # 1d conv layers
        encode = self.encoder(target)

        z = torch.flatten(encode, start_dim=1)
        z = self.encoder_linear(z)

        decode = self.decoder(encode)

        # concat
        xc = torch.cat((x, z), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        out = nn.Sigmoid()(out)
        return out, x, decode
