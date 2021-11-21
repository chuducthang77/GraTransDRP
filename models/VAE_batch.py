import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# GCN-CNN based model

class VAE(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=1024, dropout=0.2):

        super(VAE, self).__init__()

        # cell line feature
        self.encoder_1 = nn.Conv1d(in_channels=1, out_channels = n_filters, kernel_size = 32)
        self.pool_1 = nn.MaxPool1d(13)
        self.encoder_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters*2, kernel_size=32)
        self.pool_2 = nn.MaxPool1d(11)
        self.encoder_3 = nn.Conv1d(in_channels=n_filters*2, out_channels=n_filters*4, kernel_size=32)
        self.pool_3 = nn.MaxPool1d(9)
        
        self.unpool_3 = nn.Upsample(scale_factor=9)
        self.decoder_3 = nn.ConvTranspose1d(in_channels=n_filters*4, out_channels=n_filters*2, kernel_size=32)
        self.unpool_2 = nn.Upsample(scale_factor = 11)
        self.decoder_2 = nn.ConvTranspose1d(in_channels=n_filters*2, out_channels=n_filters, kernel_size=32)
        self.unpool_1 = nn.Upsample(scale_factor = 13)
        self.decoder_1 = nn.ConvTranspose1d(in_channels=n_filters, out_channels=1, kernel_size=32)
        
        self.fc_mu = nn.Linear(1280, output_dim)
        self.fc_var = nn.Linear(1280, output_dim)
        
        self.decoder_input = nn.Linear(output_dim, 1280)
        
        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):

        # protein input feed-forward:
        target = data[:,None,:]
        encode = self.encoder_1(target)
        encode = self.pool_1(encode)
        encode = self.encoder_2(encode)
        encode = self.pool_2(encode)
        encode = self.encoder_3(encode)
        encode = self.pool_3(encode)
        encode_flatten = torch.flatten(encode, start_dim=1)

        mu = self.fc_mu(encode_flatten)
        var = self.fc_var(encode_flatten)

        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        z = eps * std + mu

        decode = self.decoder_input(z)
        decode = decode.view(-1,128,10)

        decode = self.unpool_3(encode)
        decode = self.decoder_3(decode)
        decode = self.unpool_2(decode)
        decode = self.decoder_2(decode)
        decode = self.unpool_1(decode)
        decode = self.decoder_1(decode)

        # 1d conv layers

        return decode, var, mu, z
