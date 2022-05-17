import torch
import torch.nn as nn
from models.gat_gcn_transformer_meth_ge import GAT_GCN_Transformer_meth_ge

model = GAT_GCN_Transformer_meth_ge()
model.load_state_dict(torch.load('model_GAT_GCN_Transformer_meth_ge_GDSC.model'))
model.eval()