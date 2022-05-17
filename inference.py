import numpy as np
import torch
import torch.nn as nn
from models.gat_gcn_transformer_meth_ge import GAT_GCN_Transformer_meth_ge
from utils import *

def predicting(model, device, loader):
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data
            
            output, _ = model(data)

            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()

model = GAT_GCN_Transformer_meth_ge()
model.load_state_dict(torch.load('model_GAT_GCN_Transformer_meth_ge_GDSC.model'))
model.eval()

test_drug = np.load('test_drug_mix.npy')
test_drug_dict = {}
temp = None
for drug in test_drug:
    if temp != drug[0]:
        temp = drug[0]
        test_drug_dict[drug[0]] = 1
    else:
        test_drug_dict[drug[0]] += 1

dataset = 'GDSC'
test_batch = 32
num_epoch = 3
test_drug_result = {}
for i in range(len(test_drug_dict.keys())):
    test_drug_result[list(test_drug_dict.keys())[i]] = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_data = TestbedDataset(root='data', dataset=dataset+'_test_mix')
test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)


for epoch in range(num_epoch):
    G_test,P_test = predicting(model, device, test_loader)

    mse_res = mse_cust(G_test,P_test)
    mse_arr = []
    for values in test_drug_dict.values():
        temp_mse = 0
        for i in range(int(values)):
            temp_mse += mse_res[i]
        temp_mse /= int(values)
        mse_arr.append(temp_mse)

    for i in range(len(test_drug_dict.keys())):
        test_drug_result[list(test_drug_dict.keys())[i]] += mse_arr[i]

for i in range(len(test_drug_dict.keys())):
        test_drug_result[list(test_drug_dict.keys())[i]] /= num_epoch
test_drug_result = dict(sorted(test_drug_result.items(), key=lambda item: item[1]))
draw_cust_mse(test_drug_result)