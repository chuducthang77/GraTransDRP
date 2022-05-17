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

list_cell_mix_test = np.load('list_drug_mix_test.npy')

dataset = 'GDSC'
test_batch = 32
num_epoch = 30

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_data = TestbedDataset(root='data', dataset=dataset+'_test_mix')
test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)

test_drug_count = {}
for i in range(len(list_cell_mix_test)):
    if list_cell_mix_test[i] in test_drug_count:
        test_drug_count[list_cell_mix_test[i]] += 1
    else:
        test_drug_count[list_cell_mix_test[i]] = 1
        
test_drug_result = {}
for epoch in range(num_epoch):
    G_test,P_test = predicting(model, device, test_loader)

    mse_res = mse_cust(G_test,P_test)
    for i in range(len(mse_res)):
        if list_cell_mix_test[i] in test_drug_result:
            test_drug_result[list_cell_mix_test[i]] += mse_res[i]
        else:
            test_drug_result[list_cell_mix_test[i]] = mse_res[i]
    
for key, value in test_drug_result.items():
    test_drug_result[key] /= (test_drug_count[key] * num_epoch)
    
test_drug_result = dict(sorted(test_drug_result.items(), key=lambda item: item[1]))
print(test_drug_result)
draw_cust_mse(test_drug_result)
