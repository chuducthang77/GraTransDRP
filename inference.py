import numpy as np
import pandas as pd
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

list_drug_mix_test = np.load('list_drug_mix_test.npy')
list_cell_mix_test = np.load('list_cell_mix_test.npy')

dataset = 'GDSC'
test_batch = 32
num_epoch = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_data = TestbedDataset(root='data', dataset=dataset+'_test_mix')
test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)

test_drug_count = {}
for i in range(len(list_drug_mix_test)):
    if list_drug_mix_test[i] in test_drug_count:
        test_drug_count[list_drug_mix_test[i]] += 1
    else:
        test_drug_count[list_drug_mix_test[i]] = 1

test_drug_result = {}
for epoch in range(num_epoch):
    G_test,P_test = predicting(model, device, test_loader)
    rmse_res = np.sqrt((G_test - P_test)**2)
    rp = np.corrcoef(G_test, P_test)[0,1]
    for i in range(len(rmse_res)):
        if list_drug_mix_test[i] in test_drug_result:
            test_drug_result[list_drug_mix_test[i]] += rmse_res[i]
        else:
            test_drug_result[list_drug_mix_test[i]] = rmse_res[i]

for key, value in test_drug_result.items():
    test_drug_result[key] /= (test_drug_count[key] * num_epoch)

sorted_dict = {}
sorted_keys = sorted(test_drug_result, key=test_drug_result.get) 

for w in sorted_keys:
    sorted_dict[w] = test_drug_result[w]

first2pairs = {k: sorted_dict[k] for k in list(sorted_dict.keys())[:10]}
last2pairs = {k: sorted_dict[k] for k in list(sorted_dict.keys())[-10:]}

list_drug_mix_test_reshape = list_drug_mix_test.reshape((list_drug_mix_test.shape[0],-1))
list_cell_mix_test_reshape = list_cell_mix_test.reshape((list_cell_mix_test.shape[0],-1))
G_test_reshape = G_test.reshape((G_test.shape[0],-1))
P_test_reshape = P_test.reshape((P_test.shape[0],-1))

test_drug_pearson = np.concatenate((list_drug_mix_test_reshape, list_cell_mix_test_reshape, G_test_reshape, P_test_reshape), axis=1)
df = pd.DataFrame(test_drug_pearson, columns=['Drug', 
                      'Cell-line', 'Label', 'Predict'])
test_drug_pearson_result = {}
grouped_df = df.groupby('Drug')
for key, item in grouped_df:
    test_drug_pearson_result[key] = pearson(grouped_df.get_group(key)['Label'].to_numpy().astype(np.float), grouped_df.get_group(key)['Predict'].to_numpy().astype(np.float))

sorted_pearson_dict = {}
sorted_pearson_keys = sorted(test_drug_pearson_result, key=test_drug_pearson_result.get) 

for w in sorted_pearson_keys:
    sorted_pearson_dict[w] = test_drug_pearson_result[w]

first2pairs_pearson = {k: sorted_pearson_dict[k] for k in list(sorted_pearson_dict.keys())[:10]}
last2pairs_pearson = {k: sorted_pearson_dict[k] for k in list(sorted_pearson_dict.keys())[-10:]}

label = list(first2pairs.keys()) + ['', ''] + list(last2pairs.keys())
values = list(first2pairs.values()) + [0, 0] + list(last2pairs.values())

plt.bar(label, values)
plt.xticks(rotation=90)
plt.ylabel('RMSE')
plt.title('GE & METH')
plt.savefig("Blind_rmse.png", bbox_inches='tight')

label_pearson = list(first2pairs_pearson.keys()) + ['', ''] + list(last2pairs_pearson.keys())
values_pearson = list(first2pairs_pearson.values()) + [0, 0] + list(last2pairs_pearson.values())

plt.bar(label_pearson, values_pearson)
plt.xticks(rotation=90)
plt.ylabel('CCp')
plt.title('GE & METH')
plt.savefig("Blind_ccp.png", bbox_inches='tight')