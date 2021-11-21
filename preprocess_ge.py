import os
import csv
from pubchempy import *
import numpy as np
import numbers
import h5py
import math
import pandas as pd
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from torch._C import device
from utils import *
import random
import pickle
import sys
import matplotlib.pyplot as plt
import argparse
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, Isomap
import torch
import torch.nn as nn
from torch import optim
from models.VAE_batch import VAE
from models.VAE_NN import VAE_NN
from models.AE_NN import AE_NN
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


def is_not_float(string_list):
    try:
        for string in string_list:
            float(string)
        return False
    except:
        return True

"""
The following 4 function is used to preprocess the drug data. We download the drug list manually, and download the SMILES format using pubchempy. Since this part is time consuming, I write the cids and SMILES into a csv file. 
"""

folder = "data/"
#folder = ""

def load_drug_list():
    filename = folder + "Druglist.csv"
    csvfile = open(filename, "rb")
    reader = csv.reader(csvfile)
    next(reader, None)
    drugs = []
    for line in reader:
        drugs.append(line[0])
    drugs = list(set(drugs))
    return drugs

def write_drug_cid():
    drugs = load_drug_list()
    drug_id = []
    datas = []
    outputfile = open(folder + 'pychem_cid.csv', 'wb')
    wr = csv.writer(outputfile)
    unknow_drug = []
    for drug in drugs:
        c = get_compounds(drug, 'name')
        if drug.isdigit():
            cid = int(drug)
        elif len(c) == 0:
            unknow_drug.append(drug)
            continue
        else:
            cid = c[0].cid
        print(drug, cid)
        drug_id.append(cid)
        row = [drug, str(cid)]
        wr.writerow(row)
    outputfile.close()
    outputfile = open(folder + "unknow_drug_by_pychem.csv", 'wb')
    wr = csv.writer(outputfile)
    wr.writerow(unknow_drug)

def cid_from_other_source():
    """
    some drug can not be found in pychem, so I try to find some cid manually.
    the small_molecule.csv is downloaded from http://lincs.hms.harvard.edu/db/sm/
    """
    f = open(folder + "small_molecule.csv", 'r')
    reader = csv.reader(f)
    reader.next()
    cid_dict = {}
    for item in reader:
        name = item[1]
        cid = item[4]
        if not name in cid_dict: 
            cid_dict[name] = str(cid)

    unknow_drug = open(folder + "unknow_drug_by_pychem.csv").readline().split(",")
    drug_cid_dict = {k:v for k,v in cid_dict.iteritems() if k in unknow_drug and not is_not_float([v])}
    return drug_cid_dict

def load_cid_dict():
    reader = csv.reader(open(folder + "pychem_cid.csv"))
    pychem_dict = {}
    for item in reader:
        pychem_dict[item[0]] = item[1]
    pychem_dict.update(cid_from_other_source())
    return pychem_dict


def download_smiles():
    cids_dict = load_cid_dict()
    cids = [v for k,v in cids_dict.iteritems()]
    inv_cids_dict = {v:k for k,v in cids_dict.iteritems()}
    download('CSV', folder + 'drug_smiles.csv', cids, operation='property/CanonicalSMILES,IsomericSMILES', overwrite=True)
    f = open(folder + 'drug_smiles.csv')
    reader = csv.reader(f)
    header = ['name'] + reader.next()
    content = []
    for line in reader:
        content.append([inv_cids_dict[line[0]]] + line)
    f.close()
    f = open(folder + "drug_smiles.csv", "w")
    writer = csv.writer(f)
    writer.writerow(header)
    for item in content:
        writer.writerow(item)
    f.close()

"""
The following code will convert the SMILES format into onehot format
"""

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index

def load_drug_smile():
    reader = csv.reader(open(folder + "drug_smiles.csv"))
    next(reader, None)

    drug_dict = {}
    drug_smile = []

    for item in reader:
        name = item[0]
        smile = item[2]

        if name in drug_dict:
            pos = drug_dict[name]
        else:
            pos = len(drug_dict)
            drug_dict[name] = pos
        drug_smile.append(smile)
    
    smile_graph = {}
    for smile in drug_smile:
        g = smile_to_graph(smile)
        smile_graph[smile] = g
    
    return drug_dict, drug_smile, smile_graph

def save_cell_mut_matrix():
    f = open(folder + "PANCANCER_Genetic_feature.csv")
    reader = csv.reader(f)
    next(reader)
    features = {}
    cell_dict = {}
    mut_dict = {}
    matrix_list = []

    for item in reader:
        cell_id = item[1]
        mut = item[5]
        is_mutated = int(item[6])

        if mut in mut_dict:
            col = mut_dict[mut]
        else:
            col = len(mut_dict)
            mut_dict[mut] = col

        if cell_id in cell_dict:
            row = cell_dict[cell_id]
        else:
            row = len(cell_dict)
            cell_dict[cell_id] = row
        if is_mutated == 1:
            matrix_list.append((row, col))
    
    cell_feature = np.zeros((len(cell_dict), len(mut_dict)))

    for item in matrix_list:
        cell_feature[item[0], item[1]] = 1

    with open('mut_dict', 'wb') as fp:
        pickle.dump(mut_dict, fp)
    
    return cell_dict, cell_feature

"""
This part is used to read PANCANCER Meth Cell line features
"""

def save_cell_meth_matrix():
    f = open(folder + "METH_CELLLINES_BEMs_PANCAN.csv")
    reader = csv.reader(f)
    firstRow = next(reader)
    numberCol = len(firstRow) - 1
    features = {}
    cell_dict = {}
    matrix_list = []
    mut_dict = {}
    for item in reader:
        cell_id = item[1]
        mut = item[2]
        is_mutated = int(item[3])

        if mut in mut_dict:
            col = mut_dict[mut]
        else:
            col = len(mut_dict)
            mut_dict[mut] = col

        if cell_id in cell_dict:
            row = cell_dict[cell_id]
        else:
            row = len(cell_dict)
            cell_dict[cell_id] = row
        if is_mutated == 1:
            matrix_list.append((row, col))
    
    cell_feature = np.zeros((len(cell_dict), len(mut_dict)))

    for item in matrix_list:
        cell_feature[item[0], item[1]] = 1

    with open('mut_dict', 'wb') as fp:
        pickle.dump(mut_dict, fp)
    

    return cell_dict, cell_feature
    

"""
This part is used to read PANCANCER Gene Expression Cell line features
"""

def save_cell_ge_matrix():
    f = open(folder + "Cell_line_RMA_proc_basalExp.csv")
    reader = csv.reader(f)
    firstRow = next(reader)
    numberCol = len(firstRow) - 1
    features = {}
    cell_dict = {}
    matrix_list = []
    for item in reader:
        cell_id = item[0]
        ge = []
        for i in range(1, len(item)):
            ge.append(int(item[i]))
        cell_dict[cell_id] = np.asarray(ge)
    return cell_dict


def save_cell_oge_matrix():
    f = open(folder + "Cell_line_RMA_proc_basalExp.txt")
    line = f.readline()
    elements = line.split()
    cell_names = []
    feature_names = []
    cell_dict = {}
    i = 0
    for cell in range(2, len(elements)):
        if i < 500:
            cell_name = elements[cell].replace("DATA.", "")
            cell_names.append(cell_name)
            cell_dict[cell_name] = []

    min = 0
    max = 12
    for line in f.readlines():
        elements = line.split("\t")
        if len(elements) < 2:
            print(line)
            continue
        feature_names.append(elements[1])

        for i in range(2, len(elements)):
            cell_name = cell_names[i-2]
            value = float(elements[i])
            if min == 0:
                min = value
            if value < min:
                min = value
            if max < value:
                value = max
            cell_dict[cell_name].append(value)
    #print(min)
    #print(max)
    cell_feature = []
    for cell_name in cell_names:
        for i in range(0, len(cell_dict[cell_name])):
            cell_dict[cell_name][i] = (cell_dict[cell_name][i] - min)/(max - min)
        cell_dict[cell_name] = np.asarray(cell_dict[cell_name])
        cell_feature.append(np.asarray(cell_dict[cell_name]))
    
    cell_feature = np.asarray(cell_feature)
    # cell_feature = cell_feature.flatten()
    # print(cell_feature.shape)
    # print((cell_feature > 11.5).sum())
    # plt.hist(cell_feature.flatten())
    # plt.show()
    # exit()
    i = 0
    for cell in list(cell_dict.keys()):
        cell_dict[cell] = i
        i += 1

    # print(len(list(cell_dict.values())))
    # exit()
    #print(cell_dict['910927'][23])
    return cell_dict, cell_feature

def train(model, device, train_loader, optimizer, epoch, log_interval, model_st):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    loss_ae = nn.MSELoss()
    avg_loss = []
    weight_fn = 0.01
    weight_ae = 2
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        #For non-variational autoencoder
        if 'VAE' not in model_st:
            output, _ = model(data)
            loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        else:
        #For variation autoencoder
            output, _, decode, log_var, mu = model(data)
            loss = weight_fn*loss_fn(output, data.y.view(-1, 1).float().to(device)) + loss_ae(decode, data.target_mut[:,None,:].float().to(device)) + torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())
        if batch_idx % log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
    return sum(avg_loss)/len(avg_loss)

def predicting(model, device, loader, model_st):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            #Non-variational autoencoder
            if 'VAE' not in model_st:
                output, _ = model(data)
            else:
            #Variational autoencoder
                output, _, decode, log_var, mu = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


"""
This part is used to extract the drug - cell interaction strength. it contains IC50, AUC, Max conc, RMSE, Z_score
"""
class DataBuilder(Dataset):
    def __init__(self, cell_feature_ge):
        self.cell_feature_ge = cell_feature_ge
        self.cell_feature_ge = torch.FloatTensor(self.cell_feature_ge)
        self.len = self.cell_feature_ge[0]
    
    def __getitem__(self, index):
        return self.cell_feature_ge[index]

    def __len__(self):
        return self.len

def save_mix_drug_cell_matrix():
    f = open(folder + "PANCANCER_IC.csv")
    reader = csv.reader(f)
    next(reader)

    cell_dict_mut, cell_feature_mut = save_cell_mut_matrix()
    cell_dict_meth, cell_feature_meth = save_cell_meth_matrix()
    cell_dict_ge, cell_feature_ge = save_cell_oge_matrix()
    # print(cell_feature_mut.shape)

    

    # print(cell_feature_ge.shape)
    drug_dict, drug_smile, smile_graph = load_drug_smile()

    temp_data = []

    # xem lai cai nay nao
    bExist = np.zeros((len(drug_dict), len(cell_dict_mut)))

    for item in reader:
        drug = item[0]
        cell = item[3]
        ic50 = item[8]
        ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))
        temp_data.append((drug, cell, ic50))

    xd = []
    xc_mut = []
    xc_meth = []
    xc_ge = []
    y = []
    lst_drug = []
    lst_cell = []
    random.shuffle(temp_data)
    
    #Train the VAE on cell_feature_ge
    D_in = cell_feature_ge.shape[1]
    # cell_feature_ge = torch.FloatTensor(cell_feature_ge).to(0)
    # model = AE_NN().to(0)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # loss_fn = nn.MSELoss()
    # loss_ae = nn.MSELoss()
    weight_fn = 0.01
    weight_ae = 2
    epochs = 500
    log_interval = 50
    val_losses = []
    train_losses = []
    dataset = []
    batch_size = 64
    num_batch = round(cell_feature_ge.shape[0]/batch_size)
    # for i in range(num_batch):
    #     if i * batch_size < cell_feature_ge.shape[0]:
    #         dataset.append(cell_feature_ge[i*batch_size: (i+1)*batch_size,:])
    #     else:
    #         dataset.append(cell_feature_ge[i*batch_size:,:])
    # for epoch in range(1, epochs+1):
    #     model.train()
    #     train_loss = 0
    #     i = 0
    #     for i in range(num_batch):
    #         data = torch.FloatTensor(dataset[i]).to(0)
    #         optimizer.zero_grad()
    #         # #Variational Autoencoder
    #         # decode, var, mu, z = model(data)
    #         # loss = loss_ae(decode, data[:,None,:]) + torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim = 1), dim = 0) 
    #         # Autoencoder 
    #         encode, decode = model(data)
    #         loss = loss_ae(decode, data)
    #         loss.backward()
    #         train_loss += loss.item()
    #         optimizer.step()
    #         if epoch % 1 == 0:        
    #             print('====> Epoch: {} Average loss: {:.4f}'.format(
    #                 epoch, train_loss / len(data)))
    #             train_losses.append(train_loss / len(data))
    # embedding = []
    # with torch.no_grad():
    #     for i in range(num_batch):
    #         data = torch.FloatTensor(dataset[i]).to(0)
    #         optimizer.zero_grad()
    #         # decode, var, mu, z  = model(data)
    #         encode, decode = model(data)
    #         embedding.append(encode)
    #     embedding = torch.cat(embedding)
    # print(embedding.size())
    # cell_feature_ge = embedding.cpu().detach().numpy()
    # print(cell_feature_ge.shape)
    # prinhttps://www.youtube.com/watch?v=Q-M1RahsqWY&t=872st(cell_feature_ge.shape)
    # exit()
    # PCA
    # pca = PCA(n_components=1000)
    # cell_feature_ge = pca.fit_transform(cell_feature_ge)
    # Kernel PCA
    kpca = KernelPCA(n_components=1000, kernel='rbf', gamma=131, random_state=42)
    cell_feature_ge = kpca.fit_transform(cell_feature_ge)

    # Tsne
    # tsne = TSNE(n_components=480,method='exact')
    # cell_feature_ge = tsne.fit_transform(cell_feature_ge)

    #Isomap
    isomap = Isomap(n_components=480)
    # cell_feature_ge = isomap.fit_transform(cell_feature_ge)

    for data in temp_data:
        drug, cell, ic50 = data
        if drug in drug_dict and cell in cell_dict_ge and cell in cell_dict_meth:
            xd.append(drug_smile[drug_dict[drug]])
            xc_mut.append(cell_feature_mut[cell_dict_mut[cell]])
            xc_ge.append(cell_feature_ge[cell_dict_ge[cell]])
            xc_meth.append(cell_feature_meth[cell_dict_meth[cell]])
            #nc = np.concatenate((cell_feature_mut[cell_dict_mut[cell]], cell_dict_ge[cell]), axis=None)
            #xc_mut.append(nc)
            # xc_mut.append(cell_feature_mut[cell_dict_mut[cell]])
            y.append(ic50)
            bExist[drug_dict[drug], cell_dict_mut[cell]] = 1
            
            lst_drug.append(drug)
            lst_cell.append(cell)

    with open('drug_dict', 'wb') as fp:
        pickle.dump(drug_dict, fp)

    # xd, xc_mut, xc_meth, xc_ge, y = np.asarray(xd), np.asarray(xc_mut), np.asarray(xc_meth), np.asarray(xc_ge), np.asarray(y)
    xd = np.asarray(xd)
    xc_mut = np.asarray(xc_mut)
    xc_ge = np.asarray(xc_ge)
    xc_meth = np.asarray(xc_meth)
    y = np.asarray(y)

    size = int(xd.shape[0] * 0.8)
    size1 = int(xd.shape[0] * 0.9)

    with open('list_drug_mix_test', 'wb') as fp:
        pickle.dump(lst_drug[size1:], fp)
        
    with open('list_cell_mix_test', 'wb') as fp:
        pickle.dump(lst_cell[size1:], fp)

    xd_train = xd[:size]
    xd_val = xd[size:size1]
    xd_test = xd[size1:]

    xc_ge_train = xc_ge[:size]
    xc_ge_val = xc_ge[size:size1]
    xc_ge_test = xc_ge[size1:]

    xc_meth_train = xc_meth[:size]
    xc_meth_val = xc_meth[size:size1]
    xc_meth_test = xc_meth[size1:]
    
    xc_mut_train = xc_mut[:size]
    xc_mut_val = xc_mut[size:size1]
    xc_mut_test = xc_mut[size1:]
    
    
    y_train = y[:size]
    y_val = y[size:size1]
    y_test = y[size1:]

    dataset = 'GDSC'
    print('preparing ', dataset + '_train.pt in pytorch format!')

    train_data = TestbedDataset(root='data', dataset=dataset+'_train_mix', xd=xd_train, xt_ge=xc_ge_train, xt_meth=xc_meth_train, xt_mut=xc_mut_train, y=y_train, smile_graph=smile_graph)
    val_data = TestbedDataset(root='data', dataset=dataset+'_val_mix', xd=xd_val, xt_ge=xc_ge_val, xt_meth=xc_meth_val,xt_mut=xc_mut_val, y=y_val, smile_graph=smile_graph)
    test_data = TestbedDataset(root='data', dataset=dataset+'_test_mix', xd=xd_test, xt_ge=xc_ge_test, xt_meth=xc_meth_test, xt_mut=xc_mut_test, y=y_test, smile_graph=smile_graph)
    print("build data complete")

def save_blind_drug_matrix():
    f = open(folder + "PANCANCER_IC.csv")
    reader = csv.reader(f)
    next(reader)

    cell_dict, cell_feature = save_cell_mut_matrix()
    drug_dict, drug_smile, smile_graph = load_drug_smile()

    matrix_list = []

    temp_data = []

    xd_train = []
    xc_train = []
    y_train = []

    xd_val = []
    xc_val = []
    y_val = []

    xd_test = []
    xc_test = []
    y_test = []

    xd_unknown = []
    xc_unknown = []
    y_unknown = []

    dict_drug_cell = {}

    bExist = np.zeros((len(drug_dict), len(cell_dict)))

    for item in reader:
        drug = item[0]
        cell = item[3]
        ic50 = item[8]
        ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))
        
        temp_data.append((drug, cell, ic50))

    random.shuffle(temp_data)
    
    for data in temp_data:
        drug, cell, ic50 = data
        if drug in drug_dict and cell in cell_dict:
            if drug in dict_drug_cell:
                dict_drug_cell[drug].append((cell, ic50))
            else:
                dict_drug_cell[drug] = [(cell, ic50)]
            
            bExist[drug_dict[drug], cell_dict[cell]] = 1

    lstDrugTest = []
    print(len(dict_drug_cell))
    # size = int(len(dict_drug_cell) * 0.8)
    # size1 = int(len(dict_drug_cell) * 0.9)
    # pos = 0
    # for drug,values in dict_drug_cell.items():
        # pos += 1
        # for v in values:
            # cell, ic50 = v
            # if pos < size:
                # xd_train.append(drug_smile[drug_dict[drug]])
                # xc_train.append(cell_feature[cell_dict[cell]])
                # y_train.append(ic50)
            # elif pos < size1:
                # xd_val.append(drug_smile[drug_dict[drug]])
                # xc_val.append(cell_feature[cell_dict[cell]])
                # y_val.append(ic50)
            # else:
                # xd_test.append(drug_smile[drug_dict[drug]])
                # xc_test.append(cell_feature[cell_dict[cell]])
                # y_test.append(ic50)
                # lstDrugTest.append(drug)

    # with open('drug_bind_test', 'wb') as fp:
        # pickle.dump(lstDrugTest, fp)
    
    # print(len(y_train), len(y_val), len(y_test))

    # xd_train, xc_train, y_train = np.asarray(xd_train), np.asarray(xc_train), np.asarray(y_train)
    # xd_val, xc_val, y_val = np.asarray(xd_val), np.asarray(xc_val), np.asarray(y_val)
    # xd_test, xc_test, y_test = np.asarray(xd_test), np.asarray(xc_test), np.asarray(y_test)

    # dataset = 'GDSC'
    # print('preparing ', dataset + '_train.pt in pytorch format!')
    # train_data = TestbedDataset(root='data', dataset=dataset+'_train_blind', xd=xd_train, xt=xc_train, y=y_train, smile_graph=smile_graph)
    # val_data = TestbedDataset(root='data', dataset=dataset+'_val_blind', xd=xd_val, xt=xc_val, y=y_val, smile_graph=smile_graph)
    # test_data = TestbedDataset(root='data', dataset=dataset+'_test_blind', xd=xd_test, xt=xc_test, y=y_test, smile_graph=smile_graph)


def save_blind_cell_matrix():
    f = open(folder + "PANCANCER_IC.csv")
    reader = csv.reader(f)
    next(reader)

    cell_dict, cell_feature = save_cell_mut_matrix()
    drug_dict, drug_smile, smile_graph = load_drug_smile()

    matrix_list = []

    temp_data = []

    xd_train = []
    xc_train = []
    y_train = []

    xd_val = []
    xc_val = []
    y_val = []

    xd_test = []
    xc_test = []
    y_test = []

    xd_unknown = []
    xc_unknown = []
    y_unknown = []

    dict_drug_cell = {}

    bExist = np.zeros((len(drug_dict), len(cell_dict)))

    for item in reader:
        drug = item[0]
        cell = item[3]
        ic50 = item[8]
        ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))
        
        temp_data.append((drug, cell, ic50))

    random.shuffle(temp_data)
    
    for data in temp_data:
        drug, cell, ic50 = data
        if drug in drug_dict and cell in cell_dict:
            if cell in dict_drug_cell:
                dict_drug_cell[cell].append((drug, ic50))
            else:
                dict_drug_cell[cell] = [(drug, ic50)]
            
            bExist[drug_dict[drug], cell_dict[cell]] = 1

    lstCellTest = []

    size = int(len(dict_drug_cell) * 0.8)
    size1 = int(len(dict_drug_cell) * 0.9)
    pos = 0
    for cell,values in dict_drug_cell.items():
        pos += 1
        for v in values:
            drug, ic50 = v
            if pos < size:
                xd_train.append(drug_smile[drug_dict[drug]])
                xc_train.append(cell_feature[cell_dict[cell]])
                y_train.append(ic50)
            elif pos < size1:
                xd_val.append(drug_smile[drug_dict[drug]])
                xc_val.append(cell_feature[cell_dict[cell]])
                y_val.append(ic50)
            else:
                xd_test.append(drug_smile[drug_dict[drug]])
                xc_test.append(cell_feature[cell_dict[cell]])
                y_test.append(ic50)
                lstCellTest.append(cell)

    with open('cell_bind_test', 'wb') as fp:
        pickle.dump(lstCellTest, fp)
    
    print(len(y_train), len(y_val), len(y_test))

    xd_train, xc_train, y_train = np.asarray(xd_train), np.asarray(xc_train), np.asarray(y_train)
    xd_val, xc_val, y_val = np.asarray(xd_val), np.asarray(xc_val), np.asarray(y_val)
    xd_test, xc_test, y_test = np.asarray(xd_test), np.asarray(xc_test), np.asarray(y_test)

    dataset = 'GDSC'
    print('preparing ', dataset + '_train.pt in pytorch format!')
    train_data = TestbedDataset(root='data', dataset=dataset+'_train_cell_blind', xd=xd_train, xt=xc_train, y=y_train, smile_graph=smile_graph)
    val_data = TestbedDataset(root='data', dataset=dataset+'_val_cell_blind', xd=xd_val, xt=xc_val, y=y_val, smile_graph=smile_graph)
    test_data = TestbedDataset(root='data', dataset=dataset+'_test_cell_blind', xd=xd_test, xt=xc_test, y=y_test, smile_graph=smile_graph)

def save_best_individual_drug_cell_matrix():
    f = open(folder + "PANCANCER_IC.csv")
    reader = csv.reader(f)
    next(reader)

    cell_dict, cell_feature = save_cell_mut_matrix()
    drug_dict, drug_smile, smile_graph = load_drug_smile()

    matrix_list = []

    temp_data = []

    xd_train = []
    xc_train = []
    y_train = []

    dict_drug_cell = {}

    bExist = np.zeros((len(drug_dict), len(cell_dict)))
    i=0
    for item in reader:
        drug = item[0]
        cell = item[3]
        ic50 = item[8]
        ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))
        
        if drug == "Bortezomib":
            temp_data.append((drug, cell, ic50))
    random.shuffle(temp_data)
    
    for data in temp_data:
        drug, cell, ic50 = data
        if drug in drug_dict and cell in cell_dict:
            if drug in dict_drug_cell:
                dict_drug_cell[drug].append((cell, ic50))
            else:
                dict_drug_cell[drug] = [(cell, ic50)]
            
            bExist[drug_dict[drug], cell_dict[cell]] = 1
    cells = []
    for drug,values in dict_drug_cell.items():
        for v in values:
            cell, ic50 = v
            xd_train.append(drug_smile[drug_dict[drug]])
            xc_train.append(cell_feature[cell_dict[cell]])
            y_train.append(ic50)
            cells.append(cell)

    xd_train, xc_train, y_train = np.asarray(xd_train), np.asarray(xc_train), np.asarray(y_train)
    with open('cell_blind_sal', 'wb') as fp:
        pickle.dump(cells, fp)
    dataset = 'GDSC'
    print('preparing ', dataset + '_train.pt in pytorch format!')
    train_data = TestbedDataset(root='data', dataset=dataset+'_bortezomib', xd=xd_train, xt=xc_train, y=y_train, smile_graph=smile_graph, saliency_map=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='prepare dataset to train model')
    parser.add_argument('--choice', type=int, required=False, default=0, help='0.mix test, 1.saliency value, 2.drug blind, 3.cell blind')
    args = parser.parse_args()
    choice = args.choice
    if choice == 0:
        # save mix test dataset
        save_mix_drug_cell_matrix()
    elif choice == 1:
        # save saliency map dataset
        save_best_individual_drug_cell_matrix()
    elif choice == 2:
        # save blind drug dataset
        save_blind_drug_matrix()
    elif choice == 3:
        # save blind cell dataset
        save_blind_cell_matrix()
    else:
        print("Invalide option, choose 0 -> 4")