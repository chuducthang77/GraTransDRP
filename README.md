# Resources:
+ README.md: this file.
+ data: [download here](https://drive.google.com/drive/folders/1CKswGNVdlRupZIAUw3yKyqkSn0NNhdyr?usp=sharing)

###  source codes:
+ preprocess.py: create data in pytorch format
+ utils.py: include TestbedDataset used by create_data.py to create data, performance measures and functions to draw loss, pearson by epoch.
+ models/gat_gcn_transformer_meth_ge_mut.py, gat_gcn_transformer_ge_mut.py, gat_gcn_transformer_meth_mut.py, gat_gcn_transformer_meth_ge.py, gat_gcn_transformer_ge_only.py, gat_gcn_transformer.py, gat_gcn_transformer_meth_only.py,: proposed models receiving graphs as input for drugs.
+ training.py: train a GraTransDRP model.
+ saliancy_map.py: run this to get saliency value.


## Dependencies
+ [Torch](https://pytorch.org/)
+ [Pytorch_geometric](https://github.com/rusty1s/pytorch_geometric)
+ [Rdkit](https://www.rdkit.org/)
+ [Matplotlib](https://matplotlib.org/)
+ [Pandas](https://pandas.pydata.org/)
+ [Numpy](https://numpy.org/)
+ [Scipy](https://docs.scipy.org/doc/)

# Step-by-step running:

## 1. Create data in pytorch format
```sh
python preprocess.py --choice 0
```
choice:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0: KernelPCA
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1: PCA 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2: Isomap

This returns file pytorch format (.pt) stored at data/processed including training, validation, test set.

## 2. Train a GraTransDRP model
```sh
python training.py --model 0 --train_batch 1024 --val_batch 1024 --test_batch 1024 --lr 0.0001 --num_epoch 300 --log_interval 20 --cuda_name "cuda:0"
```
model:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0: ge_mut_meth
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1: ge_mut
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2: meth_mut
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3: meth_ge
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4: ge
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5: mut
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6: meth


To train a model using training data. The model is chosen if it gains the best MSE for testing data. 

This returns the model and result files for the modelling achieving the best MSE for testing data throughout the training.
