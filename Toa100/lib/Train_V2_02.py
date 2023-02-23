import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
import torch.utils.data as tud
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter, scatter_add, scatter_max

from torch_geometric.data import Data, Dataset, InMemoryDataset, Batch, download_url
from torch_geometric.datasets import GNNBenchmarkDataset, ZINC
from torch_geometric.loader import DataLoader

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.pool import graclus, max_pool
from torch_geometric.typing import OptTensor

import torch_geometric.utils as ut
from torch_geometric.utils import to_undirected, add_self_loops, dense_to_sparse, degree, softmax
from torch_geometric.utils import unbatch_edge_index, coalesce
from torch_geometric.utils.num_nodes import maybe_num_nodes

import os.path as osp
import numpy as np
from scipy.io import loadmat
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import datetime
from typing import Optional
from typing import Callable, Optional, Tuple, Union

import argparse
from HL_GCN_V2 import HL_HGCNN
from ZINC_Dataset import ZINC_HG_BM_par1

start_time = datetime.datetime.now()
if __name__ == '__main__':

    # settting parameters
    parser = argparse.ArgumentParser()
    # parser.add_argument('--node_channels', type=list, default=[64,64,64,64])
    # parser.add_argument('--edge_channels', type=list, default=[64,64,64,64])
    parser.add_argument('--n_time_64channels', type=int, default=4)
    parser.add_argument('--mlp_channels', type=list, default=[128])
    parser.add_argument('--K', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--dropout_ratio', type=float, default=0)
    parser.add_argument('--dropout_ratio_mlp', type=float, default=0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=1)
    sets = parser.parse_args(args=[]) #jupyter notebook need the input args=[]
    print(sets)

    # gpu test
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    total_epochs = 1000
    # getting model
    # torch.manual_seed(sets.manual_seed)
    model = HL_HGCNN(sets).to(device)
    # print (model)
    #loss function
    loss_func = torch.nn.L1Loss().to(device)
    # optimizer
    initial_learning_rate=0.005
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)


    # train dataset
    # root_zinc = '/home/cqf/GNN/GNN_tmp_result/1/'
    root_zinc = '/home/qiufeng/GNN/ZINC_experiment/'
    dataset_raw_train = ZINC(root=root_zinc + 'train_data ', subset=True, split='train')
    # mean and std of Y for standardisation
    data_loader_whole_train = DataLoader(dataset_raw_train, batch_size=len(dataset_raw_train))
    for id, whole_batch in enumerate(data_loader_whole_train):
        y_mean = torch.mean(whole_batch.y).to(device)
        y_std = torch.std(whole_batch.y).to(device)
        print(id)
        print(len(whole_batch))

    dataset = ZINC_HG_BM_par1(root_zinc + 'train_data ', dataset_raw_train)
    data_loader_train = DataLoader(dataset, batch_size=sets.batch_size, shuffle=True, follow_batch=['x_s','x_t'],num_workers=sets.num_workers)

    # val
    dataset_raw_val = ZINC(root=root_zinc + 'val_data ', subset=True, split='val')
    dataset = ZINC_HG_BM_par1(root_zinc + 'val_data ', dataset_raw_val)
    data_loader_val = DataLoader(dataset, batch_size=sets.batch_size, shuffle=True, follow_batch=['x_s','x_t'],num_workers=sets.num_workers)

    # out_root = '/home/cqf/GNN/GNN_tmp_result/1/'
    out_root = '/home/qiufeng/GNN/ZINC_experiment/'
    # save_name = '{}nodeChannel_{}mlpChannel_{}beta_{}batchSize_{}K_{}epoch.pth.tar'. \
    #     format(sets.node_channels, sets.mlp_channels, sets.beta, sets.batch_size, sets.K, epoch)
    save_name = '{}time_64channels_{}mlpChannel_{}beta_{}batchSize_{}K.pth.tar'. \
        format(sets.n_time_64channels, sets.mlp_channels, sets.beta, sets.batch_size, sets.K)
    save_path = out_root + save_name

    loss_train = []
    loss_val = []
    patience_val_loss = 0
    loss_preEpoch_val = 100
    loss_best_val = 100

    for epoch in range(total_epochs):
        loss_Subjects_train = torch.FloatTensor([]).to(device) # save every subject's loss to calculate the mean and std of loss in the best epoch
        loss_Subjects_val = torch.FloatTensor([]).to(device)
        # training
        model.train()
        total_train_loss = 0
        for batch_id, batch_data in enumerate(data_loader_train):
            Y = (batch_data.y.to(device)-y_mean)/y_std
            optimizer.zero_grad()
            predicted_Y = model(batch_data.to(device))
            # calculating loss
            loss = loss_func(Y, predicted_Y.squeeze(-1))
            total_train_loss = total_train_loss + loss.item()
            loss.backward()
            optimizer.step()
            loss_Subjects_train = torch.cat((loss_Subjects_train,torch.abs(Y-predicted_Y.squeeze(-1))))
        scheduler.step()
        loss_currentEpoch_train = total_train_loss / len(data_loader_train) # mean of loss
        loss_std_currentEpoch_train = torch.std(loss_Subjects_train) # std of loss
        loss_train.append(loss_currentEpoch_train)
        # validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_id, batch_data in enumerate(data_loader_val):
                Y = (batch_data.y.to(device)-y_mean)/y_std
                predicted_Y = model(batch_data.to(device))
                # calculating loss
                loss = loss_func(Y, predicted_Y.squeeze(-1))
                total_val_loss = total_val_loss + loss.item()
                loss_Subjects_val = torch.cat((loss_Subjects_val,torch.abs(Y-predicted_Y.squeeze(-1))))
        loss_currentEpoch_val = total_val_loss/len(data_loader_val)
        loss_std_currentEpoch_val = torch.std(loss_Subjects_val)
        loss_val.append(loss_currentEpoch_val)

        print('epochs {:<3}, train loss is {:.8f}, validation loss is {:.8f}'.format(epoch,loss_currentEpoch_train,loss_currentEpoch_val))
        if loss_currentEpoch_val < loss_best_val:
            loss_best_val = loss_currentEpoch_val
            # save model
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_train': loss_train,
                        'loss_val': loss_val,
                        'loss_currentEpoch_train': loss_currentEpoch_train,
                        'loss_currentEpoch_val': loss_currentEpoch_val, # save the best validation loss
                        'loss_std_currentEpoch_train':loss_std_currentEpoch_train,
                        'loss_std_currentEpoch_val':loss_std_currentEpoch_val
                        }, save_path)

        if loss_currentEpoch_val < loss_preEpoch_val:
            patience_val_loss = 0
            loss_preEpoch_val = loss_currentEpoch_val
        else:
            patience_val_loss += 1
            loss_preEpoch_val = loss_currentEpoch_val
        if patience_val_loss > 10 or loss_currentEpoch_val < 1e-5:
            print(f'current loss{loss_currentEpoch_val}')
            print(f'patience is {patience_val_loss}')
            print(f'early stop epoch is {epoch}')
            break


# run time calculation
end_time = datetime.datetime.now()
print(end_time - start_time)
print((end_time - start_time).seconds)
# load model
checkpoint = torch.load(save_path)
epoch = checkpoint['epoch']
loss_train= checkpoint['loss_train']
loss_val = checkpoint['loss_val']
loss_currentEpoch_train = checkpoint['loss_currentEpoch_train']
loss_currentEpoch_val = checkpoint['loss_currentEpoch_val']
loss_std_currentEpoch_train = checkpoint['loss_std_currentEpoch_train']
loss_std_currentEpoch_val = checkpoint['loss_std_currentEpoch_val']
print(f'best epoch is {epoch}')
print(f'loss_std_train is {loss_std_currentEpoch_train}')
print(f'loss_std_val is {loss_std_currentEpoch_val}')
print(f'min_loss_train is {min(loss_train)}')
print(f'min_loss_val is {min(loss_val)}')
print(f'loss_currentEpoch_train is {loss_currentEpoch_train}')
print(f'loss_currentEpoch_val is {loss_currentEpoch_val}')
print(f'the epoch val loss is {loss_val[epoch]}')

model = HL_HGCNN(sets)
model.load_state_dict(checkpoint['model_state_dict'])
# print(model)

# test
dataset_raw_test = ZINC(root=root_zinc + 'test_data ', subset=True, split='test')
dataset = ZINC_HG_BM_par1(root_zinc + 'test_data ', dataset_raw_test)
data_loader_test = DataLoader(dataset, batch_size=sets.batch_size, shuffle=True, follow_batch=['x_s', 'x_t'],
                             num_workers=sets.num_workers)
# testing
model.to(device)
model.eval()
total_test_loss = 0
loss_Subjects_test = torch.FloatTensor([]).to(device)
with torch.no_grad():
    for batch_id, batch_data in enumerate(data_loader_test):
        Y = (batch_data.y.to(device) - y_mean) / y_std
        predicted_Y = model(batch_data.to(device))
        # calculating loss
        loss = loss_func(Y, predicted_Y.squeeze(-1))
        total_test_loss = total_test_loss + loss.item()
        loss_Subjects_test = torch.cat((loss_Subjects_test,torch.abs(Y-predicted_Y.squeeze(-1))))
loss_test = (total_test_loss / len(data_loader_test))
loss_std_currentEpoch_test = torch.std(loss_Subjects_test)
print(f'loss_std_test is {loss_std_currentEpoch_test}')
print('loss_test is {:.8f}'.format(loss_test))
# save figure
plt.figure()
# epoch start form 0
plt.plot(range(epoch+1), loss_train, 'b', label='loss_train_min={0:.4f}'.format(min(loss_train)))
plt.plot(range(epoch+1), loss_val, 'r', label='loss_val_min={0:.4f}'.format(min(loss_val)))
plt.title(save_name[:-8],fontsize=10,loc='center',color='purple')
plt.ylim(0, 0.8)
plt.ylabel('loss')
plt.xlabel('best epoch is {}--loss_test={:.4f}'.format(epoch,loss_test))
plt.legend()
plt.savefig( save_path[:-8]+'.jpg')
# plt.show()
