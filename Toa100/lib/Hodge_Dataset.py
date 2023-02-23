#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 21:07:12 2022

@author: jinghan
"""

import os.path as osp
from torch_geometric.data import Dataset, download_url, Data, InMemoryDataset
from torch_geometric.utils import add_self_loops, degree, to_undirected, dense_to_sparse, coalesce
from scipy.io import loadmat
import torch
import torch.utils.data as tud
import numpy as np
from lib.Hodge_Cheb_Conv import *
from torch_geometric.datasets import GNNBenchmarkDataset, ZINC
from torch_geometric.loader import DataLoader


class PairData(Data):
    def __init__(self, edge_index_s=None, x_s=None, edge_index_t=None, x_t=None,
                edge_weight_s=None, edge_weight_t=None, y=None):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.x_t = x_t
        self.edge_weight_s = edge_weight_s
        self.edge_weight_t = edge_weight_t
        self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)
        

def adj2par1(edge_index, num_node, num_edge):
    col_idx = torch.cat([torch.arange(edge_index.shape[1]),torch.arange(edge_index.shape[1])]
                        ,dim=-1).to(edge_index.device)
    row_idx = torch.cat([edge_index[0],edge_index[1]], dim=-1).to(edge_index.device)
    val = torch.cat([edge_index[0].new_full(edge_index[0].shape,-1),
                     edge_index[0].new_full(edge_index[0].shape,1)],dim=-1).to(torch.float)
    par1_sparse = torch.sparse.FloatTensor(torch.cat([row_idx, col_idx], dim=-1).view(2,-1),
                                           val,torch.Size([num_node, num_edge]))
    return par1_sparse

###############################################################################
#########################  Image Graph  ############################
###############################################################################
class ZINC_HG_BM_par1(Dataset):
    def __init__(self, root, dataset):

        self.root = root
        self.dataset = dataset
        super().__init__(root)
  
    @property
    def processed_file_names(self):
        return ['ZINC_BM_'+str(fileidx+1)+'.pt' for fileidx in range(self.len())]

    def len(self):
        return len(self.dataset)
#         return len(self.processed_file_names)

    def get(self,idx):
        data_zip = torch.load(osp.join(self.processed_dir, 'ZINC_BM_'+str(idx+1)+'.pt'))
        data = data_zip['graph']
        data.y = (data.y - 0.0153)/2.0109
        data.num_node1 = data.x_t.shape[0]
        data.num_edge1 = data.x_s.shape[0]
        data.num_nodes = data.x_t.shape[0]
        data.x_s = data.x_s.to(torch.float)
        data.x_t = data.x_t.to(torch.float)
        return data
            
    def process(self):
        i=0
        loader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=1)
        for data in loader:

            edge_index,edge_attr = to_undirected(data.edge_index, data.edge_attr,reduce='min')
            idx = edge_index[0]<edge_index[1]

            edge_index,edge_attr = edge_index[:,idx], edge_attr[idx]
            par1 = adj2par1(edge_index, data.x.shape[0], edge_index.shape[1]).to_dense()
            L0 = torch.matmul(par1, par1.T)
            lambda0, _ = torch.linalg.eigh(L0)
            L1 = torch.matmul(par1.T, par1)
            edge_index_t, edge_weight_t = dense_to_sparse(2*L0/lambda0.max())
            edge_index_s, edge_weight_s = dense_to_sparse(2*L1/lambda0.max())

            x_s = F.one_hot(edge_attr-1,num_classes=3) # the min value of edge_attr=1, one_hot start from value 0
            x_t = F.one_hot(data.x.squeeze(-1),num_classes=21)

            graph = PairData(x_s=x_s, edge_index_s=edge_index_s, edge_weight_s=edge_weight_s,
                             x_t=x_t, edge_index_t=edge_index_t, edge_weight_t=edge_weight_t,
                             y = data.y)
            graph.edge_index=edge_index
            data_zip = {'graph':graph}
            torch.save(data_zip, osp.join(self.processed_dir, 'ZINC_BM_'+str(i+1)+'.pt'))
            i += 1

            # print(f'max_attr is {max_attr}')
            # print(f'min_attr is {min_attr}')
            # print(f'max_x_t is {max_x_t}')
            # print(f'min_x_t is {min_x_t}')
            # print(f'max_Y is {max_Y}')
            # print(f'min_Y is {min_Y}')
            # print(f'total gragh number is {i}')
###############################################################################
###############################################################################
class CIFAR_HG_BM_par1(Dataset):
    def __init__(self, root, dataset):

        self.root = root 
#         self.size = size
#         self.if_norm = if_norm # if normalize (standardize)
        self.dataset = dataset
        self.signal_distribution = torch.load(osp.join('CIFAR10_bm','signal_distribution.pt'))
        super().__init__(root)
        
    @property
    def processed_file_names(self):
        return ['CIFAR10_BM_'+str(fileidx+1)+'.pt' for fileidx in range(self.len())]
    
    def len(self):
        return self.dataset.len()
#         return len(self.processed_file_names)

    def get(self,idx):
        data_zip = torch.load(osp.join(self.processed_dir, 'CIFAR10_BM_'+str(idx+1)+'.pt'))
        data = data_zip['graph']
        data.num_node1 = data.x_t.shape[0]
        data.num_edge1 = data.x_s.shape[0]
        data.num_nodes = data.x_t.shape[0]
        # remove the dy/dx
        data.x_s = data.x_s[:,:-1]
        data.x_s = torch.cat([data.x_s, (data.x_t[:,:3][data.edge_index[0]]+data.x_t[:,:3][data.edge_index[1]])/2], dim=-1)
        return data
            
    def process(self):       
        i = 0
        loader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=1)
        for data in loader:
            # standardization
            data.x = (data.x-self.signal_distribution['x_mean']) / self.signal_distribution['x_std']
            data.edge_attr = (data.edge_attr-self.signal_distribution['edge_attr'][0]) / self.signal_distribution['edge_attr'][1]
            edge_index,edge_attr = to_undirected(data.edge_index, data.edge_attr, reduce='min')
            idx = edge_index[0]<edge_index[1]
            edge_index,edge_attr = edge_index[:,idx], edge_attr[idx]
            
            par1 = adj2par1(edge_index, data.x.shape[0], edge_index.shape[1]).to_dense()
            L0 = torch.matmul(par1, par1.T)
            lambda0, _ = torch.linalg.eigh(L0)
            L1 = torch.matmul(par1.T, par1)
            edge_index_t, edge_weight_t = dense_to_sparse(2*L0/lambda0.max())
            edge_index_s, edge_weight_s = dense_to_sparse(2*L1/lambda0.max())
            
            x_s = torch.abs(data.x[edge_index[0]]-data.x[edge_index[1]]) # edge signal
            x_s = torch.cat([x_s, edge_attr.view(-1,1)], dim=-1)
            x_s = torch.cat([x_s, (data.pos[edge_index[0]]+data.pos[edge_index[1]])/2], dim=-1)
            temp = (data.pos[edge_index[0]]-data.pos[edge_index[1]])
            temp = temp[:,1] / (temp[:,0]+1e-6)
            x_s = torch.cat([x_s, temp.view(-1,1)], dim=-1)
            x_t = torch.cat([data.x, data.pos], dim=-1)
            
            graph = PairData(x_s=x_s, edge_index_s=edge_index_s, edge_weight_s=edge_weight_s, 
                             x_t=x_t, edge_index_t=edge_index_t, edge_weight_t=edge_weight_t,
                             y = data.y)
            graph.edge_index=edge_index
            data_zip = {'graph':graph}
            torch.save(data_zip, osp.join(self.processed_dir, 'CIFAR10_BM_'+str(i+1)+'.pt'))
            i += 1
            
            
####################################################################################################
class CIFAR_HG(Dataset):
    def __init__(self, root, dataset, size=50000, if_norm=True):

        self.root = root 
        self.size = size
        self.if_norm = if_norm # if normalize (standardize)
        self.dataset = dataset
 
        data_zip = torch.load(osp.join(root, 'img_topology.pt'))
        
        lambda0 = data_zip['lambda0']
        self.edge_index = data_zip['edge_index']
        HL1 = 2 * data_zip['L1'] / lambda0.max()
        HL0 = 2 * data_zip['L0'] / lambda0.max()
        self.par1 = data_zip['par1']

        self.edge_index_t, self.edge_weight_t = dense_to_sparse(HL0)
        self.edge_index_s, self.edge_weight_s = dense_to_sparse(HL1)
        
        if if_norm:
            self.rgb_m = data_zip['rgb_m']
            self.rgb_s = data_zip['rgb_s']
        # self.edge_index_t, self.edge_weight_t = add_self_loops(self.edge_index_t, self.edge_weight_t, fill_value=-1.,num_nodes=268)
        # self.edge_index_t, self.edge_weight_t = coalesce(self.edge_index_t, self.edge_weight_t)
        super().__init__(root)

    @property
    def processed_file_names(self):
        return ['CIFAR100_'+str(fileidx+1)+'.pt' for fileidx in range(self.size)]
    
    def len(self):
        return self.size
#         return len(self.processed_file_names)

    def get(self,idx):
        data_zip = torch.load(osp.join(self.processed_dir, 'CIFAR100_'+str(idx+1)+'.pt'))
        data = data_zip['graph']
            
        # data.y = label
        data.edge_index_t, data.edge_weight_t = self.edge_index_t, self.edge_weight_t
        data.edge_index_s, data.edge_weight_s = self.edge_index_s, self.edge_weight_s
        # data.par1 = self.par1
            
        if self.if_norm:
            data.x_t = (data.x_t - self.rgb_m) / self.rgb_s

        data.x_s = torch.linalg.vector_norm(data.x_t[self.edge_index[0]]-data.x_t[self.edge_index[1]], dim=1) # edge signal
        return data
    
    def process(self):       
        # trainset = tv.datasets.CIFAR100(root='CIFAR100', train=True)
        # testset = tv.datasets.CIFAR100(root='CIFAR100', train=False)
        idx = 0
        for img, y in zip(self.dataset.data, self.dataset.targets):
                
            img = torch.tensor(img,dtype=torch.float)/255
            img = img.view(-1,3) # RGB channels (node signal)
            # edge = torch.linalg.vector_norm(img[edge_index[0]]-img[edge_index[1]], dim=1) # edge signal
            y = torch.tensor(y)
            # y = F.one_hot(torch.tensor(y),num_classes=100)

            graph = PairData(x_s=None, edge_index_s=None, edge_weight_s=None,
                                x_t=img, edge_index_t=None, edge_weight_t=None, y = y)
            # iq_mean, iq_std = 0, 1
            # iq = (torch.tensor(data['iq'][0].astype(np.float64))-iq_mean) / iq_std
            
            data_zip = {'graph':graph}
            torch.save(data_zip, osp.join(self.processed_dir, 'CIFAR100_'+str(idx+1)+'.pt'))
            idx += 1
            
            ##########################################
            
class CIFAR_HG_conn8_par1(Dataset):
    def __init__(self, root, dataset, size=50000, if_norm=True):

        self.root = root 
        self.size = size
        self.if_norm = if_norm # if normalize (standardize)
        self.dataset = dataset
 
        data_zip = torch.load(osp.join(root, 'img_topology_conn8.pt'))
        
        lambda0 = data_zip['lambda0']
        self.edge_index = data_zip['edge_index']
        HL1 = 2 * data_zip['L1'] / lambda0.max()
        HL0 = 2 * data_zip['L0'] / lambda0.max()
        self.par1 = data_zip['par1']

        self.edge_index_t, self.edge_weight_t = dense_to_sparse(HL0)
        self.edge_index_s, self.edge_weight_s = dense_to_sparse(HL1)
        
        if if_norm:
            self.rgb_m = data_zip['rgb_m']
            self.rgb_s = data_zip['rgb_s']
        # self.edge_index_t, self.edge_weight_t = add_self_loops(self.edge_index_t, self.edge_weight_t, fill_value=-1.,num_nodes=268)
        # self.edge_index_t, self.edge_weight_t = coalesce(self.edge_index_t, self.edge_weight_t)
        super().__init__(root)
        
    @property
    def processed_file_names(self):
        return ['CIFAR10_conn8_'+str(fileidx+1)+'.pt' for fileidx in range(self.size)]
    
    def len(self):
        return self.size
#         return len(self.processed_file_names)

    def get(self,idx):
        data_zip = torch.load(osp.join(self.processed_dir, 'CIFAR10_conn8_'+str(idx+1)+'.pt'))
        data = data_zip['graph']
            
        # data.y = label
        data.edge_index_t, data.edge_weight_t = self.edge_index_t, self.edge_weight_t
        data.edge_index_s, data.edge_weight_s = self.edge_index_s, self.edge_weight_s
        data.edge_index = self.edge_index
        data.x = torch.ones(data.x_t.shape[0],1)
        # data.par1 = self.par1
            
        if self.if_norm:
            data.x_t = (data.x_t - self.rgb_m) / self.rgb_s

        data.x_s = torch.abs(data.x_t[self.edge_index[0]]-data.x_t[self.edge_index[1]]) # edge signal
        data.x_s = torch.cat([data.x_s, data.x_t[self.edge_index[0]]+data.x_t[self.edge_index[1]]], dim=-1) # edge signal
        return data
            
    def process(self):       
        # trainset = tv.datasets.CIFAR100(root='CIFAR100', train=True)
        # testset = tv.datasets.CIFAR100(root='CIFAR100', train=False)
        idx = 0
        for img, y in zip(self.dataset.data, self.dataset.targets):
                
            img = torch.tensor(img,dtype=torch.float)/255
            img = img.view(-1,3) # RGB channels (node signal)
            # edge = torch.linalg.vector_norm(img[edge_index[0]]-img[edge_index[1]], dim=1) # edge signal
            y = torch.tensor(y)
            # y = F.one_hot(torch.tensor(y),num_classes=100)

            graph = PairData(x_s=None, edge_index_s=None, edge_weight_s=None,
                                x_t=img, edge_index_t=None, edge_weight_t=None, y = y)
            # iq_mean, iq_std = 0, 1
            # iq = (torch.tensor(data['iq'][0].astype(np.float64))-iq_mean) / iq_std
            
            data_zip = {'graph':graph}
            torch.save(data_zip, osp.join(self.processed_dir, 'CIFAR100_'+str(idx+1)+'.pt'))
            idx += 1
            
###############################################################################
class CIFAR_HG_conn8(Dataset):
    def __init__(self, root, dataset, size=50000, if_norm=True):

        self.root = root 
        self.size = size
        self.if_norm = if_norm # if normalize (standardize)
        self.dataset = dataset
 
        data_zip = torch.load(osp.join(root, 'img_topology_conn8.pt'))
        
        lambda0 = data_zip['lambda0']
        self.edge_index = data_zip['edge_index']
        HL1 = 2 * data_zip['L1'] / lambda0.max()
        HL0 = 2 * data_zip['L0'] / lambda0.max()
        self.par1 = data_zip['par1']

        self.edge_index_t, self.edge_weight_t = dense_to_sparse(HL0)
        self.edge_index_s, self.edge_weight_s = dense_to_sparse(HL1)
        
        if if_norm:
            self.rgb_m = data_zip['rgb_m']
            self.rgb_s = data_zip['rgb_s']
        # self.edge_index_t, self.edge_weight_t = add_self_loops(self.edge_index_t, self.edge_weight_t, fill_value=-1.,num_nodes=268)
        # self.edge_index_t, self.edge_weight_t = coalesce(self.edge_index_t, self.edge_weight_t)
        super().__init__(root)

    @property
    def processed_file_names(self):
        return ['CIFAR10_conn8_'+str(fileidx+1)+'.pt' for fileidx in range(self.size)]
    
    def len(self):
        return self.size
#         return len(self.processed_file_names)

    def get(self,idx):
        data_zip = torch.load(osp.join(self.processed_dir, 'CIFAR10_conn8_'+str(idx+1)+'.pt'))
        data = data_zip['graph']
            
        # data.y = label
        data.edge_index_t, data.edge_weight_t = self.edge_index_t, self.edge_weight_t
        data.edge_index_s, data.edge_weight_s = self.edge_index_s, self.edge_weight_s
        # data.par1 = self.par1
            
        if self.if_norm:
            data.x_t = (data.x_t - self.rgb_m) / self.rgb_s

        data.x_s = data.x_t[self.edge_index[0]]-data.x_t[self.edge_index[1]] # edge signal
        data.x_s = torch.cat([data.x_s, data.x_t[self.edge_index[0]]+data.x_t[self.edge_index[1]]], dim=-1) # edge signal
        return data
    
    def process(self):       
        # trainset = tv.datasets.CIFAR100(root='CIFAR100', train=True)
        # testset = tv.datasets.CIFAR100(root='CIFAR100', train=False)
        idx = 0
        for img, y in zip(self.dataset.data, self.dataset.targets):
                
            img = torch.tensor(img,dtype=torch.float)/255
            img = img.view(-1,3) # RGB channels (node signal)
            # edge = torch.linalg.vector_norm(img[edge_index[0]]-img[edge_index[1]], dim=1) # edge signal
            y = torch.tensor(y)
            # y = F.one_hot(torch.tensor(y),num_classes=100)

            graph = PairData(x_s=None, edge_index_s=None, edge_weight_s=None,
                                x_t=img, edge_index_t=None, edge_weight_t=None, y = y)
            # iq_mean, iq_std = 0, 1
            # iq = (torch.tensor(data['iq'][0].astype(np.float64))-iq_mean) / iq_std
            
            data_zip = {'graph':graph}
            torch.save(data_zip, osp.join(self.processed_dir, 'CIFAR10_conn8_'+str(idx+1)+'.pt'))
            idx += 1


###############################################################################
#########################  Spatial Temporal Graph  ############################
###############################################################################


class Multi_hodge_Laplacian(Dataset):
    def __init__(self, root, lap_root=None, size=7693, class_idx=0, if_fc_norm=False,
                 if_spool=False, if_nspool=False, if_max_edge_conn=False, if_spool1=False):
        # class_idx:
        #      0. IQ
        #      1. gender
        #      2. age
        self.root = root 
        self.size = size
        self.class_idx = class_idx
        self.if_fc_norm = if_fc_norm # if use fisher z transform
        self.if_spool = if_spool
        self.if_nspool = if_nspool

        laplacians = loadmat(lap_root)
        
        if self.if_nspool:
            data_zip = torch.load(osp.join(osp.join('Data', 'ABCD_HL0.pt')))
            HL0_1 = data_zip['L0_1']
            HL0 = data_zip['L0_0']
            idx_dic_t = data_zip['idx_dic']
            self.idx_dic_t = idx_dic_t
            self.edge_index_t1, self.edge_weight_t1 = dense_to_sparse(HL0_1)
            self.edge_index_t1, self.edge_weight_t1 = add_self_loops(self.edge_index_t1, self.edge_weight_t1, fill_value=-1.,num_nodes=134)
            self.edge_index_t1, self.edge_weight_t1 = coalesce(self.edge_index_t1, self.edge_weight_t1)
        else:
            HL0 = torch.tensor(laplacians['HL0'])
 
        if self.if_spool:
            if if_max_edge_conn:
                data_zip = torch.load(osp.join(osp.join('Data', 'ABCD_HL1_maxconn.pt')))
                if if_spool1:
                    HL1_2 = data_zip['L1_2']
                    self.edge_index_s2, self.edge_weight_s2 = dense_to_sparse(HL1_2)
                    self.idx_dic1 = data_zip['idx_dic1']
            else:
                data_zip = torch.load(osp.join(osp.join('Data', 'ABCD_HLs.pt')))
            HL1_1 = data_zip['L1_1']
            HL1 = data_zip['L1_0']
            idx_dic = data_zip['idx_dic']
            self.idx_dic = idx_dic
            self.edge_index_s1, self.edge_weight_s1 = dense_to_sparse(HL1_1)
        else:
            HL1 = torch.tensor(laplacians['HL1'])
            
        # if if_incidence:
        #     matpath = '/home/jinghan/Documents/MATLAB/Hodge_Laplacian/data/ABCD7693_25_avg_incidence_matrix.mat'
        #     laplacians = loadmat(matpath)
        #     self.incmat = torch.tensor(laplacians['c'])    

        self.edge_index_t, self.edge_weight_t = dense_to_sparse(HL0)
        self.edge_index_s, self.edge_weight_s = dense_to_sparse(HL1)
        self.edge_index_t, self.edge_weight_t = add_self_loops(self.edge_index_t, self.edge_weight_t, fill_value=-1.,num_nodes=268)
        self.edge_index_t, self.edge_weight_t = coalesce(self.edge_index_t, self.edge_weight_t)
        super().__init__(root)

    @property
    def processed_file_names(self):
        return ['ABCD'+str(fileidx+1)+'.pt' for fileidx in range(self.size)]
    
    def len(self):
        return self.size
#         return len(self.processed_file_names)

    def get(self,idx):
        data_zip = torch.load(osp.join(self.processed_dir, 'ABCD'+str(idx+1)+'.pt'))
        data = data_zip['graph']
            
        # data.y = label
        data.edge_index_t, data.edge_weight_t = self.edge_index_t, self.edge_weight_t
        data.edge_index_s, data.edge_weight_s = self.edge_index_s, self.edge_weight_s           
            
        if self.if_fc_norm:
            data.x_s = 0.5*torch.log((1+data.x_s)/(1-data.x_s))
            # data.x_s = (data.x_s-self.fc_mean.view(-1,1)) / self.fc_std.view(-1,1)
        
        if self.if_spool:
            data.x_s = data.x_s[self.idx_dic] # permute the edge order

        if self.if_nspool:
            data.x_t = data.x_t[self.idx_dic_t] # permute the node order
            
        # if self.if_incidence:
        #     return data, self.incmat
        # else:
        return data
    
    def process(self):       
        for idx in range(self.size):
            data = loadmat(osp.join(self.raw_dir, 'ABCD'+str(idx+1)+'.mat'))
            # fc: Functional connectivity (vectorized by a template)
            # ft: Functional time series
            fc, ft = torch.tensor(data['fc']), torch.tensor(data['ftime_series'])
            iq = torch.tensor(data['iq'][0].astype(np.float64)) 
            graph = PairData(x_s=fc, edge_index_s=None, edge_weight_s=None,
                                x_t=ft, edge_index_t=None, edge_weight_t=None, y = iq)
            # iq_mean, iq_std = 0, 1
            # iq = (torch.tensor(data['iq'][0].astype(np.float64))-iq_mean) / iq_std
            
            data_zip = {'graph':graph}
            torch.save(data_zip, osp.join(self.processed_dir, 'ABCD'+str(idx+1)+'.pt'))


###############################################################################
class Multi_hodge_Laplacian_Tnorm(Dataset):
    def __init__(self, root, lap_root=None, size=7693, class_idx=0, if_fc_norm=False,
                 if_spool=False, if_nspool=False, if_max_edge_conn=False, if_spool1=False,
                 if_tnorm=True):
        # class_idx:
        #      0. IQ
        #      1. gender
        #      2. age
        self.root = root 
        self.size = size
        self.class_idx = class_idx
        self.if_fc_norm = if_fc_norm # if use fisher z transform
        self.if_spool = if_spool
        self.if_nspool = if_nspool

        laplacians = loadmat(lap_root)
        
        if self.if_nspool:
            data_zip = torch.load(osp.join(osp.join('Data', 'ABCD_HL0.pt')))
            HL0_1 = data_zip['L0_1']
            HL0 = data_zip['L0_0']
            idx_dic_t = data_zip['idx_dic']
            self.idx_dic_t = idx_dic_t
            self.edge_index_t1, self.edge_weight_t1 = dense_to_sparse(HL0_1)
            self.edge_index_t1, self.edge_weight_t1 = add_self_loops(self.edge_index_t1, self.edge_weight_t1, fill_value=-1.,num_nodes=134)
            self.edge_index_t1, self.edge_weight_t1 = coalesce(self.edge_index_t1, self.edge_weight_t1)
        else:
            HL0 = torch.tensor(laplacians['HL0'])
 
        if self.if_spool:
            if if_max_edge_conn:
                data_zip = torch.load(osp.join(osp.join('Data', 'ABCD_HL1_maxconn.pt')))
                if if_spool1:
                    HL1_2 = data_zip['L1_2']
                    self.edge_index_s2, self.edge_weight_s2 = dense_to_sparse(HL1_2)
                    self.idx_dic1 = data_zip['idx_dic1']
            else:
                data_zip = torch.load(osp.join(osp.join('Data', 'ABCD_HLs.pt')))
            HL1_1 = data_zip['L1_1']
            HL1 = data_zip['L1_0']
            idx_dic = data_zip['idx_dic']
            self.idx_dic = idx_dic
            self.edge_index_s1, self.edge_weight_s1 = dense_to_sparse(HL1_1)
        else:
            HL1 = torch.tensor(laplacians['HL1'])
            
        # if if_incidence:
        #     matpath = '/home/jinghan/Documents/MATLAB/Hodge_Laplacian/data/ABCD7693_25_avg_incidence_matrix.mat'
        #     laplacians = loadmat(matpath)
        #     self.incmat = torch.tensor(laplacians['c'])    

        self.edge_index_t, self.edge_weight_t = dense_to_sparse(HL0)
        self.edge_index_s, self.edge_weight_s = dense_to_sparse(HL1)
        self.edge_index_t, self.edge_weight_t = add_self_loops(self.edge_index_t, self.edge_weight_t, fill_value=-1.,num_nodes=268)
        self.edge_index_t, self.edge_weight_t = coalesce(self.edge_index_t, self.edge_weight_t)
        super().__init__(root)

    @property
    def processed_file_names(self):
        return ['ABCD'+str(fileidx+1)+'.pt' for fileidx in range(self.size)]
    
    def len(self):
        return self.size
#         return len(self.processed_file_names)

    def get(self,idx):
        data_zip = torch.load(osp.join(self.processed_dir, 'ABCD'+str(idx+1)+'.pt'))
        data = data_zip['graph']
            
        # data.y = label
        data.edge_index_t, data.edge_weight_t = self.edge_index_t, self.edge_weight_t
        data.edge_index_s, data.edge_weight_s = self.edge_index_s, self.edge_weight_s
        
        if self.class_idx == 2:
            temp = loadmat(osp.join(self.raw_dir, 'ABCD'+str(idx+1)+'.mat'))
            data.y = torch.tensor(temp['gf'][0].astype(np.float64))
            
        if self.if_fc_norm:
            data.x_s = 0.5*torch.log((1+data.x_s)/(1-data.x_s))
            # data.x_s = (data.x_s-self.fc_mean.view(-1,1)) / self.fc_std.view(-1,1)
        
        if self.if_spool:
            data.x_s = data.x_s[self.idx_dic] # permute the edge order

        if self.if_nspool:
            data.x_t = data.x_t[self.idx_dic_t] # permute the node order
            
        # if self.if_incidence:
        #     return data, self.incmat
        # else:
        return data
    
    def process(self):       
        for idx in range(self.size):
  
            if self.class_idx == 0:
                data = loadmat(osp.join(self.raw_dir, 'ABCD'+str(idx+1)+'.mat'))
                iq = torch.tensor(data['iq'][0].astype(np.float64))
            elif self.class_idx == 1:
                data = loadmat(osp.join(self.raw_dir, 'OASIS'+str(idx+1)+'.mat'))
                iq = torch.tensor(data['age'][0].astype(np.float64))
                
            # fc: Functional connectivity (vectorized by a template)
            # ft: Functional time series
            fc, ft = torch.tensor(data['fc']), torch.tensor(data['standard_ftime_series'])
                
            graph = PairData(x_s=fc, edge_index_s=None, edge_weight_s=None,
                                x_t=ft, edge_index_t=None, edge_weight_t=None, y = iq)
            # iq_mean, iq_std = 0, 1
            # iq = (torch.tensor(data['iq'][0].astype(np.float64))-iq_mean) / iq_std
            
            data_zip = {'graph':graph}
            torch.save(data_zip, osp.join(self.processed_dir, 'ABCD'+str(idx+1)+'.pt'))
            
            
###############################################################################
class Multi_hodge_Laplacian_Maxconn(Dataset):
    def __init__(self, root, size=7693, class_idx=0, if_fc_norm=True, fold=None):
        # class_idx:
        #      0. IQ
        #      1. gender
        #      2. age
        self.root = root 
        self.size = size
        self.class_idx = class_idx
        self.if_fc_norm = if_fc_norm # if use fisher z transform
        
        if self.class_idx == 2:
            GA = loadmat('/home/jinghan/Documents/MATLAB/Hodge_Laplacian/data/ABCD/GA7693.mat')
            info = loadmat('/home/jinghan/Documents/MATLAB/Hodge_Laplacian/data/ABCD/TrainValidTest_r1.mat')
            self.GA = torch.tensor(GA['GeneralAb'].astype(np.float64))
            self.GA_mean = torch.tensor(info['GA_mean'][0][fold])
            self.GA_std = torch.tensor(info['GA_std'][0][fold])

 
        data_zip = torch.load(osp.join(osp.join('Data', 'ABCD_maxconn.pt')))

        HL1_1 = data_zip['L1_1']
        HL1 = data_zip['L1_0']
        HL0_1 = data_zip['L0_1']
        HL0 = data_zip['L0_0']
        idx_dic_t = data_zip['idx_dic_n']
        idx_dic = data_zip['idx_dic']
        
        self.idx_dic = idx_dic
        self.edge_index_s1, self.edge_weight_s1 = dense_to_sparse(HL1_1)
        self.idx_dic_t = idx_dic_t
        self.edge_index_t1, self.edge_weight_t1 = dense_to_sparse(HL0_1)
        self.edge_index_t1, self.edge_weight_t1 = add_self_loops(self.edge_index_t1, self.edge_weight_t1, fill_value=-1.,num_nodes=134)
        self.edge_index_t1, self.edge_weight_t1 = coalesce(self.edge_index_t1, self.edge_weight_t1)
        self.edge_index_t, self.edge_weight_t = dense_to_sparse(HL0)
        self.edge_index_s, self.edge_weight_s = dense_to_sparse(HL1)
        self.edge_index_t, self.edge_weight_t = add_self_loops(self.edge_index_t, self.edge_weight_t, fill_value=-1.,num_nodes=268)
        self.edge_index_t, self.edge_weight_t = coalesce(self.edge_index_t, self.edge_weight_t)
        super().__init__(root)

    @property
    def processed_file_names(self):
        return ['ABCD'+str(fileidx+1)+'.pt' for fileidx in range(self.size)]
    
    def len(self):
        return self.size
#         return len(self.processed_file_names)

    def get(self,idx):
        data_zip = torch.load(osp.join(self.processed_dir, 'ABCD'+str(idx+1)+'.pt'))
        data = data_zip['graph']
            
        # data.y = label
        data.edge_index_t, data.edge_weight_t = self.edge_index_t, self.edge_weight_t
        data.edge_index_s, data.edge_weight_s = self.edge_index_s, self.edge_weight_s
        
        if self.class_idx == 2:
            data.y = (self.GA[idx]-self.GA_mean)/self.GA_std#/9
            
        if self.if_fc_norm:
            data.x_s = 0.5*torch.log((1+data.x_s)/(1-data.x_s))
            # data.x_s = (data.x_s-self.fc_mean.view(-1,1)) / self.fc_std.view(-1,1)

        data.x_s = data.x_s[self.idx_dic] # permute the edge order
        data.x_t = data.x_t[self.idx_dic_t] # permute the node order
            
        # if self.if_incidence:
        #     return data, self.incmat
        # else:
        return data
    
    def process(self):       
        for idx in range(self.size):
  
            if self.class_idx == 0:
                data = loadmat(osp.join(self.raw_dir, 'ABCD'+str(idx+1)+'.mat'))
                iq = torch.tensor(data['iq'][0].astype(np.float64))
            elif self.class_idx == 1:
                data = loadmat(osp.join(self.raw_dir, 'OASIS'+str(idx+1)+'.mat'))
                iq = torch.tensor(data['age'][0].astype(np.float64))
                
            # fc: Functional connectivity (vectorized by a template)
            # ft: Functional time series
            fc, ft = torch.tensor(data['fc']), torch.tensor(data['standard_ftime_series'])
                
            graph = PairData(x_s=fc, edge_index_s=None, edge_weight_s=None,
                                x_t=ft, edge_index_t=None, edge_weight_t=None, y = iq)
            # iq_mean, iq_std = 0, 1
            # iq = (torch.tensor(data['iq'][0].astype(np.float64))-iq_mean) / iq_std
            
            data_zip = {'graph':graph}
            torch.save(data_zip, osp.join(self.processed_dir, 'ABCD'+str(idx+1)+'.pt'))

###############################################################################
##############################  FC profile Graph  #############################
###############################################################################

class ABCD_EHGNN(Dataset):
    def __init__(self, root, size=7693):
        # class_idx:
        #      0. IQ
        #      1. gender
        #      2. age
        self.root = root 
        self.size = size
        super().__init__(root)

    @property
    def processed_file_names(self):
        return ['ABCD'+str(fileidx+1)+'.pt' for fileidx in range(self.size)]
    
    def len(self):
        return self.size
#         return len(self.processed_file_names)

    def get(self,idx):
        data = torch.load(osp.join(self.processed_dir, 'ABCD'+str(idx+1)+'.pt'))
        data.edge_attr = data.edge_attr.view(-1,1)
        data.x = data.x.to(torch.double)
        return data
    
    def process(self):
        for idx in range(self.size):
            data = loadmat(osp.join(self.raw_dir, 'ABCD'+str(idx+1)+'.mat'))
            # fc: Functional connectivity (vectorized by a template)
            # ft: Functional time series
            fc = torch.tensor(data['fc'])
            edge_index, edge_attr = dense_to_sparse(fc)
            iq = torch.tensor(data['iq'][0].astype(np.float64))            
            graph = Data(x=fc, edge_index=edge_index, edge_attr=edge_attr, y=iq)
            
            # iq = (torch.tensor(data['iq'][0].astype(np.float64))-iq_mean) / iq_std
            # data_zip = {'graph':graph}
            torch.save(graph, osp.join(self.processed_dir, 'ABCD'+str(idx+1)+'.pt'))

###############################################################################
class ABCD_braingnn(Dataset):
    def __init__(self, root, size=7693):
        # class_idx:
        #      0. IQ
        #      1. gender
        #      2. age
        self.root = root 
        self.size = size
        super().__init__(root)
        
        data_zip = torch.load(osp.join(osp.join('Data', 'ABCD_maxconn.pt')))

        HL1_1 = data_zip['L1_1']
        HL1 = data_zip['L1_0']
        HL0_1 = data_zip['L0_1']
        HL0 = data_zip['L0_0']
        idx_dic_t = data_zip['idx_dic_n']
        idx_dic = data_zip['idx_dic']
        
        self.idx_dic = idx_dic
        self.idx_dic_t = idx_dic_t
        self.edge_index_t, self.edge_weight_t = dense_to_sparse(HL0)

        
    @property
    def processed_file_names(self):
        return ['ABCD'+str(fileidx+1)+'.pt' for fileidx in range(self.size)]
    
    def len(self):
        return self.size
#         return len(self.processed_file_names)

    def get(self,idx):
        data = torch.load(osp.join(self.processed_dir, 'ABCD'+str(idx+1)+'.pt'))
        
        data.x = data.x.to(torch.double)
        data.pos = torch.eye(268).to(torch.double)
        
        data.edge_index_t, data.edge_weight_t = self.edge_index_t, self.edge_weight_t
        data.x = data.x[self.idx_dic_t] # permute the node order
        
        data.edge_weight = torch.ones(data.edge_index.shape[-1]).to(torch.double)
        return data
    
    def process(self):
        for idx in range(self.size):
            data = loadmat(osp.join(self.raw_dir, 'ABCD'+str(idx+1)+'.mat'))
            # fc: Functional connectivity (vectorized by a template)
            # ft: Functional time series
            fc, pc_mask, ft = torch.tensor(data['fc']), torch.tensor(data['pc_mask']), torch.tensor(data['ftime_series'])
            edge_index, _ = dense_to_sparse(pc_mask)
            iq = torch.tensor(data['iq'][0].astype(np.float64))            
            graph = Data(x=fc, edge_index=edge_index, y=iq)
            
            # iq = (torch.tensor(data['iq'][0].astype(np.float64))-iq_mean) / iq_std
            # data_zip = {'graph':graph}
            torch.save(graph, osp.join(self.processed_dir, 'ABCD'+str(idx+1)+'.pt'))
            
            
###############################################################################
#########################  Spatial Temporal Graph  ############################
###############################################################################
