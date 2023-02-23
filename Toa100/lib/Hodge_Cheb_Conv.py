#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 21:58:34 2022

@author: jinghan
"""
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import OptTensor
from torch_geometric.nn.pool import graclus, max_pool
from torch_geometric.data import Data, Batch
from torch_scatter import scatter
from torch_geometric.utils import add_self_loops, dense_to_sparse
from typing import Callable, Optional, Tuple, Union

from torch_scatter import scatter_add, scatter_max

from torch_geometric.utils import softmax
from torch_geometric.utils import unbatch_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes
import torch_geometric.utils as ut
from scipy.sparse.linalg import eigsh

###############################################################################
def weighted_mse_loss(y, target):
    return torch.sum(torch.exp(target.abs()) * (y-target)**2)


def unbatch_edge_attr(edge_index: Tensor, edge_attr: Tensor, batch: Tensor):
    deg = ut.degree(batch, dtype=torch.int64)
    ptr = torch.cat([deg.new_zeros(1), deg.cumsum(dim=0)[:-1]], dim=0)

    edge_batch = batch[edge_index[0]]
    edge_index = edge_index - ptr[edge_batch]
    sizes = ut.degree(edge_batch, dtype=torch.int64).cpu().tolist()
    return edge_index.split(sizes, dim=1), edge_attr.split(sizes, dim=0)

###############################################################################
######################## Graph Pooling ########################################
###############################################################################
def adj2par1(edge_index, num_node, num_edge):
    col_idx = torch.cat([torch.arange(edge_index.shape[1]),torch.arange(edge_index.shape[1])]
                        ,dim=-1).to(edge_index.device)
    row_idx = torch.cat([edge_index[0],edge_index[1]], dim=-1).to(edge_index.device)
    val = torch.cat([edge_index[0].new_full(edge_index[0].shape,-1),
                     edge_index[0].new_full(edge_index[0].shape,1)],dim=-1).to(torch.float)
    par1_sparse = torch.sparse.FloatTensor(torch.cat([row_idx, col_idx], dim=-1).view(2,-1),
                                           val,torch.Size([num_node, num_edge]))
    return par1_sparse

def par2adj(par1):
    a = par1.to_sparse()
    _, perm = a.indices()[1].sort(dim=-1, descending=False)
    return a.indices()[0][perm].view(-1,2).T


#######################################################################################3
class JointPooling_Node_Res(torch.nn.Module):
    def __init__(
        self,
        in_channels_n: int,
        in_channels_e: int,
        ratio: Union[float, int] = 0.5,
        K: int = 2,
        nonlinearity: Callable = torch.sigmoid,
        **kwargs,
    ):
        super().__init__()
        self.in_channels_n = in_channels_n
        self.in_channels_e = in_channels_e
        self.ratio = ratio
        self.gnn_n = HodgeLaguerreConv(in_channels_n, 1, K=K)
        self.gnn_e = HodgeLaguerreConv(in_channels_e, 1, K=K)
        self.nonlinearity = nonlinearity

        self.reset_parameters()

    def reset_parameters(self):
        self.gnn_n.reset_parameters()
        self.gnn_e.reset_parameters()


    def forward(
        self,
        x_n: Tensor,
        edge_index_n: Tensor,
        edge_weight_n: Tensor,
        x_e: Tensor,
        edge_index_e: Tensor,
        edge_weight_e: Tensor,
        edge_index: Tensor,
        n_batch: Tensor,
        e_batch: Tensor,
        x_n0: Tensor,
        x_e0: Tensor,
    ):
        """"""

        score_n = self.gnn_n(x_n, edge_index_n, edge_weight_n).view(-1)
        score_e = self.gnn_e(x_e, edge_index_e, edge_weight_e).view(-1)
        score_n = self.nonlinearity(score_n)
        score_e = self.nonlinearity(score_e)
        x_n = x_n * score_n.view(-1, 1)
        x_e = x_e * score_e.view(-1, 1)

        num_nodes = scatter_add(n_batch.new_ones(x_n.size(0)), n_batch, dim=0) # num of nodes in each sample
        num_edges = scatter_add(e_batch.new_ones(x_e.size(0)), e_batch, dim=0) # num of edges in each sample
        batch_size, max_num_nodes = num_nodes.size(0), int(num_nodes.max())
        # print(edge_index)
        datas = unbatch_edge_index(edge_index, n_batch)
        score_n, score_e = ut.unbatch(score_n,n_batch), ut.unbatch(score_e,e_batch)
        x_n, x_e = ut.unbatch(x_n,n_batch), ut.unbatch(x_e,e_batch)
        x_n0, x_e0 = ut.unbatch(x_n0,n_batch), ut.unbatch(x_e0,e_batch)
        ngraph_list, egraph_list, par_list = [], [], []
        x_n0_list, x_e0_list = [],[]

        for i,ei in enumerate(datas):
            par_i = adj2par1(ei, num_nodes[i], num_edges[i])
#             print(par_i.shape)
            x_ni = score_n[i]
            x_ei = score_e[i]
            signal_ni = x_n[i]# * x_ni.view(-1,1)
            signal_ei = x_e[i]# * x_ei.view(-1,1)
            x_n0_i, x_e0_i = x_n0[i], x_e0[i]
            
            x_ni = x_ni + 0.125*torch.sparse.mm(par_i.abs(), x_ei.view(-1,1)).view(-1)
            _, perm_ni = x_ni.sort(dim=-1, descending=True)
            k_ni = (float(self.ratio) * num_nodes[i].to(x_ni.dtype)).ceil().to(torch.long)
            perm_ni = perm_ni[torch.arange(k_ni, device=x_ni.device)]
            mask_ni = torch.zeros(num_nodes[i]).to(x_ni.device)
            
            mask_ni[perm_ni] = 1
            mask_ei = torch.sparse.mm(par_i.abs().transpose(0,1),
                                      (1-mask_ni.view(1,-1)).T).T.view(-1)
            
            par_i = par_i.to_dense()
            par_i = par_i[mask_ni>0]
            par_i = par_i.T[mask_ei==0].T
            temp = par_i.to_sparse()
            L0_i = torch.sparse.mm(temp, par_i.T)
            L1_i = torch.sparse.mm(temp.transpose(0,1), par_i)

            edge_index_t, edge_weight_t = dense_to_sparse(L0_i)
            edge_index_s, edge_weight_s = dense_to_sparse(L1_i)
            
            # scipy sparse eigenvalue decomposition
            sci_sparse_t = ut.to_scipy_sparse_matrix(edge_index_t, edge_weight_t, num_nodes=L0_i.shape[0])
#             sci_sparse_s = ut.to_scipy_sparse_matrix(edge_index_s, edge_weight_s, num_nodes=L1_i.shape[0])
            max_lambda = eigsh(sci_sparse_t, k=1, which='LM', return_eigenvectors=False)[0]
            edge_weight_t = 2*edge_weight_t / max_lambda
            edge_weight_s = 2*edge_weight_s / max_lambda
            
            ngraph_list.append(Data(x=signal_ni[mask_ni>0], edge_index=edge_index_t, edge_weight=edge_weight_t, num_nodeas=L0_i.shape[0]))
            egraph_list.append(Data(x=signal_ei[mask_ei==0], edge_index=edge_index_s, edge_weight=edge_weight_s, num_nodeas=L1_i.shape[0]))
            # print(par_i.shape)
            par_list.append(Data(x=torch.ones(L0_i.shape[0],1), edge_index=par2adj(par_i), num_nodeas=L0_i.shape[0]))
            x_n0_list.append(x_n0_i[mask_ni>0])
            x_e0_list.append(x_e0_i[mask_ei==0])
            
                
        new_ndata = Batch.from_data_list(ngraph_list)
        new_edata = Batch.from_data_list(egraph_list)
        new_par = Batch.from_data_list(par_list)
        x_n0 = torch.cat(x_n0_list, dim=0)
        x_e0 = torch.cat(x_e0_list, dim=0)
        # e_batch = torch.cat([torch.ones(g.x.shape[0], device=g.x.device) for idx,g in enumerate(egraph_list)], dim=0).to(torch.long)
        # print(e_batch.shape, new_edata.x.shape)

        return [new_ndata.x, new_ndata.edge_index, new_ndata.edge_weight, 
                new_ndata.batch, new_edata.x, new_edata.edge_index, new_edata.edge_weight, 
                new_edata.batch, new_par.edge_index, x_n0, x_e0]

            
################################################################################

class JointPooling_Node(torch.nn.Module):
    def __init__(
        self,
        in_channels_n: int,
        in_channels_e: int,
        ratio: Union[float, int] = 0.5,
        K: int = 2,
        nonlinearity: Callable = torch.sigmoid,
        **kwargs,
    ):
        super().__init__()
        self.in_channels_n = in_channels_n
        self.in_channels_e = in_channels_e
        self.ratio = ratio
        self.gnn_n = HodgeLaguerreConv(in_channels_n, 1, K=K)
        self.gnn_e = HodgeLaguerreConv(in_channels_e, 1, K=K)
        self.nonlinearity = nonlinearity

        self.reset_parameters()

    def reset_parameters(self):
        self.gnn_n.reset_parameters()
        self.gnn_e.reset_parameters()


    def forward(
        self,
        x_n: Tensor,
        edge_index_n: Tensor,
        edge_weight_n: Tensor,
        x_e: Tensor,
        edge_index_e: Tensor,
        edge_weight_e: Tensor,
        edge_index: Tensor,
        n_batch: OptTensor = None,
        e_batch: OptTensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, OptTensor, OptTensor, OptTensor]:
        """"""

        score_n = self.gnn_n(x_n, edge_index_n, edge_weight_n).view(-1)
        score_e = self.gnn_e(x_e, edge_index_e, edge_weight_e).view(-1)
        score_n = self.nonlinearity(score_n)
        score_e = self.nonlinearity(score_e)
        x_n = x_n * score_n.view(-1, 1)
        x_e = x_e * score_e.view(-1, 1)

        num_nodes = scatter_add(n_batch.new_ones(x_n.size(0)), n_batch, dim=0) # num of nodes in each sample
        num_edges = scatter_add(e_batch.new_ones(x_e.size(0)), e_batch, dim=0) # num of edges in each sample
        batch_size, max_num_nodes = num_nodes.size(0), int(num_nodes.max())
        # print(edge_index)
        datas = unbatch_edge_index(edge_index, n_batch)
        score_n, score_e = ut.unbatch(score_n,n_batch), ut.unbatch(score_e,e_batch)
        x_n, x_e = ut.unbatch(x_n,n_batch), ut.unbatch(x_e,e_batch)
        ngraph_list, egraph_list, par_list = [], [], []

        for i,ei in enumerate(datas):
            par_i = adj2par1(ei, num_nodes[i], num_edges[i])
#             print(par_i.shape)
            x_ni = score_n[i]
            x_ei = score_e[i]
            signal_ni = x_n[i]# * x_ni.view(-1,1)
            signal_ei = x_e[i]# * x_ei.view(-1,1)
            
            x_ni = x_ni + torch.sparse.mm(par_i.abs(), x_ei.view(-1,1)).view(-1)
            _, perm_ni = x_ni.sort(dim=-1, descending=True)
            k_ni = (float(self.ratio) * num_nodes[i].to(x_ni.dtype)).ceil().to(torch.long)
            perm_ni = perm_ni[torch.arange(k_ni, device=x_ni.device)]
            mask_ni = torch.zeros(num_nodes[i]).to(x_ni.device)
            
            mask_ni[perm_ni] = 1
            mask_ei = torch.sparse.mm(par_i.abs().transpose(0,1),
                                      (1-mask_ni.view(1,-1)).T).T.view(-1)
            
            par_i = par_i.to_dense()
            par_i = par_i[mask_ni>0]
            par_i = par_i.T[mask_ei==0].T
            temp = par_i.to_sparse()
            L0_i = torch.sparse.mm(temp, par_i.T)
            L1_i = torch.sparse.mm(temp.transpose(0,1), par_i)

            edge_index_t, edge_weight_t = dense_to_sparse(L0_i)
            edge_index_s, edge_weight_s = dense_to_sparse(L1_i)
            
            # scipy sparse eigenvalue decomposition
            sci_sparse_t = ut.to_scipy_sparse_matrix(edge_index_t, edge_weight_t, num_nodes=L0_i.shape[0])
#             sci_sparse_s = ut.to_scipy_sparse_matrix(edge_index_s, edge_weight_s, num_nodes=L1_i.shape[0])
            max_lambda = eigsh(sci_sparse_t, k=1, which='LM', return_eigenvectors=False)[0]
            edge_weight_t = 2*edge_weight_t / max_lambda
            edge_weight_s = 2*edge_weight_s / max_lambda
            
            ngraph_list.append(Data(x=signal_ni[mask_ni>0], edge_index=edge_index_t, edge_weight=edge_weight_t, num_nodeas=L0_i.shape[0]))
            egraph_list.append(Data(x=signal_ei[mask_ei==0], edge_index=edge_index_s, edge_weight=edge_weight_s, num_nodeas=L1_i.shape[0]))
            # print(par_i.shape)
            par_list.append(Data(x=torch.ones(L0_i.shape[0],1), edge_index=par2adj(par_i), num_nodeas=L0_i.shape[0]))
                
        new_ndata = Batch.from_data_list(ngraph_list)
        new_edata = Batch.from_data_list(egraph_list)
        new_par = Batch.from_data_list(par_list)
        # e_batch = torch.cat([torch.ones(g.x.shape[0], device=g.x.device) for idx,g in enumerate(egraph_list)], dim=0).to(torch.long)
        # print(e_batch.shape, new_edata.x.shape)

        return new_ndata.x, new_ndata.edge_index, new_ndata.edge_weight, new_ndata.batch, new_edata.x, new_edata.edge_index, new_edata.edge_weight, new_edata.batch, new_par.edge_index

###############################################################################
# calculate two mask

class JointPooling(torch.nn.Module):

    def __init__(
        self,
        in_channels_n: int,
        in_channels_e: int,
        ratio: Union[float, int] = 0.5,
        K: int = 2,
        nonlinearity: Callable = torch.tanh,
        **kwargs,
    ):
        super().__init__()

        self.in_channels_n = in_channels_n
        self.in_channels_e = in_channels_e
        self.ratio = ratio
        self.gnn_n = HodgeLaguerreConv(in_channels_n, 1, K=K)
        self.gnn_e = HodgeLaguerreConv(in_channels_e, 1, K=K)
        self.nonlinearity = nonlinearity

        self.reset_parameters()

    def reset_parameters(self):
        self.gnn_n.reset_parameters()
        self.gnn_e.reset_parameters()


    def forward(
        self,
        x_n: Tensor,
        edge_index_n: Tensor,
        edge_weight_n: Tensor,
        x_e: Tensor,
        edge_index_e: Tensor,
        edge_weight_e: Tensor,
        edge_index: Tensor,
        n_batch: OptTensor = None,
        e_batch: OptTensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, OptTensor, OptTensor, OptTensor]:
        """"""

        score_n = self.gnn_n(x_n, edge_index_n, edge_weight_n).view(-1)
        score_e = self.gnn_e(x_e, edge_index_e, edge_weight_e).view(-1)
        score_n = self.nonlinearity(score_n)
        score_e = self.nonlinearity(score_e)
        x_n = x_n * score_n.view(-1, 1)
        x_e = x_e * score_e.view(-1, 1)

        num_nodes = scatter_add(n_batch.new_ones(x_n.size(0)), n_batch, dim=0) # num of nodes in each sample
        num_edges = scatter_add(e_batch.new_ones(x_e.size(0)), e_batch, dim=0) # num of edges in each sample
        batch_size, max_num_nodes = num_nodes.size(0), int(num_nodes.max())
        # print(edge_index)
        datas = unbatch_edge_index(edge_index, n_batch)
        ngraph_list, egraph_list, par_list = [], [], []

        for i,ei in enumerate(datas):
            par_i = adj2par1(ei, num_nodes[i], num_edges[i]).to(torch.float)
            # print(par_i.shape)
            x_ni = score_n[n_batch==i]
            x_ei = score_e[e_batch==i]
            signal_ni = x_n[n_batch==i]# * x_ni.view(-1, 1)
            signal_ei = x_e[e_batch==i]# * x_ei.view(-1,1)
            x_ni += torch.matmul(par_i.abs(), x_ei)
            x_ei += torch.matmul(par_i.abs().T, x_ni)

            _, perm_ni = x_ni.sort(dim=-1, descending=True)
            _, perm_ei = x_ei.sort(dim=-1, descending=True)
            k_ni = (float(self.ratio) * num_nodes[i].to(x_ni.dtype)).ceil().to(torch.long)
            k_ei = (float(self.ratio) * num_edges[i].to(x_ei.dtype)).ceil().to(torch.long)

            perm_ni = perm_ni[torch.arange(k_ni, device=x_ni.device)]
            perm_ei = perm_ei[torch.arange(k_ei, device=x_ni.device)]
            mask_ni = torch.zeros(num_nodes[i]).to(x_ni.device)
            mask_ei = torch.zeros(num_edges[i]).to(x_ni.device)
            mask_ni[perm_ni] = 1
            mask_ei[perm_ei] = 1
            mask_ni += torch.matmul(par_i.abs(), mask_ei)

            par_i = par_i[mask_ni>0]
            par_i = par_i.T[mask_ei>0].T
            # print(par_i.shape)
            L0_i = torch.matmul(par_i, par_i.T)
            lambda0, _ = torch.linalg.eigh(L0_i)
            L1_i = torch.matmul(par_i.T, par_i)
            # lambda1, _ = torch.linalg.eigh(L1_i)

            L0_i = 2 * L0_i / lambda0.max()
            L1_i = 2 * L1_i / lambda0.max()

            edge_index_t, edge_weight_t = dense_to_sparse(L0_i)
            edge_index_s, edge_weight_s = dense_to_sparse(L1_i)
            ngraph_list.append(Data(x=signal_ni[mask_ni>0], edge_index=edge_index_t, edge_weight=edge_weight_t, num_nodeas=L0_i.shape[0]))
            egraph_list.append(Data(x=signal_ei[mask_ei>0], edge_index=edge_index_s, edge_weight=edge_weight_s, num_nodeas=L1_i.shape[0]))
            par_list.append(Data(edge_index=par2adj(par_i), num_nodeas=L0_i.shape[0]))
                
        new_ndata = Batch.from_data_list(ngraph_list)
        new_edata = Batch.from_data_list(egraph_list)
        new_par = Batch.from_data_list(par_list)
        # e_batch = torch.cat([torch.ones(g.x.shape[0], device=g.x.device) for idx,g in enumerate(egraph_list)], dim=0).to(torch.long)
        # print(e_batch.shape, new_edata.x.shape)

        return new_ndata.x, new_ndata.edge_index, new_ndata.edge_weight, new_ndata.batch, new_edata.x, new_edata.edge_index, new_edata.edge_weight, new_edata.batch, new_par.edge_index 


    def __repr__(self) -> str:
        ratio = f'ratio={self.ratio}'

        return (f'{self.__class__.__name__}({self.gnn.__class__.__name__}, '
                f'{self.in_channels}, {ratio}, multiplier={None})')
##########################################################################################

class Graclus_Pooling(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool = nn.MaxPool1d(2, stride=2)
    
    def forward(self, data, s_batch, edge_index_s1, edge_weight_s1):
        # Input: edge feature: [N*E,C]
        x, num_graphs = data.x_s, data.num_graphs
        x = torch.transpose( x.view(num_graphs, -1, x.shape[-1]), 1,2) 
        # change dimension to [N,E,C], then transpose to [N,C,E]
        x = self.max_pool(x) # dim: [N,C,E/2]
        x = torch.transpose(x, 1,2) # dim: [N,E/2,C]
        data_list = []
        for i in range(num_graphs):
            data_list.append(Data(x=x[i], edge_index=edge_index_s1, edge_weight=edge_weight_s1))
            
        new_data = Batch.from_data_list(data_list)
        data.x_s, data.edge_index_s, data.edge_weight_s = new_data.x, new_data.edge_index, new_data.edge_weight
        return data, s_batch[::2]


###############################################################################
class Graclus_Pooling_Permutation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool = nn.MaxPool1d(2, stride=2)
        self.max_pool_batch = nn.MaxPool1d(2, stride=2)
    
    def forward(self, data, s_batch, edge_index_s1, edge_weight_s1, idx_dic1):
        # Input: edge feature: [N*E,C]
        x, num_graphs = data.x_s, data.num_graphs
        x = torch.transpose( x.view(num_graphs, -1, x.shape[-1]), 0,1) 
        x = torch.transpose( torch.transpose( x[idx_dic1], 0,1), 1,2) 

        s_batch = torch.transpose( s_batch.view(num_graphs, -1, 1), 0,1) 
        s_batch = torch.transpose( torch.transpose( s_batch[idx_dic1], 0,1), 1,2)  
        s_batch = self.max_pool_batch(s_batch.to(torch.double))
        # change dimension to [N,E,C], then transpose to [E,N,C]
        # permute with 'idx_dic1' the transpose to [N,C,E]
        
        x = self.max_pool(x) # dim: [N,C,E/2]
        x = torch.transpose(x, 1,2) # dim: [N,E/2,C]
        data_list = []
        for i in range(num_graphs):
            data_list.append(Data(x=x[i], edge_index=edge_index_s1, edge_weight=edge_weight_s1))
            
        new_data = Batch.from_data_list(data_list)
        data.x_s, data.edge_index_s, data.edge_weight_s = new_data.x, new_data.edge_index, new_data.edge_weight
        return data, s_batch.view(-1).to(torch.long)
    
    
###############################################################################
class Graclus_Node_Pooling(torch.nn.Module):
    def __init__(self, is_abs = True):
        super().__init__()
        self.max_pool = nn.MaxPool1d(2, stride=2)
    
    def forward(self, data, t_batch, edge_index_t1, edge_weight_t1):
        # Input: Node feature: [B*N,T,C]
        x, num_graphs = data.x_t, data.num_graphs
        xshape = x.shape
        
        # x = torch.transpose(x,1,2) # change dim to [B*N,C,T]
        x = torch.transpose( x.view(data.num_graphs, 268, -1) , 1,2) 
        # change dimension to [B,N,T*C], then transpose to [B,T*C,N]
        x = self.max_pool(x) # dim: [B,T*C,N/2]
        x = torch.transpose(x, 1,2) # dim: [B,N/2,T*C]
        x = x.view(x.shape[0],x.shape[1],xshape[1],xshape[2]) # dim: [B,N/2,C,T]
        data_list = []
        for i in range(num_graphs):
            data_list.append(Data(x=x[i], edge_index=edge_index_t1, edge_weight=edge_weight_t1))
            
        new_data = Batch.from_data_list(data_list)
        data.x_t, data.edge_index_t, data.edge_weight_t = new_data.x, new_data.edge_index, new_data.edge_weight
        return data, t_batch[::2]


    
    
###############################################################################
############################# Convolution #####################################
###############################################################################

class Inception1D(nn.Module):
    def __init__(self, in_channels, num_channels, maxpool=3, if_dim_reduction=False, 
                 leaky_slope=0.1):
        super(Inception1D, self).__init__()
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.if_dim_reduction = if_dim_reduction
        
        if if_dim_reduction:
            # if need dimension reduction
            self.channel1_11 = nn.Conv1d(in_channels, num_channels, 1, padding=0)
            self.channel2_11 = nn.Conv1d(in_channels, int(in_channels/2), 1, padding=0)
            self.channel3_11 = nn.Conv1d(in_channels, int(in_channels/2), 1, padding=0)
            self.channel4_11 = nn.Conv1d(in_channels, int(in_channels/2), 1, padding=0)
            in_channels = int(in_channels / 2)
            self.out_size = in_channels + self.num_channels*3
        else:
            self.out_size = self.in_channels*2 + self.num_channels*2
        
        self.channel2_13 = nn.Conv1d(in_channels, num_channels, 3, padding=1)
        self.channel3_15 = nn.Conv1d(in_channels, num_channels, 5, padding=2)
        self.channel4_mp = nn.MaxPool1d(3, stride=1, padding=1)
        self.cat_mp = nn.MaxPool1d(maxpool, stride=int(maxpool-1), padding=int((maxpool-1)/2))
        self.leakyReLU = nn.LeakyReLU(leaky_slope)
        self.bn = nn.BatchNorm1d(self.out_size)
    
    def forward(self, x):
        # Temporal Feature x: N*T*C
        # print(x.shape)
        x = torch.transpose(x,1,2) # change dim to N*C*T
        if self.if_dim_reduction:
            x1 = self.channel1_11(x)
            x2 = self.channel2_11(x)
            x3 = self.channel3_11(x)
            x4 = self.channel4_11(self.channel4_mp(x))
        else:
            x1, x2, x3 = x, x, x
            x4 = self.channel4_mp(x)
        
        x2 = self.channel2_13(self.leakyReLU(x2))
        x3 = self.channel3_15(self.leakyReLU(x3))
        x = self.cat_mp(self.leakyReLU(self.bn(torch.cat((x1,x2,x3,x4),dim=1))))
        
        return torch.transpose(x,1,2)
    
###############################################################################
class Inception1D_large_recp(nn.Module):
    def __init__(self, in_channels, num_channels, maxpool=3, if_dim_reduction=False, 
                 leaky_slope=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.if_dim_reduction = if_dim_reduction
        
        if if_dim_reduction:
            # if need dimension reduction
            self.channel1_11 = nn.Conv1d(in_channels, num_channels, 3, padding=1)
            self.channel2_11 = nn.Conv1d(in_channels, int(in_channels/2), 1, padding=0)
            self.channel3_11 = nn.Conv1d(in_channels, int(in_channels/2), 1, padding=0)
            self.channel4_11 = nn.Conv1d(in_channels, int(in_channels/2), 1, padding=0)
            in_channels = int(in_channels / 2)
            self.out_size = in_channels + self.num_channels*3
        else:
            self.out_size = self.in_channels*2 + self.num_channels*2
        
        self.channel2_13 = nn.Conv1d(in_channels, num_channels, 7, padding=3)
        self.channel3_15 = nn.Conv1d(in_channels, num_channels, 11, padding=5)
        self.channel4_mp = nn.MaxPool1d(5, stride=1, padding=2)
        self.cat_mp = nn.MaxPool1d(maxpool, stride=int(maxpool-1), padding=int((maxpool-1)/2))
        self.leakyReLU = nn.LeakyReLU(0.1)
        self.bn = nn.BatchNorm1d(self.out_size)
    
    def forward(self, x):
        # Temporal Feature x: N*T*C
        # print(x.shape)
        x = torch.transpose(x,1,2) # change dim to N*C*T
        if self.if_dim_reduction:
            x1 = self.channel1_11(x)
            x2 = self.channel2_11(x)
            x3 = self.channel3_11(x)
            x4 = self.channel4_11(self.channel4_mp(x))
        else:
            x1, x2, x3 = x, x, x
            x4 = self.channel4_mp(x)
        
        x2 = self.channel2_13(self.leakyReLU(x2))
        x3 = self.channel3_15(self.leakyReLU(x3))
        x = self.cat_mp(self.leakyReLU(self.bn(torch.cat((x1,x2,x3,x4),dim=1))))
        
        return torch.transpose(x,1,2)
        

###################################################################################
class HodgeLaguerreResConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, K: int, 
                  bias: bool = True, beta=0.5, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        assert K > 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lins = torch.nn.ModuleList([
            Linear(in_channels, out_channels, bias=False,
                    weight_initializer='glorot') for _ in range(K)
        ])
        self.beta = beta
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        zeros(self.bias)

    
    def forward(self, x: Tensor, edge_index: Tensor,
                edge_weight: OptTensor = None, batch: OptTensor = None):
        """"""
        # x: N*T*C
        norm = edge_weight
        Tx_0 = x
        Tx_1 = x  # Dummy.
        out = self.beta*self.lins[0](Tx_0) + (1-self.beta)*Tx_0
        xshape = x.shape
        k = 1

        # propagate_type: (x: Tensor, norm: Tensor)
        if len(self.lins) > 1:
            x = x.view(xshape[0],-1)
            Tx_1 = x - self.propagate(edge_index, x=x, norm=norm, size=None)
            if len(xshape)>=3:
                Tx_1 = Tx_1.view(xshape[0],xshape[1],-1)
            out = out + self.beta*self.lins[1](Tx_1) + (1-self.beta)*Tx_1

        for lin in self.lins[2:]:
            inshape = Tx_1.shape
            Tx_1 = Tx_1.view(inshape[0],-1)
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
            if len(xshape)>=3:
                Tx_2 = Tx_2.view(inshape[0],inshape[1],-1)
                Tx_1 = Tx_1.view(xshape[0],xshape[1],-1)
            # print(Tx_0.shape,Tx_1.shape,Tx_2.shape)
            Tx_2 = (-Tx_2 + (2*k+1)*Tx_1 - k* Tx_0) / (k+1)
            k += 1
            out = out + self.beta*lin.forward(Tx_2) + (1-self.beta)*Tx_2
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={len(self.lins)}')
    
###############################################################################
class HodgeChebConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, K: int, 
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        assert K > 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lins = torch.nn.ModuleList([
            Linear(in_channels, out_channels, bias=False,
                   weight_initializer='glorot') for _ in range(K)
        ])

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        zeros(self.bias)

    
    def forward(self, x: Tensor, edge_index: Tensor,
                edge_weight: OptTensor = None, batch: OptTensor = None):
        """"""
        # x: N*T*C
        norm = edge_weight
        Tx_0 = x
        Tx_1 = x  # Dummy.
        out = self.lins[0](Tx_0)
        xshape = x.shape

        # propagate_type: (x: Tensor, norm: Tensor)
        if len(self.lins) > 1:
            # print(x.shape,xshape[0])
            # x = x.view(xshape[0],-1)
            
            if len(xshape)==3:
                x = torch.transpose(x,1,2) # change dim to [N,C,T]
                x = x.view(xshape[0],-1)
                Tx_1 = self.propagate(edge_index, x=x, norm=norm, size=None)
                Tx_1 = Tx_1.view(xshape[0],xshape[2],-1) #[N,C,T]
                Tx_1 = torch.transpose(Tx_1,1,2)
            else:
                Tx_1 = self.propagate(edge_index, x=x, norm=norm, size=None)
            out = out + self.lins[1](Tx_1)

        for lin in self.lins[2:]:
            
            if len(xshape)>=3:
                Tx_1 = torch.transpose(Tx_1,1,2) # change dim to [N,C,T]
                Tx_1 = Tx_1.view(xshape[0],-1)
                Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
                Tx_2 = Tx_2.view(xshape[0],xshape[2],-1) #[N,C,T]
                Tx_2 = torch.transpose(Tx_2,1,2) #[N,T,C]
                Tx_1 = Tx_1.view(xshape[0],xshape[2],-1) #[N,C,T]
                Tx_1 = torch.transpose(Tx_1,1,2)
            else:
                Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
            # print(Tx_0.shape,Tx_1.shape,Tx_2.shape)
            Tx_2 = 2. * Tx_2 - Tx_0
            out = out + lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={len(self.lins)}, '
                f'normalization={self.normalization})')
    
###############################################################################

class HodgeLaguerreConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, K: int, 
                  bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        assert K > 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lins = torch.nn.ModuleList([
            Linear(in_channels, out_channels, bias=False,
                    weight_initializer='glorot') for _ in range(K)
        ])

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        zeros(self.bias)

    
    def forward(self, x: Tensor, edge_index: Tensor,
                edge_weight: OptTensor = None, batch: OptTensor = None):
        """"""
        # x: N*T*C
        norm = edge_weight
        Tx_0 = x
        Tx_1 = x  # Dummy.
        out = self.lins[0](Tx_0)
        xshape = x.shape
        k = 1

        # propagate_type: (x: Tensor, norm: Tensor)
        if len(self.lins) > 1:
            x = x.view(xshape[0],-1)
            Tx_1 = x - self.propagate(edge_index, x=x, norm=norm, size=None)
            if len(xshape)>=3:
                Tx_1 = Tx_1.view(xshape[0],xshape[1],-1)
            out = out + self.lins[1](Tx_1)

        for lin in self.lins[2:]:
            inshape = Tx_1.shape
            Tx_1 = Tx_1.view(inshape[0],-1)
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
            if len(xshape)>=3:
                Tx_2 = Tx_2.view(inshape[0],inshape[1],-1)
                Tx_1 = Tx_1.view(xshape[0],xshape[1],-1)
            # print(Tx_0.shape,Tx_1.shape,Tx_2.shape)
            Tx_2 = (-Tx_2 + (2*k+1)*Tx_1 - k* Tx_0) / (k+1)
            k += 1
            out = out + lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={len(self.lins)}')

    
# class HodgeLaguerreConv(MessagePassing):
#     def __init__(self, in_channels: int, out_channels: int, K: int, 
#                  bias: bool = True, **kwargs):
#         kwargs.setdefault('aggr', 'add')
#         super().__init__(**kwargs)

#         assert K > 0

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.lins = torch.nn.ModuleList([
#             Linear(in_channels, out_channels, bias=False,
#                    weight_initializer='glorot') for _ in range(K)
#         ])

#         if bias:
#             self.bias = Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)

#         self.reset_parameters()

#     def reset_parameters(self):
#         for lin in self.lins:
#             lin.reset_parameters()
#         zeros(self.bias)

    
#     def forward(self, x: Tensor, edge_index: Tensor,
#                 edge_weight: OptTensor = None, batch: OptTensor = None):
#         """"""
#         # x: N*T*C
#         norm = edge_weight
#         Tx_0 = x
#         Tx_1 = x  # Dummy.
#         out = self.lins[0](Tx_0)
#         xshape = x.shape
#         k = 1
#         # propagate_type: (x: Tensor, norm: Tensor)
#         if len(self.lins) > 1:
#             # print(x.shape,xshape[0])
#             # x = x.view(xshape[0],-1)
#             if len(xshape)==3:
#                 x = torch.transpose(x,1,2) # change dim to [N,C,T]
#                 x = x.view(xshape[0],-1)
#                 Tx_1 = x - self.propagate(edge_index, x=x, norm=norm, size=None)
#                 Tx_1 = Tx_1.view(xshape[0],xshape[2],-1) #[N,C,T]
#                 Tx_1 = torch.transpose(Tx_1,1,2)
#             else:
#                 Tx_1 = x - self.propagate(edge_index, x=x, norm=norm, size=None)
#             out = out + self.lins[1](Tx_1)

#         for lin in self.lins[2:]:
            
#             if len(xshape)>=3:
#                 Tx_1 = torch.transpose(Tx_1,1,2) # change dim to [N,C,T]
#                 Tx_1 = Tx_1.view(xshape[0],-1)
#                 Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
#                 Tx_2 = Tx_2.view(xshape[0],xshape[2],-1) #[N,C,T]
#                 Tx_2 = torch.transpose(Tx_2,1,2) #[N,T,C]
#                 Tx_1 = Tx_1.view(xshape[0],xshape[2],-1) #[N,C,T]
#                 Tx_1 = torch.transpose(Tx_1,1,2)
#             else:
#                 Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
#             # print(Tx_0.shape,Tx_1.shape,Tx_2.shape)
#             Tx_2 = (-Tx_2 + (2*k+1)*Tx_1 - k* Tx_1) / (k+1)
#             k += 1
#             out = out + lin.forward(Tx_2)
#             Tx_0, Tx_1 = Tx_1, Tx_2

#         if self.bias is not None:
#             out = out + self.bias

#         return out    
###############################################################################
################################ BACKUP #######################################
###############################################################################
# class Hodge_Edge_grad_Pooling(torch.nn.Module):
#     def __init__(self, is_abs = True):
#         super().__init__()
#         self.is_abs = is_abs
    
#     def forward(self, data, inc_mat, s_batch, device='cuda:0'):
#         # Input: edge graph (data, inc_mat, s_batch)
#         if self.is_abs:
#             n_sim = torch.abs(F.cosine_similarity(data.x_s[data.edge_index_s[0,:]], data.x_s[data.edge_index_s[1,:]]))
#         else:
#             n_sim = F.cosine_similarity(data.x_s[data.edge_index_s[0,:]], data.x_s[data.edge_index_s[1,:]]) + 1
#         # print(n_sim.shape)
#         gc = graclus(data.edge_index_s, n_sim, data.x_s.shape[0])
#         # print(gc.shape)
    
#         # based on the cluster update the signal and incidence matrix
#         idx = torch.arange(0,gc.shape[0]).to(device)
#         col_idx = scatter(idx, gc, dim=-1, reduce='max')[torch.unique(gc)] # index of columns need to retain
#         inc_mat = inc_mat[:,col_idx].to(torch.double)
#         s_batch1 = s_batch[col_idx]
#         data.x_s = scatter(data.x_s, gc, dim=0, reduce='max')
#         data.x_s = data.x_s[torch.unique(gc)]
    
#         # update the hodge laplacian
#         data_list = []
#         for i in range(data.num_graphs):
#             inc_mat_1 = inc_mat[:,s_batch1==i].to_sparse()
#             x_s_1 = data.x_s[s_batch1==i]
#             # print(x_s_1.shape, inc_mat_1.shape)
#             temp = torch.sparse.mm(inc_mat_1, inc_mat_1.t()).to_dense()
#             lambda_max = torch.max(torch.real(torch.linalg.eigvals(temp)))
    
#             # normalize the laplacian
#             edge_index, edge_weight = dense_to_sparse( torch.sparse.mm(inc_mat_1.t(), inc_mat_1).to_dense() )
#             edge_weight = (2.0 * edge_weight) / lambda_max
#             edge_weight.masked_fill_(edge_weight == float('inf'), 0)
#             edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=-1., num_nodes=x_s_1.shape[0])
#             data_list.append(Data(x=x_s_1, edge_index=edge_index, edge_weight=edge_weight))
    
#         new_data = Batch.from_data_list(data_list)
#         data.x_s, data.edge_index_s, data.edge_weight_s = new_data.x, new_data.edge_index, new_data.edge_weight
#         return data, inc_mat, s_batch1
    
    
# class Hodge_Edge_Pooling(torch.nn.Module):
#     def __init__(self, is_abs = True):
#         super().__init__()
#         self.is_abs = is_abs
    
#     def forward(self, data, inc_mat, s_batch, device='cuda:0'):
#         # Input: edge graph (data, inc_mat, s_batch)
#         # with torch.no_grad():
#         if self.is_abs:
#             n_sim = torch.abs(F.cosine_similarity(data.x_s[data.edge_index_s[0,:]], data.x_s[data.edge_index_s[1,:]]))
#         else:
#             n_sim = F.cosine_similarity(data.x_s[data.edge_index_s[0,:]], data.x_s[data.edge_index_s[1,:]]) + 1
#         # print(n_sim.shape)
#         gc = graclus(data.edge_index_s, n_sim, data.x_s.shape[0])
#         # print(gc.shape)
    
#         # based on the cluster update the signal and incidence matrix
#         idx = torch.arange(0,gc.shape[0]).to(device)
#         col_idx = scatter(idx, gc, dim=-1, reduce='max')[torch.unique(gc)] # index of columns need to retain
#         inc_mat = inc_mat[:,col_idx].to(torch.double)
#         s_batch1 = s_batch[col_idx]
#         data.x_s = scatter(data.x_s, gc, dim=0, reduce='max')
#         data.x_s = data.x_s[torch.unique(gc)]
    
#         # update the hodge laplacian
#         data_list = []
#         for i in range(data.num_graphs):
#             inc_mat_1 = inc_mat[:,s_batch1==i].to_sparse()
#             x_s_1 = data.x_s[s_batch1==i]
#             # print(x_s_1.shape, inc_mat_1.shape)
#             temp = torch.sparse.mm(inc_mat_1, inc_mat_1.t()).to_dense()
#             lambda_max = torch.max(torch.real(torch.linalg.eigvals(temp)))
    
#             # normalize the laplacian
#             edge_index, edge_weight = dense_to_sparse( torch.sparse.mm(inc_mat_1.t(), inc_mat_1).to_dense() )
#             edge_weight = (2.0 * edge_weight) / lambda_max
#             edge_weight.masked_fill_(edge_weight == float('inf'), 0)
#             edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=-1., num_nodes=x_s_1.shape[0])
#             data_list.append(Data(x=x_s_1, edge_index=edge_index, edge_weight=edge_weight))
    
#         new_data = Batch.from_data_list(data_list)
#         data.x_s, data.edge_index_s, data.edge_weight_s = new_data.x, new_data.edge_index, new_data.edge_weight
#         return data, inc_mat, s_batch1

# ###############################################################################
# class HodgePooling(torch.nn.Module):
#     def __init__(self, is_abs = True):
#         super().__init__()
#         self.is_abs = is_abs
    
#     def forward(self, x, edge_index, edge_attr, batch):
#         xshape = x.shape
#         if len(xshape)>=3:
#             x = x.view(xshape[0],-1)

#         n_sim = F.cosine_similarity(x[edge_index[0,:]], x[edge_index[1,:]])
#         if self.is_abs:
#             n_sim = torch.abs(n_sim)
#         else:
#             n_sim = n_sim + 1
#         # print(n_sim.shape,edge_index.shape)
#         gc = graclus(edge_index.detach(), n_sim.detach(), x.shape[0])
#         # print(gc.shape)
#         data1 = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
#         data1 = max_pool(gc, data1)
#         if len(xshape)>=3:
#             data1.x = data1.x.view(-1, xshape[1], xshape[2])
#         return data1.x, data1.edge_index, data1.edge_attr, data1.batch
