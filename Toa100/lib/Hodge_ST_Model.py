#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 22:04:29 2022

@author: jinghan
"""

import numpy as np
from torch.nn import Linear, Dropout
import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GATConv, TAGConv, global_mean_pool, global_max_pool, EdgeConv, DynamicEdgeConv
from torch_geometric.utils import add_self_loops, dense_to_sparse, coalesce
import torch_geometric.nn as gnn
import torch.nn as nn
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import time
# from Hodge_Cheb_Conv import *
# from Hodge_Dataset import *


###############################################################################
##########################  ST Convolution & Pooling  #########################
###############################################################################
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
    

class Hodge_STConv_Pool(torch.nn.Module):
    def __init__(self, Spatial_channels=[8, 16], Temporal_channels=[4, 8], 
                 Edge_channels=[32, 32, 1], mlp_channels=[256,128], K=4, roi_num=268, 
                 edge_num=8978, time_point=375, num_classes=1, leaky_slope=0.33,
                 dropout_ratio=0.5):
        super(Hodge_STConv_Pool, self).__init__()
        self.Spatial_channels = Spatial_channels
        self.Temporal_channels = Temporal_channels
        self.Edge_channels = Edge_channels
        self.roi_num = roi_num
        self.edge_num = edge_num
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_slope)
        self.relu = nn.ReLU()
        self.time_point=time_point
        
        gcn_insize = 1
        for i, TC in enumerate(Temporal_channels):
            if_dim_reduction = i!=0;
            fc = Inception1D(gcn_insize,TC,maxpool=5,if_dim_reduction=if_dim_reduction)
            setattr(self, 'ntc%d' % i, fc) # node temporal convolution
            gcn_insize = fc.out_size
            time_point = int(np.ceil(time_point/4))
            
            fc = HodgeChebConv(gcn_insize, Spatial_channels[i], K=K)
            setattr(self, 'nsc%d' % i, fc) # node spatial convolution
            
            fc = gnn.BatchNorm(Spatial_channels[i]*time_point)
            setattr(self, 'nbn%d' % i, fc) # node batch normalization
            
            fc = Dropout(p=dropout_ratio)
            setattr(self, 'ndo%d' % i, fc) # node dropout layer
            gcn_insize = Spatial_channels[i]

        gcn_insize = 1
        for i, gcn_outsize in enumerate(Edge_channels):
            fc = gnn.Sequential('x, edge_index, edge_weight', [
                (HodgeLaguerreConv(gcn_insize, gcn_outsize, K=K),
                 'x, edge_index, edge_weight -> x'),
                gnn.BatchNorm(gcn_outsize),
                nn.LeakyReLU(negative_slope=leaky_slope),
                (Dropout(p=dropout_ratio), 'x -> x'),
            ])

            setattr(self, 'esc%d' % i, fc)
            gcn_insize = gcn_outsize
        
        lin_insize = int(Spatial_channels[-1]*self.roi_num/2) + int(edge_num/2)
        self.lin1 = Linear(lin_insize, mlp_channels[0])
        self.lin2 = Linear(mlp_channels[0], mlp_channels[1])
        self.lin3 = Linear(mlp_channels[1], num_classes)
        self.bn1 = nn.BatchNorm1d(mlp_channels[0])
        self.bn2 = nn.BatchNorm1d(mlp_channels[1])
        self.dropout = nn.Dropout(dropout_ratio)
        self.nspool = Graclus_Node_Pooling() # node spatial pooling
        self.spool = Graclus_Pooling() # edge spatial pooling
        

    def forward(self, data, edge_index_t1, edge_weight_t1, edge_index_s1, edge_weight_s1, if_att=False):

        # 1. Obtain node embeddings
        
        ## node convolution
        batch = int(data.x_t.shape[0] / self.roi_num)
        t_batch = torch.tensor([[i]*self.roi_num for i in range(batch)])
        t_batch = t_batch.view(-1).to(device)
        data.x_t = data.x_t.view(-1,self.time_point,1)
        
        for i, _ in enumerate(self.Temporal_channels):
            fc = getattr(self, 'ntc%d' % i)
            data.x_t = fc(data.x_t)
            fc = getattr(self, 'nsc%d' % i)
            data.x_t = fc(data.x_t, data.edge_index_t, data.edge_weight_t)
            if i == 0:
                data, t_batch = self.nspool(data, t_batch, edge_index_t1, edge_weight_t1)

            # change dim to N*(T*C) and apply batch normalization
            fc = getattr(self, 'nbn%d' % i)
            t_shape = data.x_t.shape
            data.x_t = data.x_t.view(t_shape[0],-1)
            data.x_t = self.leaky_relu(fc(data.x_t))
            data.x_t = data.x_t.view(t_shape[0],t_shape[1],t_shape[2])
            if i == 0:
                fc = getattr(self, 'ndo%d' % i)
                data.x_t = fc(data.x_t)
        
        # edge convolution
        s_batch = torch.tensor([[i]*self.edge_num for i in range(batch)])
        s_batch = s_batch.view(-1).to(device)

        # x_s_global_ave = []
        for i, _ in enumerate(self.Edge_channels):
            fc = getattr(self, 'esc%d' % i)
            data.x_s = fc(data.x_s, data.edge_index_s, data.edge_weight_s)
            # x_s_global_ave.append(global_mean_pool(data.x_s, s_batch))
            if i == 0:#< len(self.Edge_channels)-1:
                # print(x_s.shape)
                data, s_batch = self.spool(data, s_batch, edge_index_s1, edge_weight_s1)
            # if i < 2:
            #     data.x_s = self.dropout(data.x_s)

        # 2. Readout layer
        data.x_t = data.x_t.mean(dim=1)
        x = torch.cat((data.x_t.view(data.num_graphs,-1), data.x_s.view(batch,-1)), -1)
        if if_att:
            att = data.x_s.view(batch,-1)
        # print(x.shape)
        
        # 3. Apply a final classifier
        x = self.lin1(x)
        x = self.bn1(x)
        x = x.relu()
        x = self.dropout(x)
        
        x = self.lin2(x)
        x = self.bn2(x)
        x = x.relu()
        x = self.dropout(x)
   
        if if_att:
            return self.lin3(x), att
        else:
            return self.lin3(x)

#########################################################################
class Hodge_STConv_2Dread(torch.nn.Module):
    def __init__(self, Spatial_channels=[8, 8], Temporal_channels=[4, 8], 
                 Edge_channels=[32, 32, 1], mlp_channels=[256,128], K=4, roi_num=268, 
                 edge_num=8978, time_point=375, num_classes=1, leaky_slope=0.33,
                 dropout_ratio=0.5, tpool_step=5):
        super(Hodge_STConv_2Dread, self).__init__()
        self.Spatial_channels = Spatial_channels
        self.Temporal_channels = Temporal_channels
        self.Edge_channels = Edge_channels
        self.roi_num = roi_num
        self.edge_num = edge_num
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_slope)
        self.relu = nn.ReLU()
        self.time_point=time_point
        
        gcn_insize = 1
        for i, TC in enumerate(Temporal_channels):
            if_dim_reduction = i!=0;
            fc = Inception1D(gcn_insize,TC,maxpool=tpool_step,if_dim_reduction=if_dim_reduction)
            setattr(self, 'ntc%d' % i, fc) # node temporal convolution
            gcn_insize = fc.out_size
            time_point = int(np.ceil(time_point/(tpool_step-1) ))
            
            fc = HodgeChebConv(gcn_insize, Spatial_channels[i], K=K-1)
            setattr(self, 'nsc%d' % i, fc) # node spatial convolution
            
            fc = gnn.BatchNorm(Spatial_channels[i]*time_point)
            setattr(self, 'nbn%d' % i, fc) # node batch normalization
            
            # fc = Dropout(p=dropout_ratio)
            # setattr(self, 'ndo%d' % i, fc) # node dropout layer
            gcn_insize = Spatial_channels[i]

        gcn_insize = 1
        for i, gcn_outsize in enumerate(Edge_channels):
            fc = gnn.Sequential('x, edge_index, edge_weight', [
                (HodgeLaguerreConv(gcn_insize, gcn_outsize, K=K),
                 'x, edge_index, edge_weight -> x'),
                gnn.BatchNorm(gcn_outsize),
                nn.LeakyReLU(negative_slope=leaky_slope),
                # (Dropout(p=dropout_ratio), 'x -> x'),
            ])

            setattr(self, 'esc%d' % i, fc)
            gcn_insize = gcn_outsize
        
        lin_insize = int(Spatial_channels[-1]*self.roi_num/2) + int(edge_num/2) + sum(Edge_channels[:-1])
        self.lin1 = Linear(lin_insize, mlp_channels[0])
        self.lin2 = Linear(mlp_channels[0], mlp_channels[1])
        self.lin3 = Linear(mlp_channels[1], num_classes)
        self.bn1 = nn.BatchNorm1d(mlp_channels[0])
        self.bn2 = nn.BatchNorm1d(mlp_channels[1])
        self.dropout = nn.Dropout(dropout_ratio)
        self.nspool = Graclus_Node_Pooling() # node spatial pooling
        self.spool = Graclus_Pooling() # edge spatial pooling
        

    def forward(self, data, edge_index_t1, edge_weight_t1, edge_index_s1, edge_weight_s1, if_att=False):

        # 1. Obtain node embeddings
        
        ## node convolution
        batch = int(data.x_t.shape[0] / self.roi_num)
        t_batch = torch.tensor([[i]*self.roi_num for i in range(batch)])
        t_batch = t_batch.view(-1).to(device)
        data.x_t = data.x_t.view(-1,self.time_point,1)
        
        for i, _ in enumerate(self.Temporal_channels):
            fc = getattr(self, 'ntc%d' % i)
            data.x_t = fc(data.x_t)
            fc = getattr(self, 'nsc%d' % i)
            data.x_t = fc(data.x_t, data.edge_index_t, data.edge_weight_t)
            if i == 0:
                data, t_batch = self.nspool(data, t_batch, edge_index_t1, edge_weight_t1)

            # change dim to N*(T*C) and apply batch normalization
            fc = getattr(self, 'nbn%d' % i)
            t_shape = data.x_t.shape
            data.x_t = data.x_t.view(t_shape[0],-1)
            data.x_t = self.leaky_relu(fc(data.x_t))
            data.x_t = data.x_t.view(t_shape[0],t_shape[1],t_shape[2])
            # if i == 0:
            #     fc = getattr(self, 'ndo%d' % i)
            #     data.x_t = fc(data.x_t)
        
        # edge convolution
        s_batch = torch.tensor([[i]*self.edge_num for i in range(batch)])
        s_batch = s_batch.view(-1).to(device)

        x_s_global_ave = []
        for i, _ in enumerate(self.Edge_channels):
            fc = getattr(self, 'esc%d' % i)
            data.x_s = fc(data.x_s, data.edge_index_s, data.edge_weight_s)
            
            if i == 0:#< len(self.Edge_channels)-1:
                # print(x_s.shape)
                data, s_batch = self.spool(data, s_batch, edge_index_s1, edge_weight_s1)
                x_s_global_ave.append(global_mean_pool(data.x_s, s_batch))
            elif i == 1:
                x_s_global_ave.append(global_mean_pool(data.x_s, s_batch))

        # 2. Readout layer
        data.x_t = data.x_t.mean(dim=1)
        x_s_global_ave = torch.cat(x_s_global_ave,-1)
        x = torch.cat([data.x_t.view(data.num_graphs,-1),data.x_s.view(batch,-1),x_s_global_ave], -1)

        # print(x.shape)
        
        # 3. Apply a final classifier
        x = self.lin1(x)
        x = self.bn1(x)
        x = x.relu()
        x = self.dropout(x)
        
        x = self.lin2(x)
        x = self.bn2(x)
        x = x.relu()
        x = self.dropout(x)
   
        if if_att:
            return self.lin3(x), data.x_s.view(batch,-1)
        else:
            return self.lin3(x)

    
###############################################################################
##################### Only Edge Convolution & Pooling #########################
###############################################################################
   
class Hodge_SpatialConv_2Dread(torch.nn.Module):
    def __init__(self, Edge_channels=[32, 32, 1], mlp_channels=[256,128], K=4, roi_num=268, 
                 edge_num=8978, time_point=375, num_classes=1, leaky_slope=0.33,
                 dropout_ratio=0.5):
        super(Hodge_SpatialConv_2Dread, self).__init__()
        self.Edge_channels = Edge_channels
        self.roi_num = roi_num
        self.edge_num = edge_num
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_slope)
        self.relu = nn.ReLU()

        gcn_insize = 1
        for i, gcn_outsize in enumerate(Edge_channels):
            fc = gnn.Sequential('x, edge_index, edge_weight', [
                (HodgeLaguerreConv(gcn_insize, gcn_outsize, K=K),
                 'x, edge_index, edge_weight -> x'),
                gnn.BatchNorm(gcn_outsize),
                nn.LeakyReLU(negative_slope=leaky_slope),
                (Dropout(p=dropout_ratio), 'x -> x'),
            ])

            setattr(self, 'esc%d' % i, fc)
            gcn_insize = gcn_outsize
        
        lin_insize = int(edge_num/2)+sum(Edge_channels[:2])
        self.lin1 = Linear(lin_insize, mlp_channels[0])
        self.lin2 = Linear(mlp_channels[0], mlp_channels[1])
        self.lin3 = Linear(mlp_channels[1], num_classes)
        self.bn1 = nn.BatchNorm1d(mlp_channels[0])
        self.bn2 = nn.BatchNorm1d(mlp_channels[1])
        self.dropout1 = nn.Dropout(dropout_ratio)
        self.dropout2 = nn.Dropout(dropout_ratio)
        self.spool0 = Graclus_Pooling()

    def forward(self, data, edge_index_s1, edge_weight_s1, if_att=False):

        # 1. Obtain node embeddings
        # data_t = Data(x=data.x_t, edge_index=data.edge_index_t, edge_weight=data.edge_weight_t)
        # data = Data(x=data.x_s, edge_index=data.edge_index_s, edge_weight=data.edge_weight_s)
        
        batch = int(data.x_t.shape[0] / self.roi_num)
        s_batch = torch.tensor([[i]*self.edge_num for i in range(batch)])
        s_batch = s_batch.view(-1).to(device)

        x_s_global_ave = []
        for i, _ in enumerate(self.Edge_channels):
            fc = getattr(self, 'esc%d' % i)
            data.x_s = fc(data.x_s, data.edge_index_s, data.edge_weight_s)
            
            if i == 0:#< len(self.Edge_channels)-1:
                # print(x_s.shape)
                data, s_batch = self.spool0(data, s_batch, edge_index_s1, edge_weight_s1)
                x_s_global_ave.append(global_mean_pool(data.x_s, s_batch))
            elif i == 1:
                x_s_global_ave.append(global_mean_pool(data.x_s, s_batch))

        # 2. Readout layer
        x_s_global_ave = torch.cat(x_s_global_ave,-1)
        x = torch.cat([data.x_s.view(batch,-1),x_s_global_ave], -1)
        # print(x.shape)
        
        # 3. Apply a final classifier
        x = self.lin1(x)
        x = self.bn1(x)
        x = x.relu()
        x = self.dropout1(x)
        
        x = self.lin2(x)
        x = self.bn2(x)
        x = x.relu()
        x = self.dropout2(x)
        
        if if_att:
            return self.lin3(x), data.x_s.view(batch,-1)
        else:
            return self.lin3(x)    

############################################################################### 
class Hodge_SpatialConv_Pool(torch.nn.Module):
    def __init__(self, Edge_channels=[32, 32, 1], mlp_channels=[256,128], K=4, roi_num=268, 
                 edge_num=8978, time_point=375, num_classes=1, leaky_slope=0.33,
                 dropout_ratio=0.5):
        super(Hodge_SpatialConv_Pool, self).__init__()
        self.Edge_channels = Edge_channels
        self.roi_num = roi_num
        self.edge_num = edge_num
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_slope, inplace=False)
        self.relu = nn.ReLU(inplace=False)
        # inplace=True
        gcn_insize = 1
        for i, gcn_outsize in enumerate(Edge_channels):
            fc = gnn.Sequential('x, edge_index, edge_weight', [
                (HodgeLaguerreConv(gcn_insize, gcn_outsize, K=K),
                 'x, edge_index, edge_weight -> x'),
                gnn.BatchNorm(gcn_outsize),
                nn.LeakyReLU(negative_slope=leaky_slope, inplace=False),
            ])

            setattr(self, 'esc%d' % i, fc)
            gcn_insize = gcn_outsize
        
        lin_insize = int(edge_num/2) #sum(Edge_channels)
        self.lin1 = Linear(lin_insize, mlp_channels[0])
        self.lin2 = Linear(mlp_channels[0], mlp_channels[1])
        self.lin3 = Linear(mlp_channels[1], num_classes)
        self.bn1 = nn.BatchNorm1d(mlp_channels[0])
        self.bn2 = nn.BatchNorm1d(mlp_channels[1])
        self.dropout = nn.Dropout(dropout_ratio)
        self.spool = Graclus_Pooling()

    def forward(self, data, edge_index_s1, edge_weight_s1, if_att=False):

        # 1. Obtain node embeddings
        # data_t = Data(x=data.x_t, edge_index=data.edge_index_t, edge_weight=data.edge_weight_t)
        # data = Data(x=data.x_s, edge_index=data.edge_index_s, edge_weight=data.edge_weight_s)
        
        batch = int(data.x_t.shape[0] / self.roi_num)
        s_batch = torch.tensor([[i]*self.edge_num for i in range(batch)])
        s_batch = s_batch.view(-1).to(device)

        # x_s_global_ave = []
        for i, _ in enumerate(self.Edge_channels):
            fc = getattr(self, 'esc%d' % i)
            data.x_s = fc(data.x_s, data.edge_index_s, data.edge_weight_s)
            # x_s_global_ave.append(global_mean_pool(data.x_s, s_batch))
            if i == 0:#< len(self.Edge_channels)-1:
                # print(x_s.shape)
                data, s_batch = self.spool(data, s_batch, edge_index_s1, edge_weight_s1)
            if i < 2:
                data.x_s = self.dropout(data.x_s)

        # 2. Readout layer
        # x = torch.cat(x_s_global_ave,-1)
        x = data.x_s.view(batch,-1)
        if if_att:
            att = data.x_s.view(batch,-1)
        # print(x.shape)
        
        # 3. Apply a final classifier
        x = self.lin1(x)
        x = self.bn1(x)
        x = x.relu()
        x = self.dropout(x)
        
        x = self.lin2(x)
        x = self.bn2(x)
        x = x.relu()
        x = self.dropout(x)
   
        if if_att:
            return self.lin3(x), att
        else:
            return self.lin3(x)
    
    
    

###############################################################################
################################ BaseLine #####################################
###############################################################################

class DGCN(torch.nn.Module):
    def __init__(self, GCN_channels=32, roi_num=268, dropout_ratio=0.5,
                 leaky_slope=0.33, num_classes=1):
        super(DGCN, self).__init__()
        self.GCN_channels = GCN_channels
        self.roi_num = roi_num
        
        gcn_insize = roi_num*2
        gcn_outsize = GCN_channels

        mlp1 = nn.Sequential(
            Linear(gcn_insize, gcn_outsize),
            nn.LeakyReLU(negative_slope=leaky_slope),
            Linear(gcn_outsize, gcn_outsize),)
            
        self.gcn1 = gnn.Sequential('x, edge_index', [
            (EdgeConv(mlp1),'x, edge_index -> x'),
            gnn.BatchNorm(gcn_outsize),
            nn.LeakyReLU(negative_slope=leaky_slope),
            (nn.Dropout(p=dropout_ratio), 'x -> x'),
        ])

        mlp2 = nn.Sequential(
            Linear(gcn_insize, gcn_outsize),
            nn.LeakyReLU(negative_slope=leaky_slope),
            Linear(gcn_outsize, gcn_outsize),)
            
        self.gcn2 = gnn.Sequential('x, batch', [
            (DynamicEdgeConv(mlp2, k=32),'x, batch -> x'),
            gnn.BatchNorm(gcn_outsize),
            nn.LeakyReLU(negative_slope=leaky_slope),
            (nn.Dropout(p=dropout_ratio), 'x -> x'),
        ])
        
        mlp3 = Linear(gcn_insize, 1)
            
        self.gcn3 = gnn.Sequential('x, edge_index', [
            (EdgeConv(mlp3),'x, edge_index -> x'),
            gnn.BatchNorm(1),
            nn.LeakyReLU(negative_slope=leaky_slope),
        ])
        
        self.lin1 = Linear(roi_num+GCN_channels*2, 256)
        self.lin2 = Linear(256, 128)
        self.lin3 = Linear(128, 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_slope)
        self.dropout1 = nn.Dropout(dropout_ratio)
        self.dropout2 = nn.Dropout(dropout_ratio)

    def forward(self, x, edge_index, batch):
        # print(x.shape)
        # 1. Obtain node embeddings 
        x1 = self.gcn1(x, edge_index)
        x2 = self.gcn2(x, batch)
        x3 = self.gcn3(x, edge_index)

        # 2. Readout layer
        x = torch.cat( (global_mean_pool(x1, batch), global_mean_pool(x2, batch), x3.view(-1,self.roi_num)), -1)  # [batch_size, hidden_channels]
        # 3. Apply a final classifier
        x = self.lin1(x)
        x = self.leaky_relu(self.bn1(x))
        x = self.dropout1(x)
        x = self.lin2(x)
        x = self.leaky_relu(self.bn2(x))
        x = self.dropout2(x)
        
        return self.lin3(x)
    
###############################################################################
class EHGNN(torch.nn.Module):
    def __init__(self, Edge_channels=[32, 32, 1], mlp_channels=[256,128], roi_num=268, 
                 edge_num=8978, time_point=375, num_classes=1, leaky_slope=0.33,
                 dropout_ratio=0.5):
        super(EHGNN, self).__init__()
        self.Edge_channels = Edge_channels
        self.roi_num = roi_num
        self.edge_num = edge_num
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_slope)
        self.relu = nn.ReLU()

        gcn_insize = 1
        for i, gcn_outsize in enumerate(Edge_channels):

            fc = gnn.Sequential('x, edge_index', [
                (TAGConv(gcn_insize, gcn_outsize),
                 'x, edge_index -> x'),
                gnn.BatchNorm(gcn_outsize),
                nn.LeakyReLU(negative_slope=leaky_slope),
                (Dropout(p=dropout_ratio), 'x -> x'),])

            setattr(self, 'esc%d' % i, fc)
            gcn_insize = gcn_outsize
        
        lin_insize = int(edge_num/2)#+sum(Edge_channels[:-1])
        self.lin1 = Linear(lin_insize, mlp_channels[0])
        self.lin2 = Linear(mlp_channels[0], mlp_channels[1])
        self.lin3 = Linear(mlp_channels[1], num_classes)
        self.bn1 = nn.BatchNorm1d(mlp_channels[0])
        self.bn2 = nn.BatchNorm1d(mlp_channels[1])
        self.dropout1 = nn.Dropout(dropout_ratio)
        self.dropout2 = nn.Dropout(dropout_ratio)
        self.spool0 = Graclus_Pooling()

    def forward(self, data, edge_index_s1, edge_weight_s1, if_att=False):

        # 1. Obtain node embeddings
        # data_t = Data(x=data.x_t, edge_index=data.edge_index_t, edge_weight=data.edge_weight_t)
        # data = Data(x=data.x_s, edge_index=data.edge_index_s, edge_weight=data.edge_weight_s)
        
        batch = int(data.x_t.shape[0] / self.roi_num)
        s_batch = torch.tensor([[i]*self.edge_num for i in range(batch)])
        s_batch = s_batch.view(-1).to(device)

        x_s_global_ave = []
        for i, _ in enumerate(self.Edge_channels):
            fc = getattr(self, 'esc%d' % i)
            data.x_s = fc(data.x_s, data.edge_index_s)
            
            if i == 0:#< len(self.Edge_channels)-1:
                # print(x_s.shape)
                data, s_batch = self.spool0(data, s_batch, edge_index_s1, edge_weight_s1)
                # x_s_global_ave.append(global_mean_pool(data.x_s, s_batch))
            # elif i == 1:
                # x_s_global_ave.append(global_mean_pool(data.x_s, s_batch))

        # 2. Readout layer
        # x_s_global_ave = torch.cat(x_s_global_ave,-1)
        # x = torch.cat([data.x_s.view(batch,-1),x_s_global_ave], -1)
        x = data.x_s.view(batch,-1)
        # print(x.shape)
        
        # 3. Apply a final classifier
        x = self.lin1(x)
        x = self.bn1(x)
        x = x.relu()
        x = self.dropout1(x)
        
        x = self.lin2(x)
        x = self.bn2(x)
        x = x.relu()
        x = self.dropout2(x)
        
        if if_att:
            return self.lin3(x), data.x_s.view(batch,-1)
        else:
            return self.lin3(x) 
        
###############################################################################
class GIN(torch.nn.Module):
    def __init__(self, GCN_channels=[32, 32], train_eps=True, roi_num=268, dropout_ratio=0.5,
                 leaky_slope=0.33, num_classes=1):
        super(GIN, self).__init__()
        self.GCN_channels = GCN_channels
        
        gcn_insize = roi_num
        for i, gcn_outsize in enumerate(GCN_channels):
            mlp = nn.Sequential(
                Linear(gcn_insize, gcn_outsize),
                nn.LeakyReLU(negative_slope=leaky_slope, inplace=True),
                Linear(gcn_outsize, gcn_outsize),)
                
                
            fc = gnn.Sequential('x, edge_index', [
                (GINConv(mlp, train_eps=train_eps),
                 'x, edge_index -> x'),
                gnn.BatchNorm(gcn_outsize),
                nn.LeakyReLU(negative_slope=leaky_slope, inplace=True),
                (nn.Dropout(p=dropout_ratio), 'x -> x'),
            ])

            setattr(self, 'gcnfc%d' % i, fc)
            gcn_insize = gcn_outsize
            
        self.lin1 = Linear(GCN_channels[-1]*2*2, 256)
        self.lin2 = Linear(256, 128)
        self.lin3 = Linear(128, 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        self.dropout1 = nn.Dropout(dropout_ratio)
        self.dropout2 = nn.Dropout(dropout_ratio)

    def forward(self, x, edge_index, batch):

        # 1. Obtain node embeddings 
        xs = []
        for i, _ in enumerate(self.GCN_channels):
            fc = getattr(self, 'gcnfc%d' % i)
            x = fc(x, edge_index)
            xs.append(torch.cat( (global_max_pool(x, batch), global_mean_pool(x, batch)), -1))

        # 2. Readout layer
        x = torch.cat( (xs[0], xs[1]), -1)  # [batch_size, hidden_channels]
        # 3. Apply a final classifier
        x = self.lin1(x)
        x = self.leaky_relu(self.bn1(x))
        x = self.dropout1(x)
        x = self.lin2(x)
        x = self.leaky_relu(self.bn2(x))
        x = self.dropout2(x)
        
        return self.lin3(x)
    
###############################################################################
class GAT(torch.nn.Module):
    def __init__(self, GCN_channels=[32, 32], heads=2, roi_num=268, dropout_ratio=0.5,
                 leaky_slope=0.33, num_classes=1):
        super(GAT, self).__init__()
        self.GCN_channels = GCN_channels
        
        gcn_insize = roi_num
        for i, gcn_outsize in enumerate(GCN_channels):
            fc = gnn.Sequential('x, edge_index', [
                (GATConv(gcn_insize, gcn_outsize, heads=heads),
                 'x, edge_index -> x'),
                gnn.BatchNorm(gcn_outsize*heads),
                nn.LeakyReLU(negative_slope=leaky_slope, inplace=True),
                (nn.Dropout(p=dropout_ratio), 'x -> x'),
            ])

            setattr(self, 'gcnfc%d' % i, fc)
            gcn_insize = gcn_outsize*heads
            
        self.lin1 = Linear(GCN_channels[-1]*heads*2*2, 256)
        self.lin2 = Linear(256, 128)
        self.lin3 = Linear(128, 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        self.dropout1 = nn.Dropout(dropout_ratio)
        self.dropout2 = nn.Dropout(dropout_ratio)

    def forward(self, x, edge_index, batch):

        # 1. Obtain node embeddings 
        xs = []
        for i, _ in enumerate(self.GCN_channels):
            fc = getattr(self, 'gcnfc%d' % i)
            x = fc(x, edge_index)
            xs.append(torch.cat( (global_max_pool(x, batch), global_mean_pool(x, batch)), -1))

        # 2. Readout layer
        x = torch.cat( (xs[0], xs[1]), -1)  # [batch_size, hidden_channels]
        # 3. Apply a final classifier
        x = self.lin1(x)
        x = self.leaky_relu(self.bn1(x))
        x = self.dropout1(x)
        x = self.lin2(x)
        x = self.leaky_relu(self.bn2(x))
        x = self.dropout2(x)
        
        return self.lin3(x)
    
# readout dropout
# class Hodge_SpatialConv_Pool(torch.nn.Module):
#     def __init__(self, Edge_channels=[32, 32, 1], mlp_channels=[256,128], K=4, roi_num=268, 
#                  edge_num=8978, time_point=375, num_classes=1, leaky_slope=0.33,
#                  dropout_ratio=0.5):
#         super(Hodge_SpatialConv_Pool, self).__init__()
#         self.Edge_channels = Edge_channels
#         self.roi_num = roi_num
#         self.edge_num = edge_num
#         self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
#         self.relu = nn.ReLU(inplace=True)

#         gcn_insize = 1
#         for i, gcn_outsize in enumerate(Edge_channels):
#             fc = gnn.Sequential('x, edge_index, edge_weight', [
#                 (HodgeLaguerreConv(gcn_insize, gcn_outsize, K=K),
#                  'x, edge_index, edge_weight -> x'),
#                 gnn.BatchNorm(gcn_outsize),
#                 nn.LeakyReLU(negative_slope=leaky_slope, inplace=True),
#                 (Dropout(p=dropout_ratio), 'x -> x'),
#             ])

#             setattr(self, 'esc%d' % i, fc)
#             gcn_insize = gcn_outsize
        
#         lin_insize = int(edge_num/2) #sum(Edge_channels)
#         self.lin1 = Linear(lin_insize, mlp_channels[0])
#         self.lin2 = Linear(mlp_channels[0], mlp_channels[1])
#         self.lin3 = Linear(mlp_channels[1], num_classes)
#         self.bn1 = nn.BatchNorm1d(mlp_channels[0])
#         self.bn2 = nn.BatchNorm1d(mlp_channels[1])
#         self.dropout1 = nn.Dropout(dropout_ratio)
#         self.dropout2 = nn.Dropout(dropout_ratio)
#         self.spool = Graclus_Pooling()

#     def forward(self, data, edge_index_s1, edge_weight_s1):

#         # 1. Obtain node embeddings
#         # data_t = Data(x=data.x_t, edge_index=data.edge_index_t, edge_weight=data.edge_weight_t)
#         # data = Data(x=data.x_s, edge_index=data.edge_index_s, edge_weight=data.edge_weight_s)
        
#         batch = int(data.x_t.shape[0] / self.roi_num)
#         s_batch = torch.tensor([[i]*self.edge_num for i in range(batch)])
#         s_batch = s_batch.view(-1).to(device)

#         # x_s_global_ave = []
#         for i, _ in enumerate(self.Edge_channels):
#             fc = getattr(self, 'esc%d' % i)
#             data.x_s = fc(data.x_s, data.edge_index_s, data.edge_weight_s)
#             # x_s_global_ave.append(global_mean_pool(data.x_s, s_batch))
#             if i == 0:#< len(self.Edge_channels)-1:
#                 # print(x_s.shape)
#                 data, s_batch = self.spool(data, s_batch, edge_index_s1, edge_weight_s1)

#         # 2. Readout layer
#         # x = torch.cat(x_s_global_ave,-1)
#         x = data.x_s.view(batch,-1)
#         # print(x.shape)
        
#         # 3. Apply a final classifier
#         x = self.lin1(x)
#         x = self.bn1(x)
#         x = x.relu()
#         x = self.dropout1(x)
        
#         x = self.lin2(x)
#         x = self.bn2(x)
#         x = x.relu()
#         x = self.dropout2(x)
   
#         return self.lin3(x)
    
###############################################################################
class Hodge_SpatialConv_Linear_Readout(torch.nn.Module):
    def __init__(self, Edge_channels=[32, 32], mlp_channels=[256,128], K=4, roi_num=268, 
                 edge_num=8978, time_point=375, num_classes=1, leaky_slope=0.33,
                 dropout_ratio=0.5):
        super(Hodge_SpatialConv_Linear_Readout, self).__init__()
        self.Edge_channels = Edge_channels
        self.roi_num = roi_num
        self.edge_num = edge_num
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        self.relu = nn.ReLU(inplace=True)

        gcn_insize = 1
        for i, gcn_outsize in enumerate(Edge_channels):
            fc = gnn.Sequential('x, edge_index, edge_weight', [
                (HodgeLaguerreConv(gcn_insize, gcn_outsize, K=K),
                 'x, edge_index, edge_weight -> x'),
                gnn.BatchNorm(gcn_outsize),
                nn.LeakyReLU(negative_slope=leaky_slope, inplace=True),
                (Dropout(p=dropout_ratio), 'x -> x'),
            ])

            setattr(self, 'esc%d' % i, fc)
            gcn_insize = gcn_outsize
        
        lin_insize = int(edge_num/2) #sum(Edge_channels)
        self.lin1 = Linear(lin_insize, mlp_channels[0])
        self.lin2 = Linear(mlp_channels[0], mlp_channels[1])
        self.lin3 = Linear(mlp_channels[1], num_classes)
        
        self.lin0 = Linear(32, 1)
        self.bn0 = nn.BatchNorm1d(32)
        
        self.bn1 = nn.BatchNorm1d(mlp_channels[0])
        self.bn2 = nn.BatchNorm1d(mlp_channels[1])
        self.dropout0 = nn.Dropout(dropout_ratio)
        self.dropout1 = nn.Dropout(dropout_ratio)
        self.dropout2 = nn.Dropout(dropout_ratio)
        self.spool = Graclus_Pooling()

    def forward(self, data, edge_index_s1, edge_weight_s1):

        # 1. Obtain node embeddings
        # data_t = Data(x=data.x_t, edge_index=data.edge_index_t, edge_weight=data.edge_weight_t)
        # data = Data(x=data.x_s, edge_index=data.edge_index_s, edge_weight=data.edge_weight_s)
        
        batch = int(data.x_t.shape[0] / self.roi_num)
        s_batch = torch.tensor([[i]*self.edge_num for i in range(batch)])
        s_batch = s_batch.view(-1).to(device)

        # x_s_global_ave = []
        for i, _ in enumerate(self.Edge_channels):
            fc = getattr(self, 'esc%d' % i)
            data.x_s = fc(data.x_s, data.edge_index_s, data.edge_weight_s)
            # x_s_global_ave.append(global_mean_pool(data.x_s, s_batch))
            if i == 0:#< len(self.Edge_channels)-1:
                # print(x_s.shape)
                data, s_batch = self.spool(data, s_batch, edge_index_s1, edge_weight_s1)

        # 2. Readout layer
        # x = torch.cat(x_s_global_ave,-1)
        
        # data.x_s = self.bn0(self.lin0(data.x_s))
        data.x_s = self.relu(self.lin0(data.x_s))
        x = data.x_s.view(batch,-1)
        # print(x.shape)
        
        # 3. Apply a final classifier
        x = self.dropout0(x)
        
        x = self.lin1(x)
        x = self.bn1(x)
        x = x.relu()
        x = self.dropout1(x)
        
        x = self.lin2(x)
        x = self.bn2(x)
        x = x.relu()
        x = self.dropout2(x)
   
        return self.lin3(x)
    
    

###############################################################################
class Hodge_SpatialConv_Pool2(torch.nn.Module):
    def __init__(self, Edge_channels=[32, 32, 1], mlp_channels=[256,128], K=4, roi_num=268, 
                 edge_num=8978, time_point=375, num_classes=1, leaky_slope=0.33,
                 dropout_ratio=0.5):
        super(Hodge_SpatialConv_Pool2, self).__init__()
        self.Edge_channels = Edge_channels
        self.roi_num = roi_num
        self.edge_num = edge_num
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_slope)
        self.relu = nn.ReLU()

        gcn_insize = 1
        for i, gcn_outsize in enumerate(Edge_channels):
            fc = gnn.Sequential('x, edge_index, edge_weight', [
                (HodgeLaguerreConv(gcn_insize, gcn_outsize, K=K),
                 'x, edge_index, edge_weight -> x'),
                gnn.BatchNorm(gcn_outsize),
                nn.LeakyReLU(negative_slope=leaky_slope),
                (Dropout(p=dropout_ratio), 'x -> x'),
            ])

            setattr(self, 'esc%d' % i, fc)
            gcn_insize = gcn_outsize
        
        lin_insize = 2244+sum(Edge_channels[:-1])#int(edge_num/2) #sum(Edge_channels)
        self.lin1 = Linear(lin_insize, mlp_channels[0])
        self.lin2 = Linear(mlp_channels[0], mlp_channels[1])
        self.lin3 = Linear(mlp_channels[1], num_classes)
        self.bn1 = nn.BatchNorm1d(mlp_channels[0])
        self.bn2 = nn.BatchNorm1d(mlp_channels[1])
        self.dropout1 = nn.Dropout(dropout_ratio)
        self.dropout2 = nn.Dropout(dropout_ratio)
        self.spool0 = Graclus_Pooling()
        self.spool1 = Graclus_Pooling_Permutation()

    def forward(self, data, edge_index_s1, edge_weight_s1, edge_index_s2, edge_weight_s2, idx_dic1):

        # 1. Obtain node embeddings
        # data_t = Data(x=data.x_t, edge_index=data.edge_index_t, edge_weight=data.edge_weight_t)
        # data = Data(x=data.x_s, edge_index=data.edge_index_s, edge_weight=data.edge_weight_s)
        
        batch = int(data.x_t.shape[0] / self.roi_num)
        s_batch = torch.tensor([[i]*self.edge_num for i in range(batch)])
        s_batch = s_batch.view(-1).to(device)

        x_s_global_ave = []
        for i, _ in enumerate(self.Edge_channels):
            fc = getattr(self, 'esc%d' % i)
            data.x_s = fc(data.x_s, data.edge_index_s, data.edge_weight_s)
            
            if i == 0:#< len(self.Edge_channels)-1:
                # print(x_s.shape)
                data, s_batch = self.spool0(data, s_batch, edge_index_s1, edge_weight_s1)
                x_s_global_ave.append(global_mean_pool(data.x_s, s_batch))
            elif i == 1:
                data, s_batch = self.spool1(data, s_batch, edge_index_s2, edge_weight_s2, idx_dic1)
                x_s_global_ave.append(global_mean_pool(data.x_s, s_batch))

        # 2. Readout layer
        x_s_global_ave = torch.cat(x_s_global_ave,-1)
        x = torch.cat([data.x_s.view(batch,-1),x_s_global_ave], -1)
        # print(x.shape)
        
        # 3. Apply a final classifier
        x = self.lin1(x)
        x = self.bn1(x)
        x = x.relu()
        x = self.dropout1(x)
        
        x = self.lin2(x)
        x = self.bn2(x)
        x = x.relu()
        x = self.dropout2(x)
   
        return self.lin3(x)    
    
###############################################################################
class Hodge_ChebConv_Pool(torch.nn.Module):
    def __init__(self, Edge_channels=[32, 32, 1], mlp_channels=[256,128], K=4, roi_num=268, 
                 edge_num=8978, time_point=375, num_classes=1, leaky_slope=0.33,
                 dropout_ratio=0.5):
        super(Hodge_ChebConv_Pool, self).__init__()
        self.Edge_channels = Edge_channels
        self.roi_num = roi_num
        self.edge_num = edge_num
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        self.relu = nn.ReLU(inplace=True)

        gcn_insize = 1
        for i, gcn_outsize in enumerate(Edge_channels):
            fc = gnn.Sequential('x, edge_index, edge_weight', [
                (HodgeChebConv(gcn_insize, gcn_outsize, K=K),
                 'x, edge_index, edge_weight -> x'),
                gnn.BatchNorm(gcn_outsize),
                nn.LeakyReLU(negative_slope=leaky_slope, inplace=True),
                (Dropout(p=dropout_ratio), 'x -> x'),
            ])

            setattr(self, 'esc%d' % i, fc)
            gcn_insize = gcn_outsize
        
        lin_insize = int(edge_num/2) #sum(Edge_channels)
        self.lin1 = Linear(lin_insize, mlp_channels[0])
        self.lin2 = Linear(mlp_channels[0], mlp_channels[1])
        self.lin3 = Linear(mlp_channels[1], num_classes)
        self.bn1 = nn.BatchNorm1d(mlp_channels[0])
        self.bn2 = nn.BatchNorm1d(mlp_channels[1])
        self.dropout1 = nn.Dropout(dropout_ratio)
        self.dropout2 = nn.Dropout(dropout_ratio)
        self.spool = Graclus_Pooling()

    def forward(self, data, edge_index_s1, edge_weight_s1):

        # 1. Obtain node embeddings
        # data_t = Data(x=data.x_t, edge_index=data.edge_index_t, edge_weight=data.edge_weight_t)
        # data = Data(x=data.x_s, edge_index=data.edge_index_s, edge_weight=data.edge_weight_s)
        
        batch = int(data.x_t.shape[0] / self.roi_num)
        s_batch = torch.tensor([[i]*self.edge_num for i in range(batch)])
        s_batch = s_batch.view(-1).to(device)
        data.edge_index_s, data.edge_weight_s = add_self_loops(data.edge_index_s, data.edge_weight_s, fill_value=-1.)
        data.edge_index_s, data.edge_weight_s = coalesce(data.edge_index_s, data.edge_weight_s)

        # x_s_global_ave = []
        for i, _ in enumerate(self.Edge_channels):
            fc = getattr(self, 'esc%d' % i)
            data.x_s = fc(data.x_s, data.edge_index_s, data.edge_weight_s)
            # x_s_global_ave.append(global_mean_pool(data.x_s, s_batch))
            if i == 0:#< len(self.Edge_channels)-1:
                # print(x_s.shape)
                data, s_batch = self.spool(data, s_batch, edge_index_s1, edge_weight_s1)
                data.edge_index_s, data.edge_weight_s = add_self_loops(data.edge_index_s, data.edge_weight_s, fill_value=-1.)
                data.edge_index_s, data.edge_weight_s = coalesce(data.edge_index_s, data.edge_weight_s)

        # 2. Readout layer
        # x = torch.cat(x_s_global_ave,-1)
        x = data.x_s.view(batch,-1)
        # print(x.shape)
        
        # 3. Apply a final classifier
        x = self.lin1(x)
        x = self.bn1(x)
        x = x.relu()
        x = self.dropout1(x)
        
        x = self.lin2(x)
        x = self.bn2(x)
        x = x.relu()
        x = self.dropout2(x)
   
        return self.lin3(x)
    
    
###############################################################################
###############################  Backup  ######################################
###############################################################################

# class Hodge_STConv_Pool(torch.nn.Module):
#     def __init__(self, Spatial_channels=[8, 16], Temporal_channels=[4, 8], 
#                  Edge_channels=[32, 32, 1], mlp_channels=[256,128], K=4, roi_num=268, 
#                  edge_num=8978, time_point=375, num_classes=1, leaky_slope=0.33,
#                  dropout_ratio=0.5):
#         super(Hodge_STConv_Pool, self).__init__()
#         self.Spatial_channels = Spatial_channels
#         self.Temporal_channels = Temporal_channels
#         self.Edge_channels = Edge_channels
#         self.roi_num = roi_num
#         self.edge_num = edge_num
#         self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
#         self.relu = nn.ReLU(inplace=True)
#         self.time_point=time_point
        
#         gcn_insize = 1
#         for i, TC in enumerate(Temporal_channels):
#             if_dim_reduction = i!=0;
#             fc = Inception1D(gcn_insize,TC,maxpool=5,if_dim_reduction=if_dim_reduction)
#             setattr(self, 'ntc%d' % i, fc) # node temporal convolution
#             gcn_insize = fc.out_size
#             time_point = int(np.ceil(time_point/4))
            
#             fc = HodgeChebConv(gcn_insize, Spatial_channels[i], K=K)
#             setattr(self, 'nsc%d' % i, fc) # node spatial convolution
            
#             fc = gnn.BatchNorm(Spatial_channels[i]*time_point)
#             setattr(self, 'nbn%d' % i, fc) # node batch normalization
            
#             fc = Dropout(p=dropout_ratio)
#             setattr(self, 'ndo%d' % i, fc) # node dropout layer
#             gcn_insize = Spatial_channels[i]

#         gcn_insize = 1
#         for i, gcn_outsize in enumerate(Edge_channels):
#             fc = gnn.Sequential('x, edge_index, edge_weight', [
#                 (HodgeLaguerreConv(gcn_insize, gcn_outsize, K=K),
#                  'x, edge_index, edge_weight -> x'),
#                 gnn.BatchNorm(gcn_outsize),
#                 nn.LeakyReLU(negative_slope=leaky_slope, inplace=True),
#                 (Dropout(p=dropout_ratio), 'x -> x'),
#             ])

#             setattr(self, 'esc%d' % i, fc)
#             gcn_insize = gcn_outsize
        
#         lin_insize = int(Spatial_channels[-1]*self.roi_num/2) + int(edge_num/2)
#         self.lin1 = Linear(lin_insize, mlp_channels[0])
#         self.lin2 = Linear(mlp_channels[0], mlp_channels[1])
#         self.lin3 = Linear(mlp_channels[1], num_classes)
#         self.bn1 = nn.BatchNorm1d(mlp_channels[0])
#         self.bn2 = nn.BatchNorm1d(mlp_channels[1])
#         self.dropout1 = nn.Dropout(dropout_ratio)
#         self.dropout2 = nn.Dropout(dropout_ratio)
#         self.nspool = Graclus_Node_Pooling() # node spatial pooling
#         self.spool = Graclus_Pooling() # edge spatial pooling
        

#     def forward(self, data, edge_index_t1, edge_weight_t1, edge_index_s1, edge_weight_s1):

#         # 1. Obtain node embeddings
        
#         ## node convolution
#         batch = int(data.x_t.shape[0] / self.roi_num)
#         t_batch = torch.tensor([[i]*self.roi_num for i in range(batch)])
#         t_batch = t_batch.view(-1).to(device)
#         data.x_t = data.x_t.view(-1,self.time_point,1)
        
#         for i, _ in enumerate(self.Temporal_channels):
#             fc = getattr(self, 'ntc%d' % i)
#             data.x_t = fc(data.x_t)
#             fc = getattr(self, 'nsc%d' % i)
#             data.x_t = fc(data.x_t, data.edge_index_t, data.edge_weight_t)
#             if i == 0:
#                 data, t_batch = self.nspool(data, t_batch, edge_index_t1, edge_weight_t1)

#             # change dim to N*(T*C) and apply batch normalization
#             fc = getattr(self, 'nbn%d' % i)
#             t_shape = data.x_t.shape
#             data.x_t = data.x_t.view(t_shape[0],-1)
#             data.x_t = self.leaky_relu(fc(data.x_t))
#             data.x_t = data.x_t.view(t_shape[0],t_shape[1],t_shape[2])
#             fc = getattr(self, 'ndo%d' % i)
#             data.x_t = fc(data.x_t)
        
#         # edge convolution
#         s_batch = torch.tensor([[i]*self.edge_num for i in range(batch)])
#         s_batch = s_batch.view(-1).to(device)

#         # x_s_global_ave = []
#         for i, _ in enumerate(self.Edge_channels):
#             fc = getattr(self, 'esc%d' % i)
#             data.x_s = fc(data.x_s, data.edge_index_s, data.edge_weight_s)
#             # x_s_global_ave.append(global_mean_pool(data.x_s, s_batch))
#             if i == 0:#< len(self.Edge_channels)-1:
#                 # print(x_s.shape)
#                 data, s_batch = self.spool(data, s_batch, edge_index_s1, edge_weight_s1)

#         # 2. Readout layer
#         data.x_t = data.x_t.mean(dim=1)
#         x = torch.cat((data.x_t.view(data.num_graphs,-1), data.x_s.view(batch,-1)), -1)
#         # print(x.shape)
        
#         # 3. Apply a final classifier
#         x = self.lin1(x)
#         x = self.bn1(x)
#         x = x.relu()
#         x = self.dropout1(x)
        
#         x = self.lin2(x)
#         x = self.bn2(x)
#         x = x.relu()
#         x = self.dropout2(x)
   
#         return self.lin3(x)