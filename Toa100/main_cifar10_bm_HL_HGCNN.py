#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 14:32:41 2023

@author: jinghan
"""

import numpy as np
from torch.nn import Linear, Dropout
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch_geometric.nn as gnn
import torch.nn as nn
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import time
from lib.Hodge_Cheb_Conv import *
from lib.Hodge_Dataset import *
from lib.Hodge_ST_Model import *
import torchvision as tv


###############################################################################
##################### Only Edge Convolution & Pooling #########################
###############################################################################


###############################################################################
##################### Only Edge Convolution & Pooling #########################
###############################################################################
            
class HL_HGCNN(torch.nn.Module):
    def __init__(self, channels=[2,2,2,2], filters=[64,128,256,512], mlp_channels=[], K=2, node_num=1024, 
                 edge_num=3906, num_classes=10, dropout_ratio=0.0, dropout_ratio_mlp=0.0, alpha=0.5, beta=1.0):
        super(HL_HGCNN, self).__init__()
        self.channels = channels
        self.filters = filters#[]
        self.mlp_channels = mlp_channels
        self.node_num = node_num
        self.edge_num = edge_num
        self.initial_channel = self.filters[0]
        self.alpha = alpha
        self.beta = beta
        # self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_slope)
        # self.relu = nn.ReLU()
        
        gcn_insize = 9
        layers = [(HodgeLaguerreConv(gcn_insize, self.initial_channel, K=K),
                   'x, edge_index, edge_weight -> x'),
                  gnn.BatchNorm(self.initial_channel),
                  nn.ReLU()]
        fc = gnn.Sequential('x, edge_index, edge_weight', layers)

        setattr(self, 'HL_EC', fc)
        gcn_insize = self.initial_channel
            
        for i, gcn_outsize in enumerate(self.filters):
            temp = [(HodgeLaguerreConv(gcn_insize, gcn_outsize, K=K),
                       'x, edge_index, edge_weight -> x'),
                      gnn.BatchNorm(gcn_outsize),
                      nn.ReLU(),
                      (lambda x1, x2: (1-alpha)*x1+alpha*x2, 'x, x0 -> x'),]
            layers = [(HodgeLaguerreConv(gcn_outsize, gcn_outsize, K=K),
                       'x, edge_index, edge_weight -> x'),
                      gnn.BatchNorm(gcn_outsize),
                      nn.ReLU(),
                      (lambda x1, x2: (1-alpha)*x1+alpha*x2, 'x, x0 -> x'),]
            for j in range(self.channels[i]-1):
                temp.extend(layers)
            temp.extend([(Dropout(p=dropout_ratio), 'x -> x'),])
            fc = gnn.Sequential('x, edge_index, edge_weight, x0', temp)
            setattr(self, 'HL_EC%d' % i, fc)
            gcn_insize = gcn_outsize
            
            if i < len(self.filters)-1:
                # res project
                layers = [(HodgeLaguerreConv(self.filters[i], self.filters[i+1], K=1),
                           'x, edge_index, edge_weight -> x'),]
                fc = gnn.Sequential('x, edge_index, edge_weight', layers)
                setattr(self, 'e0_proj%d' % i, fc)
                # int term
                layers = [(HodgeLaguerreConv(gcn_insize*2, gcn_insize, K=1),
                           'x, edge_index, edge_weight -> x'),
                          gnn.BatchNorm(gcn_insize),
                          nn.ReLU(),
                          (HodgeLaguerreConv(gcn_insize, gcn_insize, K=1),
                           'x, edge_index, edge_weight -> x'),
                          gnn.BatchNorm(gcn_insize),
                          nn.ReLU()]
                fc = gnn.Sequential('x, edge_index, edge_weight', layers)
                setattr(self, 'int_e2n%d' % i, fc)
                
        
        gcn_insize = 5
        layers = [(HodgeLaguerreConv(gcn_insize, self.initial_channel, K=K),
                   'x, edge_index, edge_weight -> x'),
                  gnn.BatchNorm(self.initial_channel),
                  nn.ReLU()]
        fc = gnn.Sequential('x, edge_index, edge_weight', layers)

        setattr(self, 'HL_NC', fc)
        gcn_insize = self.initial_channel
        
        for i, gcn_outsize in enumerate(self.filters):
            temp = [(HodgeLaguerreConv(gcn_insize, gcn_outsize, K=K),
                       'x, edge_index, edge_weight -> x'),
                      gnn.BatchNorm(gcn_outsize),
                      nn.ReLU(),
                      (lambda x1, x2: (1-alpha)*x1+alpha*x2, 'x, x0 -> x'),]
            layers = [(HodgeLaguerreConv(gcn_outsize, gcn_outsize, K=K),
                       'x, edge_index, edge_weight -> x'),
                      gnn.BatchNorm(gcn_outsize),
                      nn.ReLU(),
                      (lambda x1, x2: (1-alpha)*x1+alpha*x2, 'x, x0 -> x'),]
            for j in range(self.channels[i]-1):
                temp.extend(layers)
            temp.extend([(Dropout(p=dropout_ratio), 'x -> x'),])
            fc = gnn.Sequential('x, edge_index, edge_weight, x0', temp)
            setattr(self, 'HL_NC%d' % i, fc)
            gcn_insize = gcn_outsize
            
            if i < len(self.filters)-1:
                # res project
                layers = [(HodgeLaguerreConv(self.filters[i], self.filters[i+1], K=1),
                           'x, edge_index, edge_weight -> x'),]
                fc = gnn.Sequential('x, edge_index, edge_weight', layers)
                setattr(self, 'n0_proj%d' % i, fc)
                # int term
                layers = [(HodgeLaguerreConv(gcn_insize*2, gcn_insize, K=1),
                           'x, edge_index, edge_weight -> x'),
                          gnn.BatchNorm(gcn_insize),
                          nn.ReLU(),
                          (HodgeLaguerreConv(gcn_insize, gcn_insize, K=1),
                           'x, edge_index, edge_weight -> x'),
                          gnn.BatchNorm(gcn_insize),
                          nn.ReLU()]
                fc = gnn.Sequential('x, edge_index, edge_weight', layers)
                setattr(self, 'int_n2e%d' % i, fc)
            
        
        mlp_insize = self.filters[-1] * 2 #sum(Node_channels)+ sum(Edge_channels)#[-1]
        for i, mlp_outsize in enumerate(mlp_channels):
            fc = nn.Sequential(
                Linear(mlp_insize, mlp_outsize),
                nn.BatchNorm1d(mlp_outsize),
                nn.ReLU(),
                nn.Dropout(dropout_ratio_mlp),
                )
            setattr(self, 'mlp%d' % i, fc)
            mlp_insize = mlp_outsize

        self.out = Linear(mlp_insize, num_classes)


    def forward(self, data):

        # 1. Obtain node embeddings
        n_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(data.num_node1)], dim=-1)
        n_batch = n_batch.to(device)
        
        s_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(data.num_edge1)], dim=-1)
        s_batch = s_batch.to(device)
        x_s, edge_index_s, edge_weight_s = data.x_s, data.edge_index_s, data.edge_weight_s
        x_t, edge_index_t, edge_weight_t = data.x_t, data.edge_index_t, data.edge_weight_t
        edge_index = data.edge_index
        
        x_s = self.HL_EC(x_s, edge_index_s, edge_weight_s)
        x_t = self.HL_NC(x_t, edge_index_t, edge_weight_t)
        x_s0, x_t0 = x_s, x_t

        for i, _ in enumerate(self.channels):
                
            fc = getattr(self, 'HL_NC%d' % i)
            # print(x_t.shape, x_t0.shape)
            x_t = fc(x_t, edge_index_t, edge_weight_t, x_t0)
            
            fc = getattr(self, 'HL_EC%d' % i)
            x_s = fc(x_s, edge_index_s, edge_weight_s, x_s0)
     
            if i < len(self.channels)-1:
                par_i = adj2par1(data.edge_index, x_t.shape[0], x_s.shape[0])
                D = degree(data.edge_index.view(-1))
                temp_xt = (1/D).view(-1,1)*torch.sparse.mm(par_i.abs(), x_s)
                temp_xs = torch.sparse.mm(par_i.abs().transpose(0,1), x_t)/2
                x_t = torch.cat([x_t, temp_xt], dim=-1)
                x_s = torch.cat([x_s, temp_xs], dim=-1)
              
                fc = getattr(self, 'int_e2n%d' % i)
                x_t = fc(x_t, edge_index_t, edge_weight_t)
                fc = getattr(self, 'int_n2e%d' % i)
                x_s = fc(x_s, edge_index_s, edge_weight_s)
                fc = getattr(self, 'n0_proj%d' % i)
                x_t0 = fc(x_t0, edge_index_t, edge_weight_t)
                fc = getattr(self, 'e0_proj%d' % i)
                x_s0 = fc(x_s0, edge_index_s, edge_weight_s)
                
        # 2. Readout layer
        x = torch.cat( (global_mean_pool(x_s, s_batch),global_mean_pool(x_t, n_batch)), -1)
        
        # 3. Apply a final classifier
        for i, _ in enumerate(self.mlp_channels):
            fc = getattr(self, 'mlp%d' % i)
            x = fc(x)

        return self.out(x)


###############################################################################
################################# Train & Test ################################
###############################################################################

def train(loader):
    model.train()
    total_loss = 0
    for data in loader:  # Iterate in batches over the training dataset.
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data.y)#.view(-1,1))  # Compute the loss.
        # loss = weighted_mse_loss(out, data.y.view(-1,1))
        
        total_loss += loss*data.num_graphs
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
    return total_loss/len(loader.dataset)


def test(loader):
     model.eval()
     # y_pred = torch.zeros(len(loader.dataset))
     # y = torch.zeros(len(loader.dataset))
    #  y_pred, y = [], []
     total_loss, acc = 0, 0
     
     for data in loader:  # Iterate in batches over the training/test dataset. 
         data = data.to(device)
         with torch.no_grad():
            out = model(data) 
         loss = criterion(out, data.y)#.view(-1,1))  # Compute the loss.
         acc += torch.count_nonzero(torch.argmax(out,dim=1) == data.y)
         total_loss += loss * data.num_graphs
        #  y_pred.extend(out.squeeze())
        #  y.extend(data.y.squeeze())
        
    #  y_pred, y = torch.tensor(y_pred), torch.tensor(y)
    #  y_pred = (y_pred - y_pred.mean()) / y_pred.std()
    #  y = (y - y.mean()) / y.std() 
     return total_loss/len(loader.dataset), acc/len(loader.dataset)


if __name__ == '__main__':
    # test degree based spatial pooling
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print(device)
    start = time.time()
    
    for fold in [0]:#range(5):
        print('Fold {} begin'.format(fold))
    
        # model = HL_HGCNN(channels=[3,4,6,3], filters=[64,128,256,512], mlp_channels=[], K=2, dropout_ratio=0.3, 
        #                   dropout_ratio_mlp=0.0).to(device)  # 2conv
        # save_path = './weights/HGCNN_res_int_pyr_ZINC_3463conv_mlp0_FOLD{}.pt'.format(fold)

        model = HL_HGCNN(channels=[2,2,2,2], filters=[64,128,256,512], mlp_channels=[], K=6, dropout_ratio=0.3, 
                          dropout_ratio_mlp=0.0).to(device)  # 2conv
        save_path = './weights/HGCNN_res_int_pyr_CIFAR10BM_2222conv_k6_mlp0_FOLD{}.pt'.format(fold)
        
        # model = HL_HGCNN(channels=[2,2,2,2], filters=[64,128,256,512], mlp_channels=[], K=5, dropout_ratio=0.3, 
        #                   dropout_ratio_mlp=0.0).to(device)  # 2conv
        # save_path = './weights/HGCNN_res_int_pyr_ZINC_2222conv_k5_mlp0_FOLD{}.pt'.format(fold)
        
        # model = HL_HGCNN(channels=[2,2,2,2], filters=[64,128,256,512], mlp_channels=[], K=4, dropout_ratio=0.3, 
        #                   dropout_ratio_mlp=0.0).to(device)  # 2conv
        # save_path = './weights/HGCNN_res_int_pyr_ZINC_2222conv_k4_mlp0_FOLD{}.pt'.format(fold)
        
        # model = HL_HGCNN(channels=[2,2,2,2], filters=[64,128,256,512], mlp_channels=[], K=2, dropout_ratio=0.3, 
        #                   dropout_ratio_mlp=0.0).to(device)  # 2conv
        # save_path = './weights/HGCNN_res_int_pyr_ZINC_2222conv_mlp0_FOLD{}.pt'.format(fold)
        
        # model.load_state_dict(torch.load(save_path))
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, 
                                                               patience=10, factor=0.5, min_lr=1e-6)
        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        batch_size = 250
        dataset = GNNBenchmarkDataset(root='CIFAR10_bm',name='CIFAR10',split='train')
        trainset = CIFAR_HG_BM_par1(root=osp.join('CIFAR10_bm','train'),dataset=dataset)
        
        dataset = GNNBenchmarkDataset(root='CIFAR10_bm',name='CIFAR10',split='val')
        validset = CIFAR_HG_BM_par1(root=osp.join('CIFAR10_bm','val'), dataset=dataset)
        
        dataset = GNNBenchmarkDataset(root='CIFAR10_bm',name='CIFAR10',split='test')
        testset = CIFAR_HG_BM_par1(root=osp.join('CIFAR10_bm','test'), dataset=dataset)

        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        valid_loader = DataLoader(validset, batch_size=batch_size, num_workers=4)
        test_loader = DataLoader(testset, batch_size=batch_size, num_workers=4)
        
        best_loss, best_acc = test(test_loader)
        print('==================================================================================')
        print(f'Test Loss: {best_loss:.4f}, Test Acc: {best_acc:.4f}')
        print('==================================================================================')
        for epoch in range(1, 600):
            total_loss = train(train_loader)
                
            # train_corr, _, _ = test(train_loader)
            valid_loss, valid_acc = test(valid_loader)
            scheduler.step(total_loss)
            # test_corr, test_loss, test_rmse = test(test_loader)
    
            elapsed = (time.time()-start) / 60
            print(f'Epoch: {epoch:03d}, time: {elapsed:.2f} Train Loss: {total_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}')
            if valid_acc>0.6 and valid_acc>best_acc:
                best_acc = valid_acc
                torch.save(model.state_dict(), save_path)
                print('Model saved! \n')   
                best_loss1, best_acc1 = test(test_loader)
                print('==================================================================================')
                print(f'Test Loss: {best_loss1:.4f}, Test Acc: {best_acc1:.4f}')
                print('==================================================================================')
                
        model.load_state_dict(torch.load(save_path))
        best_loss, best_acc = test(test_loader)
        print('==================================================================================')
        print(f'Test Loss: {best_loss:.4f}, Test Acc: {best_acc:.4f}')
        print('==================================================================================')
        
    # export CUDA_VISIBLE_DEVICES=3






# import numpy as np
# from torch.nn import Linear, Dropout
# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import global_mean_pool, global_max_pool
# import torch_geometric.nn as gnn
# import torch.nn as nn
# from torch.utils.data import Subset
# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader
# import time
# from lib.Hodge_Cheb_Conv import *
# from lib.Hodge_Dataset import *
# from lib.Hodge_ST_Model import *
# import torchvision as tv


# ###############################################################################
# ##################### Only Edge Convolution & Pooling #########################
# ###############################################################################


# ###############################################################################
# ##################### Only Edge Convolution & Pooling #########################
# ###############################################################################
            
# class HL_HGCNN(torch.nn.Module):
#     def __init__(self, Node_channels=[64,64], Edge_channels=[64,64], mlp_channels=[], K=2, node_num=1024, 
#                  edge_num=3906, num_classes=10, dropout_ratio=0.0, dropout_ratio_mlp=0.2, alpha=0.5):
#         super(HL_HGCNN, self).__init__()
#         self.Node_channels = Node_channels
#         self.Edge_channels = Edge_channels
#         self.mlp_channels = mlp_channels
#         self.node_num = node_num
#         self.edge_num = edge_num
#         self.initial_channel = 64
#         self.alpha = alpha
#         # self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_slope)
#         # self.relu = nn.ReLU()
        
#         gcn_insize = 6
#         layers = [(HodgeLaguerreConv(gcn_insize, self.initial_channel, K=3),
#                    'x, edge_index, edge_weight -> x'),
#                   gnn.BatchNorm(self.initial_channel),
#                   nn.ReLU()]
#         fc = gnn.Sequential('x, edge_index, edge_weight', layers)

#         setattr(self, 'HL_EC', fc)
#         gcn_insize = self.initial_channel
            
#         for i, gcn_outsize in enumerate(Edge_channels):
#             layers = [(HodgeLaguerreResConv(gcn_insize, gcn_outsize, K=K),
#                        'x, edge_index, edge_weight -> x'),
#                       gnn.BatchNorm(gcn_outsize),
#                       nn.ReLU(),
#                       (lambda x1, x2: (1-alpha)*x1+alpha*x2, 'x, x0 -> x'),
#                       (HodgeLaguerreResConv(gcn_outsize, gcn_outsize, K=K),
#                        'x, edge_index, edge_weight -> x'),
#                       gnn.BatchNorm(gcn_outsize),
#                       nn.ReLU(),
#                       (lambda x1, x2: (1-alpha)*x1+alpha*x2, 'x, x0 -> x'),
#                       # (Dropout(p=dropout_ratio), 'x -> x'),
#                       ]
            
#             fc = gnn.Sequential('x, edge_index, edge_weight, x0', layers)
#             setattr(self, 'HL_EC%d' % i, fc)
#             gcn_insize = gcn_outsize
        
        
#         gcn_insize = 3
#         layers = [(HodgeLaguerreConv(gcn_insize, self.initial_channel, K=3),
#                    'x, edge_index, edge_weight -> x'),
#                   gnn.BatchNorm(self.initial_channel),
#                   nn.ReLU()]
#         fc = gnn.Sequential('x, edge_index, edge_weight', layers)

#         setattr(self, 'HL_NC', fc)
#         gcn_insize = self.initial_channel
        
#         for i, gcn_outsize in enumerate(Node_channels):
#             layers = [(HodgeLaguerreResConv(gcn_insize, gcn_outsize, K=K),
#                        'x, edge_index, edge_weight -> x'),
#                       gnn.BatchNorm(gcn_outsize),
#                       nn.ReLU(),
#                       (lambda x1, x2: (1-alpha)*x1+alpha*x2, 'x, x0 -> x'),
#                       (HodgeLaguerreResConv(gcn_outsize, gcn_outsize, K=K),
#                        'x, edge_index, edge_weight -> x'),
#                       gnn.BatchNorm(gcn_outsize),
#                       nn.ReLU(),
#                       (lambda x1, x2: (1-alpha)*x1+alpha*x2, 'x, x0 -> x'),
#                       # (Dropout(p=dropout_ratio), 'x -> x'),
#                       ]
            
#             fc = gnn.Sequential('x, edge_index, edge_weight, x0', layers)
#             setattr(self, 'HL_NC%d' % i, fc)
#             gcn_insize = gcn_outsize
            
        
#         mlp_insize = Node_channels[-1] + Edge_channels[-1] #sum(Node_channels)+ sum(Edge_channels)#[-1]
#         for i, mlp_outsize in enumerate(mlp_channels):
#             fc = nn.Sequential(
#                 Linear(mlp_insize, mlp_outsize),
#                 nn.BatchNorm1d(mlp_outsize),
#                 nn.ReLU(),
#                 nn.Dropout(dropout_ratio_mlp),
#                 )
#             setattr(self, 'mlp%d' % i, fc)
#             mlp_insize = mlp_outsize

#         self.out = Linear(mlp_insize, num_classes)


#     def forward(self, data, if_att=False):

#         # 1. Obtain node embeddings
#         # data_t = Data(x=data.x_t, edge_index=data.edge_index_t, edge_weight=data.edge_weight_t)
#         # data = Data(x=data.x_s, edge_index=data.edge_index_s, edge_weight=data.edge_weight_s)
        
#         batch = int(data.x_t.shape[0] / self.node_num)
#         n_batch = torch.tensor([[i]*self.node_num for i in range(batch)])
#         n_batch = n_batch.view(-1).to(device)
        
#         s_batch = torch.tensor([[i]*self.edge_num for i in range(batch)])
#         s_batch = s_batch.view(-1).to(device)
#         x_s, edge_index_s, edge_weight_s = data.x_s.view(-1,6), data.edge_index_s, data.edge_weight_s
#         x_t, edge_index_t, edge_weight_t = data.x_t, data.edge_index_t, data.edge_weight_t
#         edge_index = data.edge_index
        
#         x_s = self.HL_EC(x_s, edge_index_s, edge_weight_s)
#         x_t = self.HL_NC(x_t, edge_index_t, edge_weight_t)
#         x_s0, x_t0 = x_s, x_t
#         # x_s_global_ave = []
#         for i, _ in enumerate(self.Node_channels):
#             fc = getattr(self, 'HL_NC%d' % i)
#             x_t = fc(x_t, edge_index_t, edge_weight_t, x_t0)
            
#             fc = getattr(self, 'HL_EC%d' % i)
#             x_s = fc(x_s, edge_index_s, edge_weight_s, x_s0)
            
#             # x_s_global_ave.append(global_mean_pool(data.x_t, n_batch))

#         # 2. Readout layer
#         # x = torch.cat(x_s_global_ave,-1)
#         x = torch.cat( (global_mean_pool(x_s, s_batch),global_mean_pool(x_t, n_batch)), -1)
        
#         # 2. Readout layer
#         # x = torch.cat(x_s_global_ave,-1)
#         # x = global_mean_pool(data.x_t, n_batch)
        
#         # 3. Apply a final classifier
#         for i, _ in enumerate(self.mlp_channels):
#             fc = getattr(self, 'mlp%d' % i)
#             x = fc(x)
        
#         if if_att:
#             return self.out(x), x_s.view(batch,-1)
#         else:
#             return self.out(x)    


# ###############################################################################
# ################################# Train & Test ################################
# ###############################################################################

# def train(loader):
#     model.train()
#     total_loss = 0
#     for data in loader:  # Iterate in batches over the training dataset.
#         data = data.to(device)
#         out = model(data)
#         loss = criterion(out, data.y)#.view(-1,1))  # Compute the loss.
#         # loss = weighted_mse_loss(out, data.y.view(-1,1))
        
#         total_loss += loss*data.num_graphs
#         loss.backward()  # Derive gradients.
#         optimizer.step()  # Update parameters based on gradients.
#         optimizer.zero_grad()  # Clear gradients.
#     return total_loss/len(loader.dataset)


# def test(loader):
#      model.eval()
#      # y_pred = torch.zeros(len(loader.dataset))
#      # y = torch.zeros(len(loader.dataset))
#     #  y_pred, y = [], []
#      total_loss, acc = 0, 0
     
#      for data in loader:  # Iterate in batches over the training/test dataset. 
#          data = data.to(device)
#          with torch.no_grad():
#             out = model(data) 
#          loss = criterion(out, data.y)#.view(-1,1))  # Compute the loss.
#          acc += torch.count_nonzero(torch.argmax(out,dim=1) == data.y)
#          total_loss += loss * data.num_graphs
#         #  y_pred.extend(out.squeeze())
#         #  y.extend(data.y.squeeze())
        
#     #  y_pred, y = torch.tensor(y_pred), torch.tensor(y)
#     #  y_pred = (y_pred - y_pred.mean()) / y_pred.std()
#     #  y = (y - y.mean()) / y.std() 
#      return total_loss/len(loader.dataset), acc/len(loader.dataset)


# if __name__ == '__main__':
#     # test degree based spatial pooling
#     if torch.cuda.is_available():
#         device = 'cuda:0'
#     else:
#         device = 'cpu'
#     print(device)
#     start = time.time()
    
#     for fold in [0]:#range(5):
#         print('Fold {} begin'.format(fold))
    
#         # model = HL_HGCNN(Edge_channels=[64, 128, 256, 512], Node_channels=[64, 128, 256, 512],
#         #                   mlp_channels=[512], K=2, dropout_ratio=0.0, dropout_ratio_mlp=0.0).to(device)
#         # save_path = './weights/HGCNN_CIFAR10_FOLD{}.pt'.format(fold)
        
#         # model = HL_HGCNN(Edge_channels=[64,64,64,64], Node_channels=[64,64,64,64],
#         #                   mlp_channels=[], K=2, dropout_ratio=0.0, dropout_ratio_mlp=0.0).to(device)  # 4conv
#         # save_path = './weights/HGCNN_res_joint_CIFAR10_conn8_4conv64_mlp0_FOLD{}.pt'.format(fold)
        
#         # model = HL_HGCNN(Edge_channels=[64,64,64], Node_channels=[64,64,64],
#         #                   mlp_channels=[], K=2, dropout_ratio=0.0, dropout_ratio_mlp=0.0).to(device)  # 3conv
#         # save_path = './weights/HGCNN_res_nopool_CIFAR10_BM_3conv64_FOLD{}.pt'.format(fold)
        
#         model = HL_HGCNN(Edge_channels=[64, 64], Node_channels=[64, 64],
#                           mlp_channels=[], K=2, dropout_ratio=0.0, dropout_ratio_mlp=0.0).to(device)  # 2conv mlp0 acc=0.6687
#         save_path = './weights/HGCNN_res_nopool_CIFAR10_BM_2conv64_FOLD{}.pt'.format(fold)
        
#         # model = HL_HGCNN(Edge_channels=[64, 64], Node_channels=[64, 64],
#         #                   mlp_channels=[], K=2, dropout_ratio=0.0, dropout_ratio_mlp=0.0).to(device)  # 2conv mlp0 hp
#         # save_path = './weights/HGCNN_res_joint_CIFAR10_conn8_2conv64_hp_FOLD{}.pt'.format(fold)
        
#         # model.load_state_dict(torch.load(save_path))
#         optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-5)
#         scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
#         # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-3)
#         criterion = torch.nn.CrossEntropyLoss()
#         batch_size = 128
        
#         dataset = GNNBenchmarkDataset(root='CIFAR10_bm',name='CIFAR10',split='train')
#         trainset = CIFAR_HG_BM_par1(root=osp.join('CIFAR10_bm','train'),dataset=dataset)
        
#         dataset = GNNBenchmarkDataset(root='CIFAR10_bm',name='CIFAR10',split='val')
#         validset = CIFAR_HG_BM_par1(root=osp.join('CIFAR10_bm','val'), dataset=dataset)
        
#         dataset = GNNBenchmarkDataset(root='CIFAR10_bm',name='CIFAR10',split='test')
#         testset = CIFAR_HG_BM_par1(root=osp.join('CIFAR10_bm','test'), dataset=dataset)

#         train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
#         valid_loader = DataLoader(validset, batch_size=batch_size, num_workers=4)
#         test_loader = DataLoader(testset, batch_size=batch_size, num_workers=4)
        
#         best_loss, best_acc = test(test_loader)
#         print('==================================================================================')
#         print(f'Test Loss: {best_loss:.4f}, Test Acc: {best_acc:.4f}')
#         print('==================================================================================')
        
#         for epoch in range(1, 600):
#             total_loss = train(train_loader)
#             # train_corr, _, _ = test(train_loader)
#             valid_loss, valid_acc = test(valid_loader)
#             scheduler.step()
#             # test_corr, test_loss, test_rmse = test(test_loader)
    
#             elapsed = (time.time()-start) / 60
#             print(f'Epoch: {epoch:03d}, time: {elapsed:.2f} Train Loss: {total_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}')
#             if epoch>10 and valid_acc>best_acc:
#                 best_acc = valid_acc
#                 torch.save(model.state_dict(), save_path)
#                 print('Model saved! \n')   
#                 best_loss1, best_acc1 = test(test_loader)
#                 print('==================================================================================')
#                 print(f'Test Loss: {best_loss1:.4f}, Test Acc: {best_acc1:.4f}')
#                 print('==================================================================================')
                
#         model.load_state_dict(torch.load(save_path))
#         best_loss, best_acc = test(test_loader)
#         print('==================================================================================')
#         print(f'Test Loss: {best_loss:.4f}, Test Acc: {best_acc:.4f}')
#         print('==================================================================================')
        
    # export CUDA_VISIBLE_DEVICES=3
    
    
 


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Wed Dec 28 00:24:08 2022

# @author: jinghan
# """



# import numpy as np
# from torch.nn import Linear, Dropout
# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import global_mean_pool, global_max_pool
# import torch_geometric.nn as gnn
# import torch.nn as nn
# from torch.utils.data import Subset
# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader
# import time
# from lib.Hodge_Cheb_Conv import *
# from lib.Hodge_Dataset import *
# from lib.Hodge_ST_Model import *
# import torchvision as tv


# ###############################################################################
# ##################### Only Edge Convolution & Pooling #########################
# ###############################################################################
# class JointPooling_Node_attloss(torch.nn.Module):
#     def __init__(
#         self,
#         in_channels_n: int,
#         in_channels_e: int,
#         ratio: Union[float, int] = 0.5,
#         K: int = 4,
#         nonlinearity: Callable = torch.sigmoid,
#         **kwargs,
#     ):
#         super().__init__()
#         self.in_channels_n = in_channels_n
#         self.in_channels_e = in_channels_e
#         self.ratio = ratio
#         self.gnn_n = HodgeLaguerreConv(in_channels_n, 1, K=K)
#         self.gnn_e = HodgeLaguerreConv(in_channels_e, 1, K=K)
#         self.nonlinearity = nonlinearity

#         self.reset_parameters()

#     def reset_parameters(self):
#         self.gnn_n.reset_parameters()
#         self.gnn_e.reset_parameters()


#     def forward(
#         self,
#         x_n: Tensor,
#         edge_index_n: Tensor,
#         edge_weight_n: Tensor,
#         x_e: Tensor,
#         edge_index_e: Tensor,
#         edge_weight_e: Tensor,
#         edge_index: Tensor,
#         n_batch: Tensor,
#         e_batch: Tensor,
#         x_n0: Tensor,
#         x_e0: Tensor,
#     ):
#         """"""

#         score_n = self.gnn_n(x_n, edge_index_n, edge_weight_n).view(-1)
#         score_e = self.gnn_e(x_e, edge_index_e, edge_weight_e).view(-1)
#         score_n = self.nonlinearity(score_n)
#         score_e = self.nonlinearity(score_e)
#         x_n = x_n * score_n.view(-1, 1)
#         x_e = x_e * score_e.view(-1, 1)

#         num_nodes = scatter_add(n_batch.new_ones(x_n.size(0)), n_batch, dim=0) # num of nodes in each sample
#         num_edges = scatter_add(e_batch.new_ones(x_e.size(0)), e_batch, dim=0) # num of edges in each sample
#         batch_size, max_num_nodes = num_nodes.size(0), int(num_nodes.max())
#         # print(edge_index)
#         datas = unbatch_edge_index(edge_index, n_batch)
#         score_n, score_e = ut.unbatch(score_n,n_batch), ut.unbatch(score_e,e_batch)
#         x_n, x_e = ut.unbatch(x_n,n_batch), ut.unbatch(x_e,e_batch)
#         x_n0, x_e0 = ut.unbatch(x_n0,n_batch), ut.unbatch(x_e0,e_batch)
#         ngraph_list, egraph_list, par_list = [], [], []
#         x_n0_list, x_e0_list = [],[]
#         corr = []

#         for i,ei in enumerate(datas):
#             par_i = adj2par1(ei, num_nodes[i], num_edges[i])
# #             print(par_i.shape)
#             x_ni = score_n[i]
#             x_ei = score_e[i]
#             signal_ni = x_n[i]# * x_ni.view(-1,1)
#             signal_ei = x_e[i]# * x_ei.view(-1,1)
#             x_n0_i, x_e0_i = x_n0[i], x_e0[i]
            
#             x_e2ni = torch.sparse.mm(par_i.abs(), x_ei.view(-1,1)).view(-1)
#             corr.append(torch.corrcoef(torch.cat([x_ni.view(1,-1),x_e2ni.view(1,-1)],dim=0))[0,1])
#             x_ni = x_ni + 0.1*x_e2ni
            
#             _, perm_ni = x_ni.sort(dim=-1, descending=True)
#             k_ni = (float(self.ratio) * num_nodes[i].to(x_ni.dtype)).ceil().to(torch.long)
#             perm_ni = perm_ni[torch.arange(k_ni, device=x_ni.device)]
#             mask_ni = torch.zeros(num_nodes[i]).to(x_ni.device)
            
#             mask_ni[perm_ni] = 1
#             mask_ei = torch.sparse.mm(par_i.abs().transpose(0,1),
#                                       (1-mask_ni.view(1,-1)).T).T.view(-1)
            
#             par_i = par_i.to_dense()
#             par_i = par_i[mask_ni>0]
#             par_i = par_i.T[mask_ei==0].T
#             temp = par_i.to_sparse()
#             L0_i = torch.sparse.mm(temp, par_i.T)
#             L1_i = torch.sparse.mm(temp.transpose(0,1), par_i)

#             edge_index_t, edge_weight_t = dense_to_sparse(L0_i)
#             edge_index_s, edge_weight_s = dense_to_sparse(L1_i)
            
#             # scipy sparse eigenvalue decomposition
#             sci_sparse_t = ut.to_scipy_sparse_matrix(edge_index_t, edge_weight_t, num_nodes=L0_i.shape[0])
# #             sci_sparse_s = ut.to_scipy_sparse_matrix(edge_index_s, edge_weight_s, num_nodes=L1_i.shape[0])
#             max_lambda = eigsh(sci_sparse_t, k=1, which='LM', return_eigenvectors=False)[0]
#             edge_weight_t = 2*edge_weight_t / max_lambda
#             edge_weight_s = 2*edge_weight_s / max_lambda
            
#             ngraph_list.append(Data(x=signal_ni[mask_ni>0], edge_index=edge_index_t, edge_weight=edge_weight_t, num_nodeas=L0_i.shape[0]))
#             egraph_list.append(Data(x=signal_ei[mask_ei==0], edge_index=edge_index_s, edge_weight=edge_weight_s, num_nodeas=L1_i.shape[0]))
#             # print(par_i.shape)
#             par_list.append(Data(x=torch.ones(L0_i.shape[0],1), edge_index=par2adj(par_i), num_nodeas=L0_i.shape[0]))
#             x_n0_list.append(x_n0_i[mask_ni>0])
#             x_e0_list.append(x_e0_i[mask_ei==0])
            
                
#         new_ndata = Batch.from_data_list(ngraph_list)
#         new_edata = Batch.from_data_list(egraph_list)
#         new_par = Batch.from_data_list(par_list)
#         x_n0 = torch.cat(x_n0_list, dim=0)
#         x_e0 = torch.cat(x_e0_list, dim=0)
#         corr = torch.tensor(corr)
#         # e_batch = torch.cat([torch.ones(g.x.shape[0], device=g.x.device) for idx,g in enumerate(egraph_list)], dim=0).to(torch.long)
#         # print(e_batch.shape, new_edata.x.shape)

#         return [new_ndata.x, new_ndata.edge_index, new_ndata.edge_weight, 
#                 new_ndata.batch, new_edata.x, new_edata.edge_index, new_edata.edge_weight, 
#                 new_edata.batch, new_par.edge_index, x_n0, x_e0, corr.mean()]

# ###############################################################################
# ##################### Only Edge Convolution & Pooling #########################
# ###############################################################################
            
# class HL_HGCNN(torch.nn.Module):
#     def __init__(self, channels=2, filters=[64], mlp_channels=[], K=2, node_num=1024, 
#                  edge_num=3906, num_classes=10, dropout_ratio=0.0, dropout_ratio_mlp=0.0, alpha=0.5, beta=1.0):
#         super(HL_HGCNN, self).__init__()
#         self.Node_channels = channels * filters
#         self.Edge_channels = channels * filters
#         self.mlp_channels = mlp_channels
#         self.node_num = node_num
#         self.edge_num = edge_num
#         self.initial_channel = 64
#         self.alpha = alpha
#         self.beta = beta
#         # self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_slope)
#         # self.relu = nn.ReLU()
        
#         gcn_insize = 9
#         layers = [(HodgeLaguerreConv(gcn_insize, self.initial_channel, K=3),
#                    'x, edge_index, edge_weight -> x'),
#                   gnn.BatchNorm(self.initial_channel),
#                   nn.ReLU()]
#         fc = gnn.Sequential('x, edge_index, edge_weight', layers)

#         setattr(self, 'HL_EC', fc)
#         gcn_insize = self.initial_channel
            
#         for i, gcn_outsize in enumerate(self.Node_channels):
#             layers = [(HodgeLaguerreResConv(gcn_insize, gcn_outsize, K=K, beta=self.beta),
#                        'x, edge_index, edge_weight -> x'),
#                       gnn.BatchNorm(gcn_outsize),
#                       nn.ReLU(),
#                       (lambda x1, x2: (1-alpha)*x1+alpha*x2, 'x, x0 -> x'),
#                        (Dropout(p=dropout_ratio), 'x -> x'),
#                       ]
            
#             fc = gnn.Sequential('x, edge_index, edge_weight, x0', layers)
#             setattr(self, 'HL_EC%d' % i, fc)
#             gcn_insize = gcn_outsize
            
#             if i < len(self.Node_channels)-1:
#                 layers = [(HodgeLaguerreConv(gcn_insize*2, gcn_insize, K=1),
#                            'x, edge_index, edge_weight -> x'),
#                           gnn.BatchNorm(self.initial_channel),
#                           nn.ReLU()]
#                 fc = gnn.Sequential('x, edge_index, edge_weight', layers)
#                 setattr(self, 'int_e2n%d' % i, fc)
                
        
#         gcn_insize = 5
#         layers = [(HodgeLaguerreConv(gcn_insize, self.initial_channel, K=3),
#                    'x, edge_index, edge_weight -> x'),
#                   gnn.BatchNorm(self.initial_channel),
#                   nn.ReLU()]
#         fc = gnn.Sequential('x, edge_index, edge_weight', layers)

#         setattr(self, 'HL_NC', fc)
#         gcn_insize = self.initial_channel
        
#         for i, gcn_outsize in enumerate(self.Node_channels):
#             layers = [(HodgeLaguerreResConv(gcn_insize, gcn_outsize, K=K, beta=self.beta),
#                        'x, edge_index, edge_weight -> x'),
#                       gnn.BatchNorm(gcn_outsize),
#                       nn.ReLU(),
#                       (lambda x1, x2: (1-alpha)*x1+alpha*x2, 'x, x0 -> x'),
#                        (Dropout(p=dropout_ratio), 'x -> x'),
#                       ]
            
#             fc = gnn.Sequential('x, edge_index, edge_weight, x0', layers)
#             setattr(self, 'HL_NC%d' % i, fc)
#             gcn_insize = gcn_outsize
            
#             if i < len(self.Node_channels)-1:
#                 layers = [(HodgeLaguerreConv(gcn_insize*2, gcn_insize, K=1),
#                             'x, edge_index, edge_weight -> x'),
#                           gnn.BatchNorm(self.initial_channel),
#                           nn.ReLU()]
#                 fc = gnn.Sequential('x, edge_index, edge_weight', layers)
#                 setattr(self, 'int_n2e%d' % i, fc)
            
        
#         mlp_insize = self.Node_channels[-1] * 2 #sum(Node_channels)+ sum(Edge_channels)#[-1]
#         for i, mlp_outsize in enumerate(mlp_channels):
#             fc = nn.Sequential(
#                 Linear(mlp_insize, mlp_outsize),
#                 nn.BatchNorm1d(mlp_outsize),
#                 nn.ReLU(),
#                 nn.Dropout(dropout_ratio_mlp),
#                 )
#             setattr(self, 'mlp%d' % i, fc)
#             mlp_insize = mlp_outsize

#         self.out = Linear(mlp_insize, num_classes)


#     def forward(self, data):

#         # 1. Obtain node embeddings
#         # data_t = Data(x=data.x_t, edge_index=data.edge_index_t, edge_weight=data.edge_weight_t)
#         # data = Data(x=data.x_s, edge_index=data.edge_index_s, edge_weight=data.edge_weight_s)
        
#         batch = 128#int(data.x_t.shape[0] / self.node_num)
#         n_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(data.num_node1)], dim=-1)
#         n_batch = n_batch.to(device)
        
#         s_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(data.num_edge1)], dim=-1)
#         s_batch = s_batch.to(device)
#         x_s, edge_index_s, edge_weight_s = data.x_s, data.edge_index_s, data.edge_weight_s
#         x_t, edge_index_t, edge_weight_t = data.x_t, data.edge_index_t, data.edge_weight_t
#         edge_index = data.edge_index
        
#         x_s = self.HL_EC(x_s, edge_index_s, edge_weight_s)
#         x_t = self.HL_NC(x_t, edge_index_t, edge_weight_t)
#         x_s0, x_t0 = x_s, x_t
#         corr = 0

#         for i, _ in enumerate(self.Node_channels):
                
#             fc = getattr(self, 'HL_NC%d' % i)
#             x_t = fc(x_t, edge_index_t, edge_weight_t, x_t0)
            
#             fc = getattr(self, 'HL_EC%d' % i)
#             x_s = fc(x_s, edge_index_s, edge_weight_s, x_s0)
     
#             if i < len(self.Node_channels)-1:
#                 par_i = adj2par1(data.edge_index, x_t.shape[0], x_s.shape[0])
#                 D = degree(data.edge_index.view(-1))
#                 temp_xt = (1/D).view(-1,1)*torch.sparse.mm(par_i.abs(), x_s)
#                 temp_xs = torch.sparse.mm(par_i.abs().transpose(0,1), x_t)/2
#                 x_t = torch.cat([x_t, temp_xt], dim=-1)
#                 x_s = torch.cat([x_s, temp_xs], dim=-1)
              
#                 fc = getattr(self, 'int_e2n%d' % i)
#                 x_t = fc(x_t, edge_index_t, edge_weight_t)
#                 fc = getattr(self, 'int_n2e%d' % i)
#                 x_s = fc(x_s, edge_index_s, edge_weight_s)

#         # 2. Readout layer
#         x = torch.cat( (global_mean_pool(x_s, s_batch),global_mean_pool(x_t, n_batch)), -1)
        
#         # 3. Apply a final classifier
#         for i, _ in enumerate(self.mlp_channels):
#             fc = getattr(self, 'mlp%d' % i)
#             x = fc(x)

#         return self.out(x), corr 


# ###############################################################################
# ################################# Train & Test ################################
# ###############################################################################

# def train(loader):
#     model.train()
#     total_loss = 0
#     for data in loader:  # Iterate in batches over the training dataset.
#         data = data.to(device)
#         out, corr = model(data)
#         loss = criterion(out, data.y) + 5*corr#.view(-1,1))  # Compute the loss.
#         # print(criterion(out, data.y), 0.1*corr)
#         # loss = weighted_mse_loss(out, data.y.view(-1,1))
        
#         total_loss += loss*data.num_graphs
#         loss.backward()  # Derive gradients.
#         optimizer.step()  # Update parameters based on gradients.
#         optimizer.zero_grad()  # Clear gradients.
#     return total_loss/len(loader.dataset)


# def test(loader):
#       model.eval()
#       # y_pred = torch.zeros(len(loader.dataset))
#       # y = torch.zeros(len(loader.dataset))
#     #  y_pred, y = [], []
#       total_loss, acc = 0, 0
#       corr_loss = 0
     
#       for data in loader:  # Iterate in batches over the training/test dataset. 
#           data = data.to(device)
#           with torch.no_grad():
#             out, corr = model(data) 
#           loss = criterion(out, data.y)#.view(-1,1))  # Compute the loss.
#           corr_loss += corr * data.num_graphs
#           acc += torch.count_nonzero(torch.argmax(out,dim=1) == data.y)
#           total_loss += loss * data.num_graphs

#       return total_loss/len(loader.dataset), corr_loss/len(loader.dataset), acc/len(loader.dataset)


# if __name__ == '__main__':
#     # test degree based spatial pooling
#     if torch.cuda.is_available():
#         device = 'cuda:0'
#     else:
#         device = 'cpu'
#     print(device)
#     start = time.time()
    
#     for fold in [0]:#range(5):
#         print('Fold {} begin'.format(fold))
        
#         model = HL_HGCNN(channels=10, filters=[64], mlp_channels=[], K=2, 
#                           dropout_ratio=0.3, dropout_ratio_mlp=0.0).to(device)  # 8conv
#         save_path = './weights/HGCNN_nopool_res_int_CIFAR10_BM_5conv64_mlp0_FOLD{}.pt'.format(fold)
        
#         # model = HL_HGCNN(channels=8, filters=[64], mlp_channels=[128], K=2, 
#         #                   dropout_ratio=0.3, dropout_ratio_mlp=0.0).to(device)  # 4conv
#         # save_path = './weights/HGCNN_nopool_res_int_CIFAR10_BM_4conv64_mlp0_FOLD{}.pt'.format(fold)
        
#         # model = HL_HGCNN(channels=6, filters=[64], mlp_channels=[128], K=2, 
#         #                   dropout_ratio=0.3, dropout_ratio_mlp=0.0).to(device)  # 4conv
#         # save_path = './weights/HGCNN_nopool_res_int_CIFAR10_BM_3conv64_mlp0_FOLD{}.pt'.format(fold)
        
#         # model = HL_HGCNN(channels=[64, 64], mlp_channels=[], K=2, 
#         #                  dropout_ratio=0.0, dropout_ratio_mlp=0.0).to(device)  # 2conv mlp0 acc=0.6687
#         # save_path = './weights/HGCNN_nopool_res_int_CIFAR10_BM_2conv64_mlp0_FOLD{}.pt'.format(fold)
        
#         # model.load_state_dict(torch.load(save_path))
#         optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
#         scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
#         # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-3)
#         criterion = torch.nn.CrossEntropyLoss()
#         batch_size = 512
        
#         dataset = GNNBenchmarkDataset(root='CIFAR10_bm',name='CIFAR10',split='train')
#         trainset = CIFAR_HG_BM_par1(root=osp.join('CIFAR10_bm','train'),dataset=dataset)
        
#         dataset = GNNBenchmarkDataset(root='CIFAR10_bm',name='CIFAR10',split='val')
#         validset = CIFAR_HG_BM_par1(root=osp.join('CIFAR10_bm','val'), dataset=dataset)
        
#         dataset = GNNBenchmarkDataset(root='CIFAR10_bm',name='CIFAR10',split='test')
#         testset = CIFAR_HG_BM_par1(root=osp.join('CIFAR10_bm','test'), dataset=dataset)

#         train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
#         valid_loader = DataLoader(validset, batch_size=batch_size, num_workers=4)
#         test_loader = DataLoader(testset, batch_size=batch_size, num_workers=4)
        
#         best_loss, best_corr, best_acc = test(test_loader)
#         print('==================================================================================')
#         print(f'Test Loss: {best_loss:.4f}, Test Acc: {best_acc:.4f}')
#         print('==================================================================================')
        
#         for epoch in range(1, 600):
#             total_loss = train(train_loader)
#             # train_corr, _, _ = test(train_loader)
#             valid_loss, valid_corr, valid_acc = test(valid_loader)
#             scheduler.step()
#             # test_corr, test_loss, test_rmse = test(test_loader)
    
#             elapsed = (time.time()-start) / 60
#             print(f'Epoch: {epoch:03d}, time: {elapsed:.2f} Train Loss: {total_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Corr: {valid_corr:.4f}, Valid Acc: {valid_acc:.4f}')
#             if valid_acc>0.6 and valid_acc>best_acc:
#                 best_acc = valid_acc
#                 torch.save(model.state_dict(), save_path)
#                 print('Model saved! \n')   
#                 best_loss1, best_corr1, best_acc1 = test(test_loader)
#                 print('==================================================================================')
#                 print(f'Test Loss: {best_loss1:.4f}, Test Corr: {best_corr1:.4f}, Test Acc: {best_acc1:.4f}')
#                 print('==================================================================================')
                
#         model.load_state_dict(torch.load(save_path))
#         best_loss, best_corr, best_acc = test(test_loader)
#         print('==================================================================================')
#         print(f'Test Loss: {best_loss:.4f}, Test Acc: {best_acc:.4f}')
#         print('==================================================================================')
        
#     # export CUDA_VISIBLE_DEVICES=3
    
