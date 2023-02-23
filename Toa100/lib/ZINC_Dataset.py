#!/usr/bin/env python
# coding: utf-8
import os.path

# In[11]:


# import torch
# import torch.nn.functional as F
# from torch.utils.data import ConcatDataset
# # import torch.utils.data as tud

# from torch_geometric.data import Data, Dataset, InMemoryDataset, download_url
# from torch_geometric.datasets import GNNBenchmarkDataset, ZINC
# from torch_geometric.loader import DataLoader

# from torch_geometric.utils import add_self_loops, degree, to_undirected, dense_to_sparse, coalesce

# import os.path as osp
# import numpy as np
# from scipy.io import loadmat

# # from lib.Hodge_Cheb_Conv import *


# In[83]:


import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
# import torch.utils.data as tud

from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.datasets import GNNBenchmarkDataset, ZINC
from torch_geometric.loader import DataLoader

from torch_geometric.utils import add_self_loops, degree, to_undirected, dense_to_sparse, coalesce

import os.path as osp
# import numpy as np
# from scipy.io import loadmat

# from lib.Hodge_Cheb_Conv import *


# In[84]:


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


# In[85]:


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
        return ['ZINC_BM_'+str(fileidx+1)+'.pt' for fileidx in range(len(self.dataset))]

    @property
    def processed_paths(self):
        return self.processed_dir

    def len(self):
        return len(self.dataset)
#         return len(self.processed_file_names)

    def get(self,idx):
        data_zip = torch.load(osp.join(self.processed_dir, 'ZINC_BM_'+str(idx+1)+'.pt'))
        data = data_zip['graph']
        data.num_node1 = data.x_t.shape[0]
        data.num_edge1 = data.x_s.shape[0]
        data.num_nodes = data.x_t.shape[0]
        return data
            
    def process(self):
        if not osp.exists(osp.join(self.processed_dir, 'ZINC_BM_1.pt')):
            min_attr = 100
            max_attr = 0
            min_x_t = 100
            max_x_t = 0
            min_Y = 100
            max_Y = 0
            print('preprocssing start, convert to pairdata graph')
            i = 0
            loader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=1)
            for data in loader:

                if min_Y > min(data.y):
                    min_Y = min(data.y)
                if max_Y < max(data.y):
                    max_Y = max(data.y)

                edge_index,edge_attr = to_undirected(data.edge_index, data.edge_attr,reduce='mean')
                idx = edge_index[0]<edge_index[1]
                if min_attr > min(data.edge_attr):
                    min_attr = min(data.edge_attr)
                if max_attr < max(data.edge_attr):
                    max_attr = max(data.edge_attr)

                edge_index,edge_attr = edge_index[:,idx], edge_attr[idx]

        #             self_edge_index = torch.cat([torch.arange(data.x.shape[0]), torch.arange(data.x.shape[0])],
        #                                         dim=-1).view(2,-1)

        #             edge_index = torch.cat([edge_index,self_edge_index],dim=-1)
        #             edge_attr = torch.cat([edge_attr, torch.ones(data.x.shape[0])*7],dim=-1).to(dtype=torch.long)

                par1 = adj2par1(edge_index, data.x.shape[0], edge_index.shape[1]).to_dense()
                L0 = torch.matmul(par1, par1.T)
                lambda0, _ = torch.linalg.eigh(L0)
                L1 = torch.matmul(par1.T, par1)
                edge_index_t, edge_weight_t = dense_to_sparse(2*L0/lambda0.max())
                edge_index_s, edge_weight_s = dense_to_sparse(2*L1/lambda0.max())

                x_s = F.one_hot(edge_attr-1,num_classes=3) # the min value of edge_attr=1, one_hot start from value 0
                x_t = F.one_hot(data.x.squeeze(-1),num_classes=21)

                if min_x_t > min(data.x.squeeze(-1)):
                    min_x_t = min(data.x.squeeze(-1))
                if max_x_t < max(data.x.squeeze(-1)):
                    max_x_t = max(data.x.squeeze(-1))

                graph = PairData(x_s=x_s, edge_index_s=edge_index_s, edge_weight_s=edge_weight_s,
                                 x_t=x_t, edge_index_t=edge_index_t, edge_weight_t=edge_weight_t,
                                 y = data.y)
                graph.edge_index=edge_index
                data_zip = {'graph':graph}
                torch.save(data_zip, osp.join(self.processed_dir, 'ZINC_BM_'+str(i+1)+'.pt'))
                i += 1

            print(f'max_attr is {max_attr}')
            print(f'min_attr is {min_attr}')
            print(f'max_x_t is {max_x_t}')
            print(f'min_x_t is {min_x_t}')
            print(f'max_Y is {max_Y}')
            print(f'min_Y is {min_Y}')
            print(f'total gragh number is {i}')


# In[88]:


# root_zinc = '/home/qiufeng/GNN/ZINC_experiment/'
# dataset_raw_train = ZINC(root=root_zinc+'train_data', subset=True,split='train')
# dataset_raw_val = ZINC(root=root_zinc+'val_data', subset=True,split='val')
# dataset_raw_test = ZINC(root=root_zinc+'test_data', subset=True,split='test')

# dataset_raw = ConcatDataset([dataset_raw_train, dataset_raw_val, dataset_raw_test])



# In[89]:


# dataset_processed_train = ZINC_HG_BM_par1(root_zinc+'train_data', dataset_raw_train)
# dataset_processed_val = ZINC_HG_BM_par1(root_zinc+'val_data', dataset_raw_val)
# dataset_processed_test = ZINC_HG_BM_par1(root_zinc+'test_data', dataset_raw_test)

# dataset_processed = ZINC_HG_BM_par1(root_zinc+'total', dataset_raw)




# In[90]:


# data1 = dataset_raw_train[0]
# data1


# In[91]:


# data2 = dataset_processed_train[0]
# data2


# In[ ]:





# In[ ]:





# In[ ]:




