##Method2: Directly compare original features and coarsed features (with the same dimension,
# GNN(fe, hidden) - Coarsen - GNN(hidden, fe) ).

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import DenseSAGEConv, DenseGCNConv, JumpingKnowledge
from Sinkhorn import sinkhorn_loss_default

class GNNBlock(torch.nn.Module): #2 layer GCN block
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNBlock, self).__init__()

        self.conv1 = DenseGCNConv(in_channels, hidden_channels)
        self.conv2 = DenseGCNConv(hidden_channels, out_channels)

        self.lin = torch.nn.Linear(hidden_channels + out_channels,
                                   out_channels)
        # self.lin1 = torch.nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, adj, mask=None, add_loop=True):
        x1 = F.relu(self.conv1(x, adj, mask, add_loop))
        x2 = F.relu(self.conv2(x1, adj, mask, add_loop))
        return self.lin(torch.cat([x1, x2], dim=-1))
        # return self.lin1(x1, dim=-1)


class GNN_ae(torch.nn.Module): # a baseline for no-coarsening unsupervised learning, essentially an auto-encoder
    def __init__(self, dataset, hidden, num_layers=2, ratio=0.5):
        super(GNN_ae, self).__init__()

        self.embed_block1 = DenseGCNConv(dataset.num_features, hidden)

        self.embed_block2 = DenseGCNConv(hidden, dataset.num_features)
        # self.embed_block2 = GNNBlock(hidden, hidden, dataset.num_features)

        self.num_layers = num_layers

        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.embed_block1.reset_parameters()
        self.embed_block2.reset_parameters()

        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data, epsilon=0.01, opt_epochs=100):
        x, adj, mask = data.x, data.adj, data.mask
        batch_num_nodes = data.mask.sum(-1)

        new_adjs = [adj]
        Ss = []

        x1 = F.relu(self.embed_block1(x, adj, mask, add_loop=True))
        xs = [x1.mean(dim=1)]
        new_adj = adj
        coarse_x = x1
        # coarse_x, new_adj, S = self.coarse_block1(x1, adj, batch_num_nodes)
        # new_adjs.append(new_adj)
        # Ss.append(S)

        x2 = self.embed_block2(coarse_x, new_adj, mask, add_loop=True)#should not add ReLu, otherwise x2 could be all zero.
        # xs.append(x2.mean(dim=1))
        opt_loss = 0.0
        for i in range(len(x)):
            opt_loss += F.mse_loss(x2[i], x[i])

        return xs, new_adjs, Ss, opt_loss

    def get_nonzero_rows(self, M):# M is a matrix
        # row_ind = M.sum(-1).nonzero().squeeze() #nonzero has bugs in Pytorch 1.2.0.........
        #So we use other methods to take place of it
        MM, MM_ind = torch.abs(M.sum(-1)).sort()
        N = (torch.abs(M.sum(-1))>0).sum()
        return M[MM_ind[:N]]

    def predict(self, xs):
        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
