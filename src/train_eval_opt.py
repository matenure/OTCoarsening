import time

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DenseDataLoader as DenseLoader
from torch.utils.data import DataLoader
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import torch.utils.data as data

class MyDataset(data.Dataset):
    def __init__(self,X, Y,transform=None):
        self.X=X
        self.Y=Y
        self.num_classes = int(max(Y).item())+1
        self.transform=transform

    def __getitem__(self,index):
        x=self.X[index]
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.Y)

class MLP(torch.nn.Module): #2 layer GCN block
    def __init__(self, in_channels, hidden_channels, out_channels, drop=0.0):
        super(MLP, self).__init__()

        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.dp = drop

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dp, training=self.training)
        x = self.lin2(x)
        return x

def getMiddleRes(dataset, model,batch_size,eps,opt_iters):
    model.eval()

    loader = DenseLoader(dataset, batch_size=batch_size, shuffle=False)
    Xs = []
    Ys = []
    for data in loader:
        data = data.to(device)
        Ys.append(data.y)
        with torch.no_grad():
            xs, new_adjs, Ss, opt_loss = model(data, epsilon=eps, opt_epochs=opt_iters)
            Xs.append(model.jump(xs))
            # Xs.append(xs[0])
    Xs = torch.cat(Xs, 0)
    Ys = torch.cat(Ys, 0).float()
    myData = MyDataset(Xs, Ys)
    return myData

def cross_validation_with_val_set_opt(myData,
                                  model,
                                  folds,
                                  epochs,
                                  batch_size,
                                  lr,
                                  weight_decay,
                                  logger=None):


    val_losses, accs, durations = [], [], []
    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*k_fold(myData, folds))):

        train_dataset = data.Subset(myData, train_idx)
        test_dataset = data.Subset(myData, test_idx)
        val_dataset = data.Subset(myData, val_idx)


        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        for epoch in range(1, epochs + 1):
            train_loss = train_MLP(model, optimizer, train_loader)
            val_losses.append(eval_loss(model, val_loader))
            accs.append(eval_acc(model, test_loader))
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_losses[-1],
                'test_acc': accs[-1],
            }
            # print(eval_info)

            if logger is not None:
                logger(eval_info)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
    loss, argmin = loss.min(dim=1)
    acc = acc[torch.arange(folds, dtype=torch.long), argmin]

    loss_mean = loss.mean().item()
    acc_mean = acc.mean().item()
    acc_std = acc.std().item()
    duration_mean = duration.mean().item()
    print('Val Loss: {:.4f}, Test Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(loss_mean, acc_mean, acc_std, duration_mean))

    return loss_mean, acc_mean, acc_std


def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.Y):
        test_indices.append(torch.from_numpy(idx))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices



def train_MLP(model, optimizer, loader):
    model.train()

    total_loss = 0
    for X,Y in loader:
        optimizer.zero_grad()
        X = X.to(device)
        Y = Y.to(device)
        out = model(X)
        loss = F.cross_entropy(out, Y.long().view(-1))
        loss.backward()
        total_loss += loss.item() * X.size(0)
        optimizer.step()
    return total_loss / len(loader.dataset)



def eval_acc(model, loader):
    model.eval()

    correct = 0
    for X,Y in loader:
        X = X.to(device)
        Y = Y.long().to(device)
        with torch.no_grad():
            pred = model(X).max(1)[1]
        correct += pred.eq(Y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader):
    model.eval()

    loss = 0
    for X,Y in loader:
        X = X.to(device)
        Y = Y.to(device)
        with torch.no_grad():
            out = model(X)
        loss += F.cross_entropy(out, Y.long().view(-1), reduction='sum').item()
    return loss / len(loader.dataset)
