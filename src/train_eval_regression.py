import time

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cross_validation_with_val_set_regression(taskid, dataset,
                                  model,
                                  folds,
                                  epochs,
                                  batch_size,
                                  lr,
                                  lr_decay_factor,
                                  lr_decay_step_size,
                                  weight_decay,
                                  logger=None):
    val_losses_mse, test_losses_mae, accs, durations = [], [], [], []
    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*k_fold(dataset, folds))):

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]

        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DenseLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        for epoch in range(1, epochs + 1):
            train_loss = train_regression(model, optimizer, train_loader, taskid)
            tmp_mseloss, _ = eval_loss(model, val_loader, taskid)
            val_losses_mse.append(tmp_mseloss)
            test_losses_mae.append(eval_loss(model, test_loader, taskid)[-1])
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_losses_mse[-1],
                'test_mae_error': test_losses_mae[-1]
                # 'test_acc': accs[-1],
            }
            # print(eval_info)

            if logger is not None:
                logger(eval_info)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    mse_loss, test_mae_error, duration = tensor(val_losses_mse), tensor(test_losses_mae), tensor(durations)
    loss, test_mae_error = mse_loss.view(folds, epochs), test_mae_error.view(folds, epochs)
    loss, argmin = loss.min(dim=1)
    test_mae_error = test_mae_error[torch.arange(folds, dtype=torch.long), argmin]

    loss_mean = loss.mean().item()
    # loss_std = loss.std().item()
    mae_error_mean = test_mae_error.mean().item()
    mae_error_std = test_mae_error.std().item()
    duration_mean = duration.mean().item()
    print('Val MSE Loss: {:.4f} , Test MAE Error: {:.4f} ± {:.4f}, Duration: {:.3f}'.
          format(loss_mean,  mae_error_mean, mae_error_std, duration_mean))

    return loss_mean, mae_error_mean, mae_error_std


def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), torch.zeros(len(dataset))):
        test_indices.append(torch.from_numpy(idx))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train_regression(model, optimizer, loader, taskid):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = F.mse_loss(out[:,taskid].squeeze().view(-1), data.y.squeeze()[:,taskid].view(-1))
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)




def eval_loss(model, loader, taskid):
    model.eval()

    loss_mse = 0
    loss_mae = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        loss_mse += F.mse_loss(out[:,taskid].squeeze().view(-1), data.y.squeeze()[:,taskid].view(-1), reduction='sum').item()
        loss_mae = F.l1_loss(out[:,taskid].squeeze().view(-1), data.y.squeeze()[:,taskid].view(-1), reduction='sum').item()
    return loss_mse / len(loader.dataset), loss_mae / len(loader.dataset)
