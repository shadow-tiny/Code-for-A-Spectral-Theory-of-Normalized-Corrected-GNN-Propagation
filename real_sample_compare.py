import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as utils
from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.nn.conv import MessagePassing, GCNConv as PyGGCNConv
from tqdm import tqdm
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T
import os

# Required to avoid type 3 fonts in figure pdfs.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

torch.set_printoptions(precision=2,sci_mode=False, linewidth=200)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)
loss_fn = nn.CrossEntropyLoss()


class PairNorm(nn.Module):
    def __init__(self, mode='PN-SI', scale=1.0):
        super().__init__()
        self.mode = mode
        self.scale = scale

    def forward(self, x):
        if self.mode == 'None':
            return x

        col_mean = x.mean(dim=0)
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            return self.scale * x / rownorm_mean
        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            return self.scale * x / rownorm_individual
        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            return self.scale * x / rownorm_individual - col_mean
        return x

def accuracy(output, labels):
    preds = output.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)

def train_model(model, optimizer, data, epochs=400):
    model.to(device)
    model.train()
    data = data.to(device)
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

def train_minibatch(model, optimizer, loader, epochs=10):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        for data in loader:
            data = data.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            batch_size = data.batch_size
            loss = loss_fn(out[:batch_size], data.y[:batch_size])
            loss.backward()
            optimizer.step()

def test_minibatch(model, loader):
    model.to(device)
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device, non_blocking=True)
            out = model(data.x, data.edge_index)
            batch_size = data.batch_size
            
            acc = accuracy(out[:batch_size], data.y[:batch_size])
            total_correct += acc * batch_size
            total_samples += batch_size
            
    return total_correct / total_samples

def test_model_cpu(model, data):
    model.cpu()
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    acc = accuracy(out[data.test_mask], data.y[data.test_mask])
    return acc

def test_model(model, data):
    model.to(device)
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    acc = accuracy(out[data.test_mask], data.y[data.test_mask])
    return acc

def extract_balanced_two_class_subset(data):
    y = data.y
    valid_mask = y >= 0
    y_valid = y[valid_mask]
    
    classes, counts = torch.unique(y_valid, return_counts=True)
    sorted_indices = torch.argsort(counts, descending=True)
    class1 = classes[sorted_indices[0]]
    class2 = classes[sorted_indices[1]]
    
    target_size = counts[sorted_indices[1]]
    
    idx1 = torch.where(y == class1)[0]
    idx2 = torch.where(y == class2)[0]
    
    perm1 = torch.randperm(idx1.size(0), device=idx1.device)[:target_size]
    perm2 = torch.randperm(idx2.size(0), device=idx2.device)[:target_size]
    
    idx1_sampled = idx1[perm1]
    idx2_sampled = idx2[perm2]
    
    subset_idx = torch.cat([idx1_sampled, idx2_sampled])
    subset_idx = torch.sort(subset_idx)[0]
    
    new_y_relabeled = torch.zeros_like(data.y)
    new_y_relabeled[idx1_sampled] = 0
    new_y_relabeled[idx2_sampled] = 1
    
    num_subset_nodes = subset_idx.size(0)
    perm = torch.randperm(num_subset_nodes, device=subset_idx.device)
    train_end = int(0.6 * num_subset_nodes)
    val_end = int(0.8 * num_subset_nodes)
    
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.x.device)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.x.device)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.x.device)
    
    train_mask[subset_idx[perm[:train_end]]] = True
    val_mask[subset_idx[perm[train_end:val_end]]] = True
    test_mask[subset_idx[perm[val_end:]]] = True
    
    new_data = data.clone()
    new_data.y = new_y_relabeled
    new_data.train_mask = train_mask
    new_data.val_mask = val_mask
    new_data.test_mask = test_mask
    
    return new_data

class CorrectedConv1(MessagePassing):
    def __init__(self, in_channels, out_channels, num_convolutions):
        super().__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.num_convolutions = num_convolutions
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        x = self.lin(x)
        edge_index, _ = utils.add_remaining_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = utils.degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        deg_sqrt = deg.pow(0.5).unsqueeze(1)
        total_edges = edge_index.size(1)
        
        for _ in range(self.num_convolutions):
            rank1_comp = deg_sqrt.T@x
            rank1_comp = deg_sqrt@rank1_comp / total_edges
            x = self.propagate(edge_index, x=x, norm=norm)
            x -= rank1_comp
        
        x += self.bias
        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class GCNDropEdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_convolutions, drop_p=0.5):
        super().__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.num_convolutions = num_convolutions
        self.drop_p = drop_p
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        edge_index, _ = utils.add_remaining_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        
        for _ in range(self.num_convolutions):
            if self.training:
                mask = torch.rand(edge_index.size(1), device=edge_index.device) > self.drop_p
                edge_index_drop = edge_index[:, mask]
            else:
                edge_index_drop = edge_index

            row, col = edge_index_drop
            deg = utils.degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

            x = self.propagate(edge_index_drop, x=x, norm=norm)
        
        x += self.bias
        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class ReverseGNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_convolutions):
        super().__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.num_convolutions = num_convolutions
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        edge_index, _ = utils.add_remaining_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        row, col = edge_index
        deg = utils.degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        for _ in range(self.num_convolutions):
            res = x
            x = self.propagate(edge_index, x=x, norm=norm)
            x = x + res
        
        x += self.bias
        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class MbaGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_convolutions):
        super().__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.num_convolutions = num_convolutions
        self.norm = nn.LayerNorm(out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()
        self.norm.reset_parameters()

    def forward(self, x, edge_index):
        edge_index, _ = utils.add_remaining_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        row, col = edge_index
        deg = utils.degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm_weights = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        for _ in range(self.num_convolutions):
            res = x
            x = self.propagate(edge_index, x=x, norm=norm_weights)
            x = self.norm(x)
            x = F.relu(x) * res + x
        
        x += self.bias
        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class GCNIIConv(MessagePassing):
    def __init__(self, channels):
        super().__init__(aggr='add')
        self.lin1 = nn.Linear(channels, channels, bias=False)
        self.lin2 = nn.Linear(channels, channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, alpha, h0, beta):
        edge_index, _ = utils.add_remaining_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = utils.degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        support = (1 - beta) * (1 - alpha) * x + beta * self.lin1(x)
        initial = (1 - beta) * alpha * h0 + beta * self.lin2(h0)
        return self.propagate(edge_index, x=support, norm=norm) + initial

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class APPNPProp(MessagePassing):
    def __init__(self, num_convolutions, alpha=0.1):
        super().__init__(aggr='add')
        self.num_convolutions = num_convolutions
        self.alpha = alpha

    def forward(self, x, edge_index):
        edge_index, _ = utils.add_remaining_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = utils.degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        h = x
        h0 = x
        for _ in range(self.num_convolutions):
            h = self.propagate(edge_index, x=h, norm=norm)
            h = (1 - self.alpha) * h + self.alpha * h0
        return h

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class BaseGNNModel(torch.nn.Module):
    def __init__(self, conv_class, input_dim, hidden_dim, output_dim, n_layers, num_convolutions=1):
        super(BaseGNNModel, self).__init__()
        self.n_layers = n_layers
        self.relu = nn.ReLU()
        channels = [input_dim] + [hidden_dim]*(n_layers-1) + [output_dim]
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(conv_class(channels[i], channels[i+1], num_convolutions))
            if i < n_layers - 1:
                self.bns.append(nn.BatchNorm1d(channels[i+1]))

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            if i < self.n_layers - 1:
                x = self.bns[i](x)
                x = self.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x


class PairNormGCNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, pairnorm_mode='PN-SI'):
        super().__init__()
        self.n_layers = n_layers
        self.relu = nn.ReLU()
        channels = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(PyGGCNConv(channels[i], channels[i + 1]))
            if i < n_layers - 1:
                self.norms.append(PairNorm(mode=pairnorm_mode))

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            if i < self.n_layers - 1:
                x = self.norms[i](x)
                x = self.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x


class GCNIIModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, alpha=0.1, lamda=0.5):
        super().__init__()
        self.dropout = 0.5
        self.alpha = alpha
        self.lamda = lamda
        self.input_lin = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList([GCNIIConv(hidden_dim) for _ in range(max(n_layers - 1, 1))])
        self.output_lin = nn.Linear(hidden_dim, output_dim)
        self.reg_params = list(self.convs.parameters())
        self.non_reg_params = list(self.input_lin.parameters()) + list(self.output_lin.parameters())

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.input_lin(x))
        h0 = x
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            beta = math.log(self.lamda / (i + 1) + 1.0)
            x = F.relu(conv(x, edge_index, self.alpha, h0, beta))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.output_lin(x)


class APPNPModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, alpha=0.1):
        super().__init__()
        self.dropout = 0.5
        self.input_lin = nn.Linear(input_dim, hidden_dim)
        self.output_lin = nn.Linear(hidden_dim, output_dim)
        self.prop = APPNPProp(n_layers, alpha=alpha)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.input_lin(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.output_lin(x)
        return self.prop(x, edge_index)


def build_optimizer(model, lr=0.01, weight_decay=1e-4):
    if hasattr(model, 'reg_params') and hasattr(model, 'non_reg_params'):
        return torch.optim.Adam([
            {'params': model.reg_params, 'weight_decay': weight_decay},
            {'params': model.non_reg_params, 'weight_decay': 0.0},
        ], lr=lr)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def experiment(model, data, loader=None, test_loader=None):
    optimizer = build_optimizer(model, lr=0.01, weight_decay=1e-4)
    if loader:
        train_minibatch(model, optimizer, loader)
        if test_loader:
            test_acc = test_minibatch(model, test_loader)
        else:
            test_acc = test_model_cpu(model, data)
    else:
        train_model(model, optimizer, data)
        test_acc = test_model(model, data)
    return test_acc

def per_dataset(dataset, n_trials=10):
    num_features = dataset.num_features
    balanced_data = extract_balanced_two_class_subset(dataset._data)
    num_classes = 2

    ds_name = getattr(dataset, 'name', dataset.__class__.__name__)

    if not os.path.exists('result/real_sample_1'):
        os.makedirs('result/real_sample_1')
    result_path = f'result/real_sample_1/Ours_vs_Oversmoothing_Baselines_{ds_name}_data.npz'

    # Initialize empty lists that will be populated either from file or from scratch
    x_values = []
    acc_ours_values = []
    acc_drop_values = []
    acc_mba_values = []
    acc_rev_values = []
    acc_pair_values = []
    acc_gcnii_values = []
    acc_appnp_values = []
    completed_layers = set()

    if os.path.exists(result_path):
        print(f"Loading existing results from {result_path}")
        data_loaded = np.load(result_path)
        x_values = data_loaded['x'].tolist()
        acc_ours_values = data_loaded['ours'].tolist()
        acc_drop_values = data_loaded['dropedge'].tolist()
        acc_mba_values = data_loaded['mbagcn'].tolist()
        acc_rev_values = data_loaded['reversegnn'].tolist()
        if 'pairnorm' in data_loaded.files:
            acc_pair_values = data_loaded['pairnorm'].tolist()
        else:
            acc_pair_values = [float('nan')] * len(x_values)
        if 'gcnii' in data_loaded.files:
            acc_gcnii_values = data_loaded['gcnii'].tolist()
        else:
            acc_gcnii_values = [float('nan')] * len(x_values)
        if 'appnp' in data_loaded.files:
            acc_appnp_values = data_loaded['appnp'].tolist()
        else:
            acc_appnp_values = [float('nan')] * len(x_values)
        completed_layers = {
            layer for idx, layer in enumerate(x_values)
            if not (
                np.isnan(acc_pair_values[idx])
                or np.isnan(acc_gcnii_values[idx])
                or np.isnan(acc_appnp_values[idx])
            )
        }
        print(f"Already completed layers: {completed_layers}")

    large_datasets = ['ogbn-products', 'Reddit']
    use_loader = ds_name in large_datasets
    loader = None
    test_loader = None

    if use_loader:
        print(f"Using NeighborLoader for {ds_name} for extreme fast mini-batch training.")
        data = balanced_data
        
        loader = NeighborLoader(
            data,
            input_nodes=data.train_mask, 
            num_neighbors=[10, 5, 5], 
            batch_size=4096,       
            shuffle=True,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            transform=T.ToUndirected()
        )
        test_loader = NeighborLoader(
            data,
            input_nodes=data.test_mask, 
            num_neighbors=[10, 5, 5],
            batch_size=4096,
            shuffle=False,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            transform=T.ToUndirected()
        )
    else:
        data = balanced_data.to(device)

    pbar = tqdm(range(2, 33, 2), leave=False, desc='Layers')
    for k in pbar:
        if k in x_values:
            idx = x_values.index(k)
            acc_ours = acc_ours_values[idx]
            acc_drop = acc_drop_values[idx]
            acc_mba = acc_mba_values[idx]
            acc_rev = acc_rev_values[idx]
            acc_pair = acc_pair_values[idx]
            acc_gcnii = acc_gcnii_values[idx]
            acc_appnp = acc_appnp_values[idx]
            need_old = False
            need_pair = np.isnan(acc_pair)
            need_gcnii = np.isnan(acc_gcnii)
            need_appnp = np.isnan(acc_appnp)
            if need_pair:
                acc_pair = 0.0
            if need_gcnii:
                acc_gcnii = 0.0
            if need_appnp:
                acc_appnp = 0.0
            if not (need_old or need_pair or need_gcnii or need_appnp):
                continue
        else:
            idx = None
            acc_ours = 0.
            acc_drop = 0.
            acc_mba = 0.
            acc_rev = 0.
            acc_pair = 0.
            acc_gcnii = 0.
            acc_appnp = 0.
            need_old = True
            need_pair = True
            need_gcnii = True
            need_appnp = True

        for t in range(n_trials):
            print(f'Layers: {k}, Trial: {t}') 
            if need_old:
                model_ours = BaseGNNModel(CorrectedConv1, input_dim=num_features, hidden_dim=64, output_dim=num_classes, n_layers=k)
                model_drop = BaseGNNModel(GCNDropEdgeConv, input_dim=num_features, hidden_dim=64, output_dim=num_classes, n_layers=k)
                model_mba = BaseGNNModel(MbaGCNConv, input_dim=num_features, hidden_dim=64, output_dim=num_classes, n_layers=k)
                model_rev = BaseGNNModel(ReverseGNNConv, input_dim=num_features, hidden_dim=64, output_dim=num_classes, n_layers=k)
                acc_ours += experiment(model_ours, data, loader, test_loader)
                acc_drop += experiment(model_drop, data, loader, test_loader)
                acc_mba += experiment(model_mba, data, loader, test_loader)
                acc_rev += experiment(model_rev, data, loader, test_loader)
            if need_pair:
                model_pair = PairNormGCNModel(input_dim=num_features, hidden_dim=64, output_dim=num_classes, n_layers=k)
                acc_pair += experiment(model_pair, data, loader, test_loader)
            if need_gcnii:
                model_gcnii = GCNIIModel(input_dim=num_features, hidden_dim=64, output_dim=num_classes, n_layers=k)
                acc_gcnii += experiment(model_gcnii, data, loader, test_loader)
            if need_appnp:
                model_appnp = APPNPModel(input_dim=num_features, hidden_dim=64, output_dim=num_classes, n_layers=k)
                acc_appnp += experiment(model_appnp, data, loader, test_loader)

        if need_old:
            acc_ours /= n_trials
            acc_drop /= n_trials
            acc_mba /= n_trials
            acc_rev /= n_trials
        if need_pair:
            acc_pair /= n_trials
        if need_gcnii:
            acc_gcnii /= n_trials
        if need_appnp:
            acc_appnp /= n_trials

        if idx is None:
            x_values.append(k)
            acc_ours_values.append(acc_ours)
            acc_drop_values.append(acc_drop)
            acc_mba_values.append(acc_mba)
            acc_rev_values.append(acc_rev)
            acc_pair_values.append(acc_pair)
            acc_gcnii_values.append(acc_gcnii)
            acc_appnp_values.append(acc_appnp)
        else:
            acc_ours_values[idx] = acc_ours
            acc_drop_values[idx] = acc_drop
            acc_mba_values[idx] = acc_mba
            acc_rev_values[idx] = acc_rev
            acc_pair_values[idx] = acc_pair
            acc_gcnii_values[idx] = acc_gcnii
            acc_appnp_values[idx] = acc_appnp
        
        # Incremental save: save immediately after each layer completes
        np.savez(
            result_path,
            x=x_values,
            ours=acc_ours_values,
            dropedge=acc_drop_values,
            mbagcn=acc_mba_values,
            reversegnn=acc_rev_values,
            pairnorm=acc_pair_values,
            gcnii=acc_gcnii_values,
            appnp=acc_appnp_values,
        )
        print(f"--> Saved progress up to layer {k}")

    fig = plt.figure(figsize=(8, 8), facecolor='white')
    plt.title(f'Balanced Two-Cluster Sampling on {ds_name}', fontsize=18, fontweight='bold', pad=15)
    plt.xlabel('Number of Layers', fontsize=18, fontweight='medium', labelpad=8)
    plt.ylabel('Test Accuracy', fontsize=18, fontweight='medium', labelpad=8)
    
    plt.plot(x_values, acc_ours_values, linestyle='-', linewidth=2.5, marker='s', markersize=8, color='tab:red', label=r'Ours')
    plt.plot(x_values, acc_drop_values, linestyle='--', linewidth=2, marker='o', markersize=8, color='tab:blue', label='DropEdge')
    plt.plot(x_values, acc_mba_values, linestyle='-.', linewidth=2, marker='^', markersize=8, color='tab:green', label='MbaGCN')
    plt.plot(x_values, acc_rev_values, linestyle=':', linewidth=2, marker='v', markersize=8, color='tab:purple', label='Reverse-GNN')
    plt.plot(x_values, acc_pair_values, linestyle='-', linewidth=2, marker='D', markersize=7, color='tab:orange', label='PairNorm')
    plt.plot(x_values, acc_gcnii_values, linestyle='--', linewidth=2, marker='P', markersize=7, color='tab:brown', label='GCNII')
    plt.plot(x_values, acc_appnp_values, linestyle='-.', linewidth=2, marker='X', markersize=7, color='tab:cyan', label='APPNP')
    
    plt.grid(True, linestyle=':', alpha=0.7, color='gray')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3, fontsize=18, frameon=True, shadow=True)
    plt.tight_layout()
    plt.show()

    fig.savefig(f'result/real_sample_1/Ours_vs_Oversmoothing_Baselines_{ds_name}_layers.pdf', bbox_inches='tight', pad_inches=0.2)

class OGBDatasetWrapper:
    def __init__(self, name, root='data/'):
        self.name = name
        self.root = root
        self.dataset = PygNodePropPredDataset(name=name, root=root, transform=T.ToUndirected())
        self.num_features = self.dataset.num_features
        self.num_classes = self.dataset.num_classes
        self._data = self.dataset[0]
        
        split_idx = self.dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
        
        self._data.train_mask = torch.zeros(self._data.num_nodes, dtype=torch.bool)
        self._data.train_mask[train_idx] = True
        
        self._data.val_mask = torch.zeros(self._data.num_nodes, dtype=torch.bool)
        self._data.val_mask[valid_idx] = True
        
        self._data.test_mask = torch.zeros(self._data.num_nodes, dtype=torch.bool)
        self._data.test_mask[test_idx] = True
        
        if self._data.y.dim() > 1 and self._data.y.shape[1] == 1:
            self._data.y = self._data.y.squeeze(1)

if __name__ == "__main__":
    datasets = [
        Planetoid(root='data/', name='Cora', transform=T.ToUndirected()),
        Planetoid(root='data/', name='CiteSeer', transform=T.ToUndirected()),
        Planetoid(root='data/', name='PubMed', transform=T.ToUndirected()),
        OGBDatasetWrapper(name='ogbn-arxiv'),
        OGBDatasetWrapper(name='ogbn-products'),
        Reddit(root='data/Reddit', transform=T.ToUndirected())
    ]

    cora = datasets[0]
    per_dataset(cora, n_trials=50)

    citeseer = datasets[1]
    per_dataset(citeseer, n_trials=50)

    pubmed = datasets[2]
    per_dataset(pubmed, n_trials=50)

    arxiv = datasets[3]
    per_dataset(arxiv, n_trials=50)

    reddit = datasets[5]
    per_dataset(reddit, n_trials=50)

    products = datasets[4]
    per_dataset(products, n_trials=50)
