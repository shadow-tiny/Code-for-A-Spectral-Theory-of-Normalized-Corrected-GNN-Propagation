import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as utils
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from tqdm import tqdm

# Required to avoid type 3 fonts in figure pdfs.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

torch.set_printoptions(precision=2,sci_mode=False, linewidth=200)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)
loss_fn = nn.BCEWithLogitsLoss()


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

def generate_csbm_data(n_points, n_features, sigma, p, q):
    mu = torch.zeros(n_features, dtype=torch.float, device=device)
    mu[0] = 0.5
    X = np.random.normal(scale=sigma, size=(n_points, n_features))
    X = torch.tensor(X, dtype=torch.float, device=device)
    X[:n_points//2] -= mu
    X[n_points//2:] += mu
    y = torch.zeros(n_points, dtype=torch.long, device=device)
    y[n_points//2:] = 1.0
    data = Data(x=X, y=y, edge_index=None)
    
    probs = torch.tensor([[p, q], [q, p]], dtype=torch.float).to(device)
    row, col = torch.combinations(torch.arange(n_points), r=2, with_replacement=True).t().to(device)
    mask = torch.bernoulli(probs[data.y[row], data.y[col]]).to(torch.bool)
    data.edge_index = torch.stack([row[mask], col[mask]], dim=0)
    data.edge_index = utils.to_undirected(data.edge_index, num_nodes=n_points)
    data.y = data.y.to(torch.float).unsqueeze(1)
    return data

def train_model(model, lr, data):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data = data.to(device)
    for _ in range(50):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()

def test_model(model, data):
    model.to(device)
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = (out > 0.0).float()
    acc = (pred == data.y).float().mean().item()
    return acc

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

class GCNCorrected1(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, num_convolutions):
        super(GCNCorrected1, self).__init__()
        self.n_layers = n_layers
        self.relu = nn.ReLU()
        channels = [input_dim] + [hidden_dim]*(n_layers-1) + [output_dim]
        self.module_list = nn.ModuleList([
            CorrectedConv1(channels[i], channels[i+1], num_convolutions) for i in range(n_layers)
        ])

    def forward(self, x, edge_index):
        for (i, module) in enumerate(self.module_list):
            x = module(x, edge_index)
            x = self.relu(x) if i < self.n_layers - 1 else x
        return x

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

class GCNDropEdge(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, num_convolutions):
        super(GCNDropEdge, self).__init__()
        self.n_layers = n_layers
        self.relu = nn.ReLU()
        channels = [input_dim] + [hidden_dim]*(n_layers-1) + [output_dim]
        self.module_list = nn.ModuleList([
            GCNDropEdgeConv(channels[i], channels[i+1], num_convolutions) for i in range(n_layers)
        ])

    def forward(self, x, edge_index):
        for i, module in enumerate(self.module_list):
            x = module(x, edge_index)
            x = self.relu(x) if i < self.n_layers - 1 else x
        return x

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

class ReverseGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, num_convolutions):
        super(ReverseGNN, self).__init__()
        self.n_layers = n_layers
        self.relu = nn.ReLU()
        channels = [input_dim] + [hidden_dim]*(n_layers-1) + [output_dim]
        self.module_list = nn.ModuleList([
            ReverseGNNConv(channels[i], channels[i+1], num_convolutions) for i in range(n_layers)
        ])

    def forward(self, x, edge_index):
        for i, module in enumerate(self.module_list):
            x = module(x, edge_index)
            x = self.relu(x) if i < self.n_layers - 1 else x
        return x

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

class MbaGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, num_convolutions):
        super(MbaGCN, self).__init__()
        self.n_layers = n_layers
        self.relu = nn.ReLU()
        channels = [input_dim] + [hidden_dim]*(n_layers-1) + [output_dim]
        self.module_list = nn.ModuleList([
            MbaGCNConv(channels[i], channels[i+1], num_convolutions) for i in range(n_layers)
        ])

    def forward(self, x, edge_index):
        for i, module in enumerate(self.module_list):
            x = module(x, edge_index)
            x = self.relu(x) if i < self.n_layers - 1 else x
        return x


class PairNormConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_convolutions, pairnorm_mode='PN-SI'):
        super().__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.num_convolutions = num_convolutions
        self.pairnorm = PairNorm(mode=pairnorm_mode)
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
            x = self.propagate(edge_index, x=x, norm=norm)
            x = self.pairnorm(x)

        x += self.bias
        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class PairNormGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, num_convolutions):
        super().__init__()
        self.n_layers = n_layers
        self.relu = nn.ReLU()
        channels = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
        self.module_list = nn.ModuleList([
            PairNormConv(channels[i], channels[i + 1], num_convolutions) for i in range(n_layers)
        ])

    def forward(self, x, edge_index):
        for i, module in enumerate(self.module_list):
            x = module(x, edge_index)
            x = self.relu(x) if i < self.n_layers - 1 else x
        return x


class GCNIIConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_convolutions, alpha=0.1, lamda=0.5):
        super().__init__(aggr='add')
        self.input_lin = nn.Linear(in_channels, out_channels, bias=False)
        self.lin1 = nn.Linear(out_channels, out_channels, bias=False)
        self.lin2 = nn.Linear(out_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.num_convolutions = num_convolutions
        self.alpha = alpha
        self.lamda = lamda
        self.reset_parameters()

    def reset_parameters(self):
        self.input_lin.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        edge_index, _ = utils.add_remaining_self_loops(edge_index, num_nodes=x.size(0))
        x = self.input_lin(x)
        h0 = x
        row, col = edge_index
        deg = utils.degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        for layer_idx in range(self.num_convolutions):
            beta = math.log(self.lamda / (layer_idx + 1) + 1.0)
            support = (1 - beta) * (1 - self.alpha) * x + beta * self.lin1(x)
            initial = (1 - beta) * self.alpha * h0 + beta * self.lin2(h0)
            x = self.propagate(edge_index, x=support, norm=norm) + initial

        x += self.bias
        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class GCNIIBaseline(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, num_convolutions):
        super().__init__()
        self.n_layers = n_layers
        self.relu = nn.ReLU()
        channels = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
        self.module_list = nn.ModuleList([
            GCNIIConv(channels[i], channels[i + 1], num_convolutions) for i in range(n_layers)
        ])

    def forward(self, x, edge_index):
        for i, module in enumerate(self.module_list):
            x = module(x, edge_index)
            x = self.relu(x) if i < self.n_layers - 1 else x
        return x


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


class APPNPBaseline(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, num_convolutions):
        super().__init__()
        self.dropout = 0.5
        self.input_lin = nn.Linear(input_dim, hidden_dim)
        self.output_lin = nn.Linear(hidden_dim, output_dim)
        self.prop = APPNPProp(num_convolutions=num_convolutions, alpha=0.1)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.input_lin(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.output_lin(x)
        return self.prop(x, edge_index)

labels = {
    'GCNCorrected1': r'Ours',
    'DropEdge': 'DropEdge',
    'MbaGCN': 'MbaGCN',
    'ReverseGNN': 'Reverse-GNN',
    'PairNorm': 'PairNorm',
    'GCNII': 'GCNII',
    'APPNP': 'APPNP',
}

linestyles = {
    'GCNCorrected1': '-',
    'DropEdge': '--',
    'MbaGCN': '-.',
    'ReverseGNN': ':',
    'PairNorm': '-',
    'GCNII': '--',
    'APPNP': '-.',
}

linewidths = {
    'GCNCorrected1': 2.5,
    'DropEdge': 2,
    'MbaGCN': 2,
    'ReverseGNN': 2,
    'PairNorm': 2,
    'GCNII': 2,
    'APPNP': 2,
}

markers = {
    'GCNCorrected1': 's',
    'DropEdge': 'o',
    'MbaGCN': '^',
    'ReverseGNN': 'v',
    'PairNorm': 'D',
    'GCNII': 'P',
    'APPNP': 'X',
}

colors = {
    'GCNCorrected1': 'tab:red',
    'DropEdge': 'tab:blue',
    'MbaGCN': 'tab:green',
    'ReverseGNN': 'tab:purple',
    'PairNorm': 'tab:orange',
    'GCNII': 'tab:brown',
    'APPNP': 'tab:cyan',
}

def plot_with_std(x, y, yerr, label, color, linestyle='-', marker='o', linewidth=2):
    y = np.asarray(y)
    plt.plot(x, y, linewidth=linewidth, color=color, linestyle=linestyle, marker=marker, markersize=8, label=label)
    if yerr is not None:
        yerr = np.asarray(yerr)
        plt.fill_between(x, np.clip(y - yerr, 0.5, 1), np.clip(y + yerr, 0.5, 1), color=color, alpha=0.1)

def plot_metrics(fname, xlabel, ylabel, xaxis, yaxes, yerrs, scales=('linear', 'linear'), vert_lines=None):
    fig = plt.figure(figsize=(10, 7), facecolor='white')
    plt.xscale(scales[0])
    plt.yscale(scales[1])
    plt.xlabel(xlabel, fontsize=26, fontweight='medium', labelpad=8)
    plt.ylabel(ylabel, fontsize=18, fontweight='medium', labelpad=8)
    for yaxis, yerr, model_type in zip(yaxes, yerrs, labels.keys()):
        plot_with_std(xaxis, yaxis, yerr, labels[model_type], color=colors[model_type], 
                      linestyle=linestyles[model_type], marker=markers[model_type],
                      linewidth=linewidths[model_type])
    
    if vert_lines is not None:
        for vert_line in vert_lines:
            plt.axvline(x=vert_line[0], color=vert_line[2], linestyle=vert_line[3], linewidth=2, label=vert_line[1])
    
    plt.grid(True, linestyle=':', alpha=0.7, color='gray')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=18, frameon=True, shadow=True)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    fig.savefig(fname, bbox_inches='tight', pad_inches=0.2)

def experiment(n_trials, model, n, d, sigma, p, q, pbar=None):
    accs = np.zeros(n_trials)
    for t in range(n_trials):
        model.to(device)
        train_data = generate_csbm_data(n, d, sigma, p, q)
        train_model(model, 0.01, train_data)
        test_data = generate_csbm_data(n, d, sigma, p, q)
        accs[t] = test_model(model, test_data)
        if pbar is not None:
            pbar.set_postfix({'Trial': t+1, 'Accuracy': accs[t]})
    return accs.mean().item(), accs.std().item()

def evaluate_metrics_sigma(n_trials, n, d, sigmas, p, q, num_convolutions):
    accs_means = {k: np.zeros(len(sigmas)) for k in labels.keys()}
    accs_stds = {k: np.zeros(len(sigmas)) for k in labels.keys()}
    mbar = tqdm(sigmas, desc='Varying sigma')
    for i, sigma in enumerate(mbar):
        model_ours = GCNCorrected1(input_dim=d, hidden_dim=1, output_dim=1, n_layers=1, num_convolutions=num_convolutions)
        accs_means['GCNCorrected1'][i], accs_stds['GCNCorrected1'][i] = experiment(n_trials, model_ours, n, d, sigma, p, q)
        
        model_drop = GCNDropEdge(input_dim=d, hidden_dim=1, output_dim=1, n_layers=1, num_convolutions=num_convolutions)
        accs_means['DropEdge'][i], accs_stds['DropEdge'][i] = experiment(n_trials, model_drop, n, d, sigma, p, q)
        
        model_mba = MbaGCN(input_dim=d, hidden_dim=1, output_dim=1, n_layers=1, num_convolutions=num_convolutions)
        accs_means['MbaGCN'][i], accs_stds['MbaGCN'][i] = experiment(n_trials, model_mba, n, d, sigma, p, q)
        
        model_rev = ReverseGNN(input_dim=d, hidden_dim=1, output_dim=1, n_layers=1, num_convolutions=num_convolutions)
        accs_means['ReverseGNN'][i], accs_stds['ReverseGNN'][i] = experiment(n_trials, model_rev, n, d, sigma, p, q)

        model_pair = PairNormGCN(input_dim=d, hidden_dim=1, output_dim=1, n_layers=1, num_convolutions=num_convolutions)
        accs_means['PairNorm'][i], accs_stds['PairNorm'][i] = experiment(n_trials, model_pair, n, d, sigma, p, q)

        model_gcnii = GCNIIBaseline(input_dim=d, hidden_dim=1, output_dim=1, n_layers=1, num_convolutions=num_convolutions)
        accs_means['GCNII'][i], accs_stds['GCNII'][i] = experiment(n_trials, model_gcnii, n, d, sigma, p, q)

        model_appnp = APPNPBaseline(input_dim=d, hidden_dim=1, output_dim=1, n_layers=1, num_convolutions=num_convolutions)
        accs_means['APPNP'][i], accs_stds['APPNP'][i] = experiment(n_trials, model_appnp, n, d, sigma, p, q)
    
    return [accs_means[k] for k in labels.keys()], [accs_stds[k] for k in labels.keys()]

def evaluate_metrics_gamma(n_trials, n, d, sigma, p, qs, num_convolutions):
    accs_means = {k: np.zeros(len(qs)) for k in labels.keys()}
    accs_stds = {k: np.zeros(len(qs)) for k in labels.keys()}
    mbar = tqdm(qs, desc='Varying gamma')
    for i, q in enumerate(mbar):
        model_ours = GCNCorrected1(input_dim=d, hidden_dim=1, output_dim=1, n_layers=1, num_convolutions=num_convolutions)
        accs_means['GCNCorrected1'][i], accs_stds['GCNCorrected1'][i] = experiment(n_trials, model_ours, n, d, sigma, p, q)
        
        model_drop = GCNDropEdge(input_dim=d, hidden_dim=1, output_dim=1, n_layers=1, num_convolutions=num_convolutions)
        accs_means['DropEdge'][i], accs_stds['DropEdge'][i] = experiment(n_trials, model_drop, n, d, sigma, p, q)
        
        model_mba = MbaGCN(input_dim=d, hidden_dim=1, output_dim=1, n_layers=1, num_convolutions=num_convolutions)
        accs_means['MbaGCN'][i], accs_stds['MbaGCN'][i] = experiment(n_trials, model_mba, n, d, sigma, p, q)
        
        model_rev = ReverseGNN(input_dim=d, hidden_dim=1, output_dim=1, n_layers=1, num_convolutions=num_convolutions)
        accs_means['ReverseGNN'][i], accs_stds['ReverseGNN'][i] = experiment(n_trials, model_rev, n, d, sigma, p, q)

        model_pair = PairNormGCN(input_dim=d, hidden_dim=1, output_dim=1, n_layers=1, num_convolutions=num_convolutions)
        accs_means['PairNorm'][i], accs_stds['PairNorm'][i] = experiment(n_trials, model_pair, n, d, sigma, p, q)

        model_gcnii = GCNIIBaseline(input_dim=d, hidden_dim=1, output_dim=1, n_layers=1, num_convolutions=num_convolutions)
        accs_means['GCNII'][i], accs_stds['GCNII'][i] = experiment(n_trials, model_gcnii, n, d, sigma, p, q)

        model_appnp = APPNPBaseline(input_dim=d, hidden_dim=1, output_dim=1, n_layers=1, num_convolutions=num_convolutions)
        accs_means['APPNP'][i], accs_stds['APPNP'][i] = experiment(n_trials, model_appnp, n, d, sigma, p, q)
    
    return [accs_means[k] for k in labels.keys()], [accs_stds[k] for k in labels.keys()]

# Varying $\sigma$
n_trials = 50
n = 2000
d = 20
sigmas = np.geomspace(0.1, 20, num=30) 
ratios = 1/sigmas
p = 0.03*(np.log(n)**5)/n
q = p/5
gamma = (p-q)/(p+q)
C1, C2, C = 6, 2, 1
ratio_thres_1 = 5*np.sqrt(np.log(n)/n)
ratio_thres_2 = lambda k: np.sqrt(np.log(n))*(C2*np.sqrt(np.log(n))/(gamma*np.sqrt(n*p)))**k

for num_convs in [1,2,4,8,10,12,16]:
    os.makedirs('./result/synthetic_1', exist_ok=True)
    data_fname = f'./result/synthetic_1/sigma_n={n}_d={d}_p={p:.2f}_q={q:.2f}_k={num_convs}.npz'
    
    if os.path.exists(data_fname):
        print(f"Loading existing results from {data_fname}")
        data_loaded = np.load(data_fname)
        if {'pairnorm_mean', 'gcnii_mean', 'appnp_mean', 'pairnorm_std', 'gcnii_std', 'appnp_std'}.issubset(set(data_loaded.files)):
            accs_means = [
                data_loaded['ours_mean'],
                data_loaded['dropedge_mean'],
                data_loaded['mbagcn_mean'],
                data_loaded['reversegnn_mean'],
                data_loaded['pairnorm_mean'],
                data_loaded['gcnii_mean'],
                data_loaded['appnp_mean'],
            ]
            accs_stds = [
                data_loaded['ours_std'],
                data_loaded['dropedge_std'],
                data_loaded['mbagcn_std'],
                data_loaded['reversegnn_std'],
                data_loaded['pairnorm_std'],
                data_loaded['gcnii_std'],
                data_loaded['appnp_std'],
            ]
        else:
            print('Existing cache is missing PairNorm/GCNII/APPNP fields. Recomputing this file.')
            accs_means, accs_stds = evaluate_metrics_sigma(n_trials, n, d, sigmas, p, q, num_convolutions=num_convs)
            for yaxis in accs_means:
                for i in range(1, len(yaxis)-1):
                    yaxis[i] = (1/3)*(yaxis[i-1] + yaxis[i] + yaxis[i+1])
            np.savez(
                data_fname,
                ratios=ratios, sigmas=sigmas,
                ours_mean=accs_means[0], dropedge_mean=accs_means[1], mbagcn_mean=accs_means[2], reversegnn_mean=accs_means[3],
                pairnorm_mean=accs_means[4], gcnii_mean=accs_means[5], appnp_mean=accs_means[6],
                ours_std=accs_stds[0], dropedge_std=accs_stds[1], mbagcn_std=accs_stds[2], reversegnn_std=accs_stds[3],
                pairnorm_std=accs_stds[4], gcnii_std=accs_stds[5], appnp_std=accs_stds[6],
            )
    else:
        accs_means, accs_stds = evaluate_metrics_sigma(n_trials, n, d, sigmas, p, q, num_convolutions=num_convs)
        for yaxis in accs_means:
            for i in range(1, len(yaxis)-1):
                yaxis[i] = (1/3)*(yaxis[i-1] + yaxis[i] + yaxis[i+1])
        np.savez(data_fname, 
                 ratios=ratios, sigmas=sigmas,
                 ours_mean=accs_means[0], dropedge_mean=accs_means[1], mbagcn_mean=accs_means[2], reversegnn_mean=accs_means[3],
                 pairnorm_mean=accs_means[4], gcnii_mean=accs_means[5], appnp_mean=accs_means[6],
                 ours_std=accs_stds[0], dropedge_std=accs_stds[1], mbagcn_std=accs_stds[2], reversegnn_std=accs_stds[3],
                 pairnorm_std=accs_stds[4], gcnii_std=accs_stds[5], appnp_std=accs_stds[6])

    ratio_vert_1 = np.max([ratio_thres_1, C*ratio_thres_2(num_convs)])
    vert_lines = [[ratio_vert_1, r'Ours threshold', colors['GCNCorrected1'], '--']]
    plot_metrics(
        fname=f'./result/synthetic_1/sigma_n={n}_d={d}_p={p:.2f}_q={q:.2f}_k={num_convs}.pdf',
        xlabel=r'$\frac{\|\mu-\nu\|}{\sigma}$', ylabel='Accuracy',
        xaxis=ratios, yaxes=accs_means, yerrs=accs_stds,
        scales=('log', 'linear'), vert_lines=vert_lines)

# Varying $\gamma$
n_trials = 50
n = 2000
d = 20
sigma = 1
p = 0.01*(np.log(n)**5)/n
qs = np.linspace(0, p, num=30)
gammas = np.array([(p-q)/(p+q) for q in qs])
C_2 = 2.5
gamma_thres_1 = lambda k: 0.5 * ((sigma*sigma*np.log(n))**(0.5/k)) * C_2 * np.sqrt(np.log(n)) / np.sqrt(n*p)

for num_convs in [1,2,3,4,5,6]:
    data_fname = f'./result/synthetic_1/gamma_n={n}_d={d}_sigma={sigma}_p={p:.2f}_k={num_convs}.npz'
    
    if os.path.exists(data_fname):
        print(f"Loading existing results from {data_fname}")
        data_loaded = np.load(data_fname)
        if {'pairnorm_mean', 'gcnii_mean', 'appnp_mean', 'pairnorm_std', 'gcnii_std', 'appnp_std'}.issubset(set(data_loaded.files)):
            accs_means = [
                data_loaded['ours_mean'],
                data_loaded['dropedge_mean'],
                data_loaded['mbagcn_mean'],
                data_loaded['reversegnn_mean'],
                data_loaded['pairnorm_mean'],
                data_loaded['gcnii_mean'],
                data_loaded['appnp_mean'],
            ]
            accs_stds = [
                data_loaded['ours_std'],
                data_loaded['dropedge_std'],
                data_loaded['mbagcn_std'],
                data_loaded['reversegnn_std'],
                data_loaded['pairnorm_std'],
                data_loaded['gcnii_std'],
                data_loaded['appnp_std'],
            ]
        else:
            print('Existing cache is missing PairNorm/GCNII/APPNP fields. Recomputing this file.')
            accs_means, accs_stds = evaluate_metrics_gamma(n_trials, n, d, sigma, p, qs, num_convolutions=num_convs)
            for yaxis in accs_means:
                for i in range(1, len(yaxis)-1):
                    yaxis[i] = (1/3)*(yaxis[i-1] + yaxis[i] + yaxis[i+1])
            np.savez(
                data_fname,
                gammas=gammas,
                ours_mean=accs_means[0], dropedge_mean=accs_means[1], mbagcn_mean=accs_means[2], reversegnn_mean=accs_means[3],
                pairnorm_mean=accs_means[4], gcnii_mean=accs_means[5], appnp_mean=accs_means[6],
                ours_std=accs_stds[0], dropedge_std=accs_stds[1], mbagcn_std=accs_stds[2], reversegnn_std=accs_stds[3],
                pairnorm_std=accs_stds[4], gcnii_std=accs_stds[5], appnp_std=accs_stds[6],
            )
    else:
        accs_means, accs_stds = evaluate_metrics_gamma(n_trials, n, d, sigma, p, qs, num_convolutions=num_convs)
        for yaxis in accs_means:
            for i in range(1, len(yaxis)-1):
                yaxis[i] = (1/3)*(yaxis[i-1] + yaxis[i] + yaxis[i+1])
        np.savez(data_fname, 
                 gammas=gammas, 
                 ours_mean=accs_means[0], dropedge_mean=accs_means[1], mbagcn_mean=accs_means[2], reversegnn_mean=accs_means[3],
                 pairnorm_mean=accs_means[4], gcnii_mean=accs_means[5], appnp_mean=accs_means[6],
                 ours_std=accs_stds[0], dropedge_std=accs_stds[1], mbagcn_std=accs_stds[2], reversegnn_std=accs_stds[3],
                 pairnorm_std=accs_stds[4], gcnii_std=accs_stds[5], appnp_std=accs_stds[6])
             
    vert_lines = [[gamma_thres_1(num_convs), r'Ours threshold', colors['GCNCorrected1'], '--']]
                  
    plot_metrics(
        fname=f'./result/synthetic_1/gamma_n={n}_d={d}_sigma={sigma}_p={p:.2f}_k={num_convs}.pdf',
        xlabel=r'$\gamma$', ylabel='Accuracy',
        xaxis=gammas, yaxes=accs_means, yerrs=accs_stds,
        vert_lines=vert_lines)
