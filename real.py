import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as utils
from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.nn.conv import MessagePassing, GCNConv
from tqdm import tqdm
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T


# Required to avoid type 3 fonts in figure pdfs.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

torch.set_printoptions(precision=2,sci_mode=False, linewidth=200)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)
loss_fn = nn.CrossEntropyLoss()


def accuracy(output, labels):
    preds = output.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)

def train_model(model, optimizer, data, epochs=200):
    model.to(device)
    model.train()
    data = data.to(device)
    # pbar = tqdm(range(epochs), leave=False, desc='Training')
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        # print(f'Epoch {epoch+1:03d}: Loss = {loss.item():.4f}')
        # pbar.set_postfix({'Loss': loss.item()})

def train_minibatch(model, optimizer, loader, epochs=5):
    model.to(device)
    model.train()
    # For million-node graphs, 5 epochs are sufficient for convergence after sampling only training nodes
    for epoch in range(epochs):
        for data in loader:
            data = data.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            # Calculate loss only on the sampled target nodes (batch_size)
            batch_size = data.batch_size
            # Since input_nodes=train_mask is specified in the loader, target nodes here are all training nodes
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
            
            # Use the global accuracy function, but multiply by batch_size to restore the correct sample count
            acc = accuracy(out[:batch_size], data.y[:batch_size])
            total_correct += acc * batch_size
            total_samples += batch_size
            
    return total_correct / total_samples

def test_model_cpu(model, data):
    model.cpu()
    model.eval()
    # Forward pass of deep networks on large graphs is performed on CPU to prevent GPU OOM
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



class CorrectedConv1(MessagePassing):
    def __init__(self, in_channels, out_channels, num_convolutions):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.num_convolutions = num_convolutions
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # Linear transformation of the input features.
        x = self.lin(x)

        # Add self-loops to the adjacency matrix.
        edge_index, _ = utils.add_remaining_self_loops(edge_index, num_nodes=x.size(0))

        # Compute normalization.
        row, col = edge_index
        deg = utils.degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # To isolate the rank-1 component
        deg_sqrt = deg.pow(0.5).unsqueeze(1)
        total_edges = edge_index.size(1)
        
        # Propagate messages and remove rank-1 component, add bias.
        for _ in range(self.num_convolutions):
            rank1_comp = deg_sqrt.T@x
            rank1_comp = deg_sqrt@rank1_comp / total_edges
            x = self.propagate(edge_index, x=x, norm=norm)
            x -= rank1_comp
        
        x += self.bias
        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class CorrectedConv2(MessagePassing):
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
        # Linear transformation of the input features.
        x = self.lin(x)

        # Add self-loops to the adjacency matrix.
        edge_index, _ = utils.add_remaining_self_loops(edge_index, num_nodes=x.size(0))

        # Compute normalization.
        row, col = edge_index
        deg = utils.degree(col, x.size(0), dtype=x.dtype)
        norm = 1.0 / deg.mean()

        # To isolate the rank-1 component
        n = x.size(0)
        J = torch.ones((n, 1), device=x.device) / np.sqrt(n)
        
        # Propagate messages and remove rank-1 component, add bias.
        for _ in range(self.num_convolutions):
            rank1_comp = J.T@x
            rank1_comp = J@rank1_comp
            x = self.propagate(edge_index, x=x, norm=norm)
            x -= rank1_comp
        
        x += self.bias
        return x

    def message(self, x_j, norm):
        return norm * x_j

class GCNConv(MessagePassing):
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
        edge_index, _ = utils.add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        row, col = edge_index
        deg = utils.degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        for _ in range(self.num_convolutions):
            x = self.propagate(edge_index, x=x, norm=norm)
        
        x += self.bias
        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, num_convolutions=1):
        super(GCN, self).__init__()
        self.n_layers = n_layers
        self.relu = nn.ReLU()
        channels = [input_dim] + [hidden_dim]*(n_layers-1) + [output_dim]
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(GCNConv(channels[i], channels[i+1], num_convolutions))
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

class GCNCorrected1(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, num_convolutions=1):
        super(GCNCorrected1, self).__init__()
        self.n_layers = n_layers
        self.relu = nn.ReLU()
        channels = [input_dim] + [hidden_dim]*(n_layers-1) + [output_dim]
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(CorrectedConv1(channels[i], channels[i+1], num_convolutions))
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

class GCNCorrected2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, num_convolutions=1):
        super(GCNCorrected2, self).__init__()
        self.n_layers = n_layers
        self.relu = nn.ReLU()
        channels = [input_dim] + [hidden_dim]*(n_layers-1) + [output_dim]
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(CorrectedConv2(channels[i], channels[i+1], num_convolutions))
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


def experiment(model, data, loader=None, test_loader=None):
    optimizer = torch.optim.Adam(model.parameters(), 0.01, weight_decay=5e-4)
    if loader:
        train_minibatch(model, optimizer, loader)
        if test_loader:
            # Ultra-fast GPU testing
            test_acc = test_minibatch(model, test_loader)
        else:
            # Pass the complete graph data during testing and evaluate on CPU to prevent OOM on large graphs
            test_acc = test_model_cpu(model, data)
    else:
        train_model(model, optimizer, data)
        test_acc = test_model(model, data)
    return test_acc


def per_dataset(dataset, n_trials=10):
    num_features = dataset.num_features
    num_classes = dataset.num_classes

    x_values = []
    acc_gcn_values = []
    acc_gcncorrected1_values = []
    acc_gcncorrected2_values = []

    ds_name = getattr(dataset, 'name', dataset.__class__.__name__)

    import os
    if not os.path.exists('result/real'):
        os.makedirs('result/real')
    result_path = f'result/real/GCN_vs_GCNCorrected_{ds_name}_data.npz'

    x_values = []
    acc_gcn_values = []
    acc_gcncorrected1_values = []
    acc_gcncorrected2_values = []
    completed_layers = set()

    if os.path.exists(result_path):
        print(f"Loading existing results from {result_path}")
        data_loaded = np.load(result_path)
        keys = data_loaded.files
        k_c1 = 'gcncorrected1' if 'gcncorrected1' in keys else 'gcnr1'
        k_c2 = 'gcncorrected2' if 'gcncorrected2' in keys else 'gcnr2'
        x_values = data_loaded['x'].tolist()
        acc_gcn_values = data_loaded['gcn'].tolist()
        acc_gcncorrected1_values = data_loaded[k_c1].tolist()
        acc_gcncorrected2_values = data_loaded[k_c2].tolist()
        completed_layers = set(x_values)
        print(f"Already completed layers: {completed_layers}")

    large_datasets = ['ogbn-products', 'Reddit']
    use_loader = ds_name in large_datasets
    loader = None
    test_loader = None

    if use_loader:
        print(f"Using NeighborLoader for {ds_name} for extreme fast mini-batch training.")
        data = dataset._data # Keep on CPU
        
        
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
        data = dataset._data.to(device)

    plt.figure(figsize=(10, 5))
    pbar = tqdm(range(2, 33, 2), leave=False, desc='Layers')
    for k in pbar:
        if k in completed_layers:
            continue

        acc_gcn = 0.
        acc_gcncorrected1 = 0.
        acc_gcncorrected2 = 0.
        for t in range(n_trials):
            print(f'Layers: {k}, Trial: {t}') # Display current progress
            gcn_model = GCN(input_dim=num_features, hidden_dim=64, output_dim=num_classes, n_layers=k)
            gcnr_model1 = GCNCorrected1(input_dim=num_features, hidden_dim=64, output_dim=num_classes, n_layers=k)
            gcnr_model2 = GCNCorrected2(input_dim=num_features, hidden_dim=64, output_dim=num_classes, n_layers=k)
            
            acc_gcn += experiment(gcn_model, data, loader, test_loader)
            acc_gcncorrected1 += experiment(gcnr_model1, data, loader, test_loader)
            acc_gcncorrected2 += experiment(gcnr_model2, data, loader, test_loader)
        
        acc_gcn /= n_trials
        acc_gcncorrected1 /= n_trials
        acc_gcncorrected2 /= n_trials
        x_values.append(k)
        acc_gcn_values.append(acc_gcn)
        acc_gcncorrected1_values.append(acc_gcncorrected1)
        acc_gcncorrected2_values.append(acc_gcncorrected2)
    
        # Save raw data incrementally
        np.savez(result_path, x=x_values, gcn=acc_gcn_values, gcncorrected1=acc_gcncorrected1_values, gcncorrected2=acc_gcncorrected2_values)
        print(f"--> Saved progress up to layer {k}")

    fig = plt.figure(figsize=(8, 5), facecolor='white')
    plt.title(f'Performance Comparison on {ds_name}', fontsize=18, fontweight='bold', pad=15)
    plt.xlabel('Number of Layers', fontsize=18, fontweight='medium', labelpad=8)
    plt.ylabel('Test Accuracy', fontsize=18, fontweight='medium', labelpad=8)
    
    # Use thicker lines, more prominent markers, and better color schemes
    plt.plot(x_values, acc_gcn_values, linestyle='--', linewidth=2, marker='o', markersize=8, color='tab:blue', label='Standard GCN')
    plt.plot(x_values, acc_gcncorrected1_values, linestyle='-', linewidth=2.5, marker='s', markersize=8, color='tab:red', label='GCN with $\hat{A}$ (Ours)')
    plt.plot(x_values, acc_gcncorrected2_values, linestyle='-.', linewidth=2, marker='^', markersize=8, color='tab:green', label=r'GCN with $\tilde{A}$')
    
    # Add transparent dashed grid lines for readability
    plt.grid(True, linestyle=':', alpha=0.7, color='gray')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Optimize legend style
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.26), ncol=3, fontsize=14, frameon=True, shadow=True)
    plt.tight_layout()
    plt.show()

    fig.savefig(f'result/real/GCN_vs_GCNCorrected_{ds_name}_layers.pdf', bbox_inches='tight', pad_inches=0.2)
    # fig.savefig(f'result/GCN_vs_GCNCorrected_{ds_name}_layers.pdf', bbox_inches='tight')

class OGBDatasetWrapper:
    def __init__(self, name, root='data/'):
        self.name = name
        self.root = root
        self.dataset = PygNodePropPredDataset(name=name, root=root, transform=T.ToUndirected())
        self.num_features = self.dataset.num_features
        self.num_classes = self.dataset.num_classes
        self._data = self.dataset[0]
        
        # Prepare masks for OGB datasets
        split_idx = self.dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
        
        self._data.train_mask = torch.zeros(self._data.num_nodes, dtype=torch.bool)
        self._data.train_mask[train_idx] = True
        
        self._data.val_mask = torch.zeros(self._data.num_nodes, dtype=torch.bool)
        self._data.val_mask[valid_idx] = True
        
        self._data.test_mask = torch.zeros(self._data.num_nodes, dtype=torch.bool)
        self._data.test_mask[test_idx] = True
        
        # OGB labels are usually (N, 1), need to be (N,) for CrossEntropyLoss
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
