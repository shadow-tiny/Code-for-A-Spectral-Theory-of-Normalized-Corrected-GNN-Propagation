import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
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
# This loss function is suitable for binary classification problems
loss_fn = nn.BCEWithLogitsLoss()


# Data generation
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
    
    # The inbuilt function stochastic_blockmodel_graph does not support
    # random permutations of the nodes, hence, design it manually.
    # Use with_replacement=True to include self-loops.
    probs = torch.tensor([[p, q], [q, p]], dtype=torch.float).to(device)
    row, col = torch.combinations(torch.arange(n_points), r=2, with_replacement=True).t().to(device)
    # Perform a Bernoulli trial for each candidate edge to determine if it is kept
    mask = torch.bernoulli(probs[data.y[row], data.y[col]]).to(torch.bool)
    # edge_index is a 2 x num_edges tensor, where each column represents the two endpoints of an edge
    data.edge_index = torch.stack([row[mask], col[mask]], dim=0)
    # Convert to undirected graph
    data.edge_index = utils.to_undirected(data.edge_index, num_nodes=n_points)
    # Data preprocessing
    data.y = data.y.to(torch.float).unsqueeze(1)
    return data


# Training / testing
def train_model(model, lr, data):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data = data.to(device)
    for epoch in range(50):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
        

def test_model(model, data, mask=None):
    model.to(device)
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        # This is where predictions differ: restrict indices first, then evaluate accuracy
        out = model(data.x, data.edge_index)
        if mask is not None:
            out = out[mask]
            target = data.y[mask]
        else:
            target = data.y
        # BCEWithLogitsLoss corresponds to the Sigmoid activation function.
        # When the raw output (logit) is greater than 0, the Sigmoid output is greater than 0.5,
        # so it is classified as class 1; otherwise, it is classified as class 0.
        pred = (out > 0.0).float()
    # Calculate the mean of all elements in the tensor. Since the correct prediction is 1.0
    # and the incorrect one is 0.0, this mean represents the classification accuracy.
    acc = (pred == target).float().mean().item()
    return acc


# Model architectures (corresponding to [hat(A)] in the paper)
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

# (corresponding to [tilde(A)] in the paper)
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
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, num_convolutions):
        super(GCN, self).__init__()
        self.n_layers = n_layers
        self.relu = nn.ReLU()
        channels = [input_dim] + [hidden_dim]*(n_layers-1) + [output_dim]
        self.module_list = []
        for i in range(n_layers):
            self.module_list.append(GCNConv(channels[i], channels[i+1], num_convolutions))
        self.module_list = nn.ModuleList(self.module_list)

    def forward(self, x, edge_index):
        for (i, module) in enumerate(self.module_list):
            x = module(x, edge_index)
            x = self.relu(x) if i < self.n_layers - 1 else x
        return x

class GCNCorrected1(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, num_convolutions):
        super(GCNCorrected1, self).__init__()
        self.n_layers = n_layers
        self.relu = nn.ReLU()
        channels = [input_dim] + [hidden_dim]*(n_layers-1) + [output_dim]
        self.module_list = []
        for i in range(n_layers):
            self.module_list.append(CorrectedConv1(channels[i], channels[i+1], num_convolutions))
        self.module_list = nn.ModuleList(self.module_list)

    def forward(self, x, edge_index):
        for (i, module) in enumerate(self.module_list):
            x = module(x, edge_index)
            x = self.relu(x) if i < self.n_layers - 1 else x
        return x

class GCNCorrected2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, num_convolutions):
        super(GCNCorrected2, self).__init__()
        self.n_layers = n_layers
        self.relu = nn.ReLU()
        channels = [input_dim] + [hidden_dim]*(n_layers-1) + [output_dim]
        self.module_list = []
        for i in range(n_layers):
            self.module_list.append(CorrectedConv2(channels[i], channels[i+1], num_convolutions))
        self.module_list = nn.ModuleList(self.module_list)

    def forward(self, x, edge_index):
        for (i, module) in enumerate(self.module_list):
            x = module(x, edge_index)
            x = self.relu(x) if i < self.n_layers - 1 else x
        return x


# Plotting helpers
labels = {
    'GCN': 'Standard GCN',
    'GCNCorrected1': r'GCN with $\hat{A}$ (Ours)',
    'GCNCorrected2': r'GCN with $\tilde{A}$'
}

linestyles = {
    'GCN': '--',
    'GCNCorrected1': '-',
    'GCNCorrected2': '-.'
}

linewidths = {
    'GCN': 2,
    'GCNCorrected1': 2.5,
    'GCNCorrected2': 2
}

markers = {
    'GCN': 'o',
    'GCNCorrected1': 's',
    'GCNCorrected2': '^'
}

colors = {
    'GCN': 'tab:blue',
    'GCNCorrected1': 'tab:red',
    'GCNCorrected2': 'tab:green'
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
    plt.xlabel(xlabel, fontsize=26, fontweight='medium', labelpad=10)
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
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=15, frameon=True, shadow=True)
    plt.tight_layout()
    plt.show()
    
    # Create directory if it doesn't exist
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


# Parameter-specific helpers for $\sigma$, $\gamma$ and $k=$ num_convs
def evaluate_metrics_sigma(n_trials, n, d, sigmas, p, q, num_convolutions):
    accs_gcn_means = np.zeros(len(sigmas))
    accs_gcn_stds = np.zeros(len(sigmas))
    accs_gcncorrected1_means = np.zeros(len(sigmas))
    accs_gcncorrected1_stds = np.zeros(len(sigmas))
    accs_gcncorrected2_means = np.zeros(len(sigmas))
    accs_gcncorrected2_stds = np.zeros(len(sigmas))
    mbar = tqdm(sigmas, desc='Varying sigma')
    for i, sigma in enumerate(mbar):
        gcn = GCN(input_dim=d, hidden_dim=1, output_dim=1, n_layers=1, num_convolutions=num_convolutions)
        accs_gcn_means[i], accs_gcn_stds[i] = experiment(n_trials, gcn, n, d, sigma, p, q, mbar)
        del gcn
        
        gcncorrected1 = GCNCorrected1(input_dim=d, hidden_dim=1, output_dim=1, n_layers=1, num_convolutions=num_convolutions)
        accs_gcncorrected1_means[i], accs_gcncorrected1_stds[i] = experiment(n_trials, gcncorrected1, n, d, sigma, p, q, mbar)
        del gcncorrected1
        
        gcncorrected2 = GCNCorrected2(input_dim=d, hidden_dim=1, output_dim=1, n_layers=1, num_convolutions=num_convolutions)
        accs_gcncorrected2_means[i], accs_gcncorrected2_stds[i] = experiment(n_trials, gcncorrected2, n, d, sigma, p, q, mbar)
        del gcncorrected2
    
    acc_means = [accs_gcn_means, accs_gcncorrected1_means, accs_gcncorrected2_means]
    acc_stds = [accs_gcn_stds, accs_gcncorrected1_stds, accs_gcncorrected2_stds]
    return acc_means, acc_stds

def evaluate_metrics_gamma(n_trials, n, d, sigma, p, qs, num_convolutions):
    accs_gcn_means = np.zeros(len(qs))
    accs_gcn_stds = np.zeros(len(qs))
    accs_gcncorrected1_means = np.zeros(len(qs))
    accs_gcncorrected1_stds = np.zeros(len(qs))
    accs_gcncorrected2_means = np.zeros(len(qs))
    accs_gcncorrected2_stds = np.zeros(len(qs))
    mbar = tqdm(qs, desc='Varying gamma')
    for i, q in enumerate(mbar):
        gcn = GCN(input_dim=d, hidden_dim=1, output_dim=1, n_layers=1, num_convolutions=num_convolutions)
        accs_gcn_means[i], accs_gcn_stds[i] = experiment(n_trials, gcn, n, d, sigma, p, q, mbar)
        del gcn
        
        gcncorrected1 = GCNCorrected1(input_dim=d, hidden_dim=1, output_dim=1, n_layers=1, num_convolutions=num_convolutions)
        accs_gcncorrected1_means[i], accs_gcncorrected1_stds[i] = experiment(n_trials, gcncorrected1, n, d, sigma, p, q, mbar)
        del gcncorrected1
        
        gcncorrected2 = GCNCorrected2(input_dim=d, hidden_dim=1, output_dim=1, n_layers=1, num_convolutions=num_convolutions)
        accs_gcncorrected2_means[i], accs_gcncorrected2_stds[i] = experiment(n_trials, gcncorrected2, n, d, sigma, p, q, mbar)
        del gcncorrected2
    
    acc_means = [accs_gcn_means, accs_gcncorrected1_means, accs_gcncorrected2_means]
    acc_stds = [accs_gcn_stds, accs_gcncorrected1_stds, accs_gcncorrected2_stds]
    return acc_means, acc_stds


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
ratio_thres_2 = lambda k: np.sqrt(np.log(n))*(C1/(gamma*np.sqrt(n*p)))**k
ratio_thres_3 = 5*np.sqrt(np.log(n)/n)
ratio_thres_4 = lambda k: np.sqrt(np.log(n))*(C2*np.sqrt(np.log(n))/(gamma*np.sqrt(n*p)))**k
for num_convs in [1,2,4,8,10,12,16]:
    print(p, q)
    print(f'Condition for exact recovery: gamma={gamma:.2f} > {0.3*num_convs*np.sqrt(np.log(n)/(n*p)):.2f}')
    
    os.makedirs('./result/synthetic', exist_ok=True)
    data_fname = f'./result/synthetic/sigma_n={n}_d={d}_p={p:.2f}_q={q:.2f}_k={num_convs}.npz'
    
    if os.path.exists(data_fname):
        print(f"Loading existing results from {data_fname}")
        data_loaded = np.load(data_fname)
        keys = data_loaded.files
        k_c1_mean = 'gcncorrected1_mean' if 'gcncorrected1_mean' in keys else 'gcnrob1_mean'
        k_c2_mean = 'gcncorrected2_mean' if 'gcncorrected2_mean' in keys else 'gcnrob2_mean'
        k_c1_std = 'gcncorrected1_std' if 'gcncorrected1_std' in keys else 'gcnrob1_std'
        k_c2_std = 'gcncorrected2_std' if 'gcncorrected2_std' in keys else 'gcnrob2_std'
        
        accs_means = [data_loaded['gcn_mean'], data_loaded[k_c1_mean], data_loaded[k_c2_mean]]
        accs_stds = [data_loaded['gcn_std'], data_loaded[k_c1_std], data_loaded[k_c2_std]]
    else:
        accs_means, accs_stds = evaluate_metrics_sigma(n_trials, n, d, sigmas, p, q, num_convolutions=num_convs)
        for yaxis in accs_means:
            for i in range(1, len(yaxis)-1):
                yaxis[i] = (1/3)*(yaxis[i-1] + yaxis[i] + yaxis[i+1])
        np.savez(data_fname, 
                 ratios=ratios, sigmas=sigmas,
                 gcn_mean=accs_means[0], gcncorrected1_mean=accs_means[1], gcncorrected2_mean=accs_means[2],
                 gcn_std=accs_stds[0], gcncorrected1_std=accs_stds[1], gcncorrected2_std=accs_stds[2])

    ratio_vert_1 = np.max([ratio_thres_1, C*ratio_thres_2(num_convs)])
    ratio_vert_2 = np.max([ratio_thres_3, C*ratio_thres_4(num_convs)])
    vert_lines = [[ratio_vert_1, r'GCN with $\tilde{A}$ threshold', colors['GCNCorrected2'], '-'],
                  [ratio_vert_2, r'GCN with $\hat{A}$ threshold', colors['GCNCorrected1'], '--']]
    plot_metrics(
        fname=f'./result/synthetic/sigma_n={n}_d={d}_p={p:.2f}_q={q:.2f}_k={num_convs}.pdf',
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
C_1 = 8
C_2 = 2.5
gamma_thres_1 = lambda k: 0.5 * ((sigma*sigma*np.log(n))**(0.5/k)) * C_1 / np.sqrt(n*p)
gamma_thres_2 = lambda k: 0.5 * ((sigma*sigma*np.log(n))**(0.5/k)) * C_2 * np.sqrt(np.log(n)) / np.sqrt(n*p)
for num_convs in [1,2,3,4,5,6]:
    print(p, 1./sigma)
    
    os.makedirs('./result/synthetic', exist_ok=True)
    data_fname = f'./result/synthetic/gamma_n={n}_d={d}_sigma={sigma}_p={p:.2f}_k={num_convs}.npz'
    
    if os.path.exists(data_fname):
        print(f"Loading existing results from {data_fname}")
        data_loaded = np.load(data_fname)
        keys = data_loaded.files
        k_c1_mean = 'gcncorrected1_mean' if 'gcncorrected1_mean' in keys else 'gcnrob1_mean'
        k_c2_mean = 'gcncorrected2_mean' if 'gcncorrected2_mean' in keys else 'gcnrob2_mean'
        k_c1_std = 'gcncorrected1_std' if 'gcncorrected1_std' in keys else 'gcnrob1_std'
        k_c2_std = 'gcncorrected2_std' if 'gcncorrected2_std' in keys else 'gcnrob2_std'
        
        accs_means = [data_loaded['gcn_mean'], data_loaded[k_c1_mean], data_loaded[k_c2_mean]]
        accs_stds = [data_loaded['gcn_std'], data_loaded[k_c1_std], data_loaded[k_c2_std]]
    else:
        accs_means, accs_stds = evaluate_metrics_gamma(n_trials, n, d, sigma, p, qs, num_convolutions=num_convs)
        for yaxis in accs_means:
            for i in range(1, len(yaxis)-1):
                yaxis[i] = (1/3)*(yaxis[i-1] + yaxis[i] + yaxis[i+1])
        np.savez(data_fname, 
                 gammas=gammas, 
                 gcn_mean=accs_means[0], gcncorrected1_mean=accs_means[1], gcncorrected2_mean=accs_means[2],
                 gcn_std=accs_stds[0], gcncorrected1_std=accs_stds[1], gcncorrected2_std=accs_stds[2])
                 
    vert_lines = [[gamma_thres_1(num_convs), r'GCN with $\tilde{A}$ threshold', colors['GCNCorrected2'], '-'],
                  [gamma_thres_2(num_convs), r'GCN with $\hat{A}$ threshold', colors['GCNCorrected1'], '--']]
                  
    plot_metrics(
        fname=f'./result/synthetic/gamma_n={n}_d={d}_sigma={sigma}_p={p:.2f}_k={num_convs}.pdf',
        xlabel=r'$\gamma$', ylabel='Accuracy',
        xaxis=gammas, yaxes=accs_means, yerrs=accs_stds,
        vert_lines=vert_lines)
