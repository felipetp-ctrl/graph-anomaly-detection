import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, global_mean_pool, global_max_pool

class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels = 64):
        super().__init__()

        self.conv1 = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU()
            )
        )

        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.conv2 = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU()
            )
        )

        self.bn2 = nn.BatchNorm1d(hidden_channels)
    
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)

        x_mean_pool = global_mean_pool(x, batch)
        x_max_pool = global_max_pool(x, batch)

        x_graph = torch.cat([x_mean_pool, x_max_pool], dim=1)

        return (x, x_graph) 
    
class GAE(nn.Module):
    def __init__(self, in_channels, hidden_channels = 64):
        super().__init__()

        self.conv1 = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU()
            )
        )

        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.conv2 = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU()
            )
        )

        self.bn2 = nn.BatchNorm1d(hidden_channels)

    def decode(self, z, batch_vector):
        unique_graphs = torch.unique(batch_vector)
        A_hats = []
        for i in unique_graphs:
            mask = batch_vector == i
            z_i = z[mask]
            A_hat_i = torch.sigmoid(z_i @ z_i.T)
            A_hats.append(A_hat_i)
        return A_hats
    
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)

        return x
    
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x
    
class SimCLRModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, proj_dim, eta=1.0, sigma=0.1):
        super().__init__()
        self.encoder = GNNEncoder(in_channels, hidden_channels)
        self.projector = ProjectionHead(2 * hidden_channels, proj_dim)
        self.eta = eta
        self.sigma = sigma

    def forward(self, x, edge_index, batch):
        x_node, x_graph = self.encoder(x, edge_index, batch)
        z_proj = self.projector(x_graph)
        return z_proj, x_node, x_graph

    def forward_perturbed(self, x, edge_index, batch):
        original_params = {}
        with torch.no_grad():
            for name, param in self.encoder.named_parameters():
                original_params[name] = param.data.clone()
                noise = torch.randn_like(param.data) * self.sigma
                param.data = param.data + self.eta * noise

        x_node, x_graph = self.encoder(x, edge_index, batch)
        z_proj = self.projector(x_graph)

        with torch.no_grad():
            for name, param in self.encoder.named_parameters():
                param.data = original_params[name]

        return z_proj, x_node, x_graph