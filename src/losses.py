import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        sim_matrix = torch.cat([z1, z2], dim=0)
        sim_matrix = (sim_matrix @ sim_matrix.T / self.temperature)
        mask = torch.eye(sim_matrix.shape[0]).to(z1.device)
        mask = mask.to(torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

        batch_size = z1.shape[0]
        pos_labels = torch.cat([torch.arange(batch_size, 2 * batch_size),torch.arange(0, batch_size)], dim=0)

        loss = F.cross_entropy(sim_matrix, pos_labels.to(z1.device))
        return loss