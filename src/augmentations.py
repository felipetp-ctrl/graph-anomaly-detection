import torch

def node_feature_dropout(data, p):
    data_copy = data.clone()
    if data_copy.x is None:
        return data_copy
    mask = torch.bernoulli(torch.full((data_copy.num_nodes, 1), 1 - p))
    data_copy.x = data_copy.x * mask
    return data_copy

def edge_dropout(data, p):
        data_copy = data.clone()
        mask = torch.bernoulli(torch.full((data_copy.num_edges, ), 1 - p))
        data_copy.edge_index = data_copy.edge_index[:, mask.bool()]
        if data_copy.edge_attr is not None:
            data_copy.edge_attr = data_copy.edge_attr[mask.bool()]
        return data_copy

def augment(data, p_node, p_edge):
        data = node_feature_dropout(data, p_node)
        data = edge_dropout(data, p_edge)
        return data