from torch_geometric.datasets import TUDataset
import numpy as np

root = "src/datasets/"
dataset = "MUTAG"

def load_dataset(root, name):
    dataset = TUDataset(root, name)
    print(f"{len(dataset)}, {dataset.num_node_features}, {dataset.num_classes}")
    return dataset



def get_dataset_stats(dataset):
    dataset_nodes = []
    dataset_edges = []
    for g in dataset:
        dataset_nodes.append(g.num_nodes)
        dataset_edges.append(g.num_edges)

    mean_nodes = np.mean(dataset_nodes)
    mean_edges = np.mean(dataset_edges)
        
    return {
        "num_graphs": len(dataset),
        "node_features": dataset.num_node_features,
        "num_classes": dataset.num_classes,
        "mean_nodes": mean_nodes,
        "mean_edges": mean_edges
    }