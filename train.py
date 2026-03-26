def train(datasets_it, anomaly_class_it):
    import os
    import random
    import numpy as np

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from torch.utils.data import Dataset, DataLoader as TorchDataLoader
    from sklearn.metrics import (
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from tqdm.auto import tqdm

    from torch_geometric.data import Batch
    from torch_geometric.nn import global_mean_pool, global_max_pool
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from torch_geometric.utils import to_dense_adj


    from torch_geometric.utils import degree

    from src.dataset import load_dataset, get_dataset_stats
    from src.models import GAE, SimCLRModel
    from src.augmentations import augment
    from src.losses import NTXentLoss

    def ensure_node_features(data):
        """Use node degree as feature when x is None."""
        if data.x is None:
            num_nodes = data.num_nodes
            deg = degree(data.edge_index[0], num_nodes=num_nodes, dtype=torch.float)
            data.x = deg.unsqueeze(-1)
        return data

    TEMPERATURE = 0.5
    PROJ_DIM = 128
    HIDDEN_DIM = 64

    # set division
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15

    EPOCHS = 100
    BASELINE_EPOCHS = 20
    BATCH_SIZE = 32

    LR = 1e-3
    WEIGHT_DECAY = 1E-4

    DEVICE = torch.device(
        #"cuda" if torch.cuda.is_available()
        #else "mps" if torch.backends.mps.is_available()
        #else 
        "cpu"
    )

    print(DEVICE)

    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    set_seed(42)

    root = "src/datasets/"
    dataset = datasets_it

    dataset = load_dataset(root, dataset)
    stats = get_dataset_stats(dataset)
    print(stats)

    #stratified split
    y = []

    def anomaly_labels_from_y(y, anomaly_class=anomaly_class_it):
        y = np.asarray(y)
        return (y == anomaly_class).astype(int)

    def auc_with_auto_flip(y_true, score, pos_label_name="anomaly"):
        y_true = np.asarray(y_true)
        score = np.asarray(score)

        auc_pos = roc_auc_score(y_true, score)
        auc_neg = roc_auc_score(y_true, -score)

        if auc_neg > auc_pos:
            return auc_neg, True
        return auc_pos, False

    for g in dataset:
        y.append(g.y.item())
    y = np.array(y)

    train_idx, temp_idx = train_test_split(np.arange(len(dataset)), test_size=VAL_SIZE + TEST_SIZE, stratify=y, random_state=42)

    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=y[temp_idx], random_state=42)

    train_graphs = [dataset[i] for i in train_idx]
    val_graphs = [dataset[i] for i in val_idx]
    test_graphs = [dataset[i] for i in test_idx]

    train_loader = PyGDataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = PyGDataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = PyGDataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

    #GAE train
    def train_gae_epoch(model, loader, optimizer, criterion):
        model.train()
        total_loss = 0
        total_items = 0

        for batch in tqdm(loader, desc=f"Training", leave=False):
            batch = batch.to(DEVICE)
            batch = ensure_node_features(batch)

            optimizer.zero_grad()

            z = model(batch.x, batch.edge_index, batch.batch)
            A_hats = model.decode(z, batch.batch)
            A = to_dense_adj(batch.edge_index, batch.batch)
            batch_loss = 0
            for i in range(len(A_hats)):
                A_hat_i = A_hats[i]
                A_i = A[i]

                n = A_hats[i].shape[0]
                A_i = A[i][:n, :n]

                batch_loss += criterion(A_hat_i, A_i)


            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            total_items += len(A_hats)
        
        return total_loss / total_items

    @torch.no_grad()
    def evaluate_gae(model, loader):
        model.eval()

        all_anomaly_score = []
        all_labels = []

        for graphs in loader:
            graphs = graphs.to(DEVICE)
            graphs = ensure_node_features(graphs)

            z = model(graphs.x, graphs.edge_index, graphs.batch)

            A_hats = model.decode(z, graphs.batch)
            A = to_dense_adj(graphs.edge_index, graphs.batch)

            for i in range(len(A_hats)):
                n = A_hats[i].shape[0]

                score_i = F.mse_loss(A_hats[i], A[i][:n, :n]).item()
                all_anomaly_score.append(score_i)
                all_labels.append(graphs.y[i].item())
        
        score = np.array(np.negative(all_anomaly_score))
        labels = np.array(all_labels)

        return {
            "score": score,
            "labels": labels
        }

    #instance model GAE
    node_feature_dim = dataset.num_node_features if dataset.num_node_features > 0 else 1
    gae_model = GAE(node_feature_dim, HIDDEN_DIM).to(DEVICE)

    gae_optimizer = torch.optim.Adam(gae_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    gae_criterion = nn.BCELoss()
    gae_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(gae_optimizer, mode='min')

    best_val_bce = float("inf")
    patience = 10
    patience_counter = 0
    os.makedirs("models/", exist_ok=True)
    best_model_path = "models/best_gae.pt"

    #train loop
    history = []
    for epoch in range(1, BASELINE_EPOCHS + 1):
        train_loss = train_gae_epoch(gae_model, train_loader, gae_optimizer, gae_criterion)
        val_metrics = evaluate_gae(gae_model, val_loader)
        gae_scheduler.step(val_metrics["score"].mean())

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_score': val_metrics['score'].mean()
        })

        print(
            f"Epoch {epoch:02d}/{BASELINE_EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"BCE={val_metrics['score'].mean():.4f} | "
        )

        if val_metrics["score"].mean() < best_val_bce:
            best_val_bce = val_metrics["score"].mean()
            patience_counter = 0
            torch.save(gae_model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    gae_model.load_state_dict(torch.load("models/best_gae.pt"))
    test_metrics = evaluate_gae(gae_model, test_loader)
    y_anom = anomaly_labels_from_y(test_metrics["labels"], anomaly_class=anomaly_class_it)
    best_auc, flipped = auc_with_auto_flip(y_anom, test_metrics["score"])
    print(f"GAE Test AUC: {best_auc:.4f} | flipped_score={flipped}")

    #main model
    class SimCLRGraphDataset(Dataset):
        def __init__(self, graphs, p_node=0.1, p_edge=0.1):
            self.graphs = graphs
            self.p_node = p_node
            self.p_edge = p_edge
        
        def __len__(self):
            return len(self.graphs)
        
        def __getitem__(self, idx):
            data = self.graphs[idx]
            view1 = augment(data, self.p_node, self.p_edge)
            view2 = augment(data, self.p_node, self.p_edge)
            return view1, view2
        
    def collate_fn(batch):
        batch_view1, batch_view2 = zip(*batch)
        batch_view1 = list(batch_view1)
        batch_view2 = list(batch_view2)
        batch_view1 = Batch.from_data_list(batch_view1)
        batch_view2 = Batch.from_data_list(batch_view2)
        return batch_view1, batch_view2

    train_graphs_CLR = SimCLRGraphDataset(train_graphs)
    val_graphs_CLR = SimCLRGraphDataset(val_graphs)
    test_graphs_CLR = SimCLRGraphDataset(test_graphs)

    CLR_train_loader = PyGDataLoader(train_graphs_CLR,batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    CLR_val_loader = PyGDataLoader(val_graphs_CLR,batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    CLR_test_loader = PyGDataLoader(test_graphs_CLR,batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    in_channels = dataset[0].x.shape[1] if dataset[0].x is not None else 1
    CLR_model = SimCLRModel(in_channels=in_channels, hidden_channels=HIDDEN_DIM, proj_dim=PROJ_DIM).to(DEVICE)

    CLR_optimizer = torch.optim.Adam(CLR_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    CLR_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(CLR_optimizer, mode='min')
    CLR_criterion = NTXentLoss(temperature=TEMPERATURE)

    #train loop
    def train_one_epoch_CLR(model, loader, optimizer, criterion):
        model.train()
        total_loss = 0
        total_graphs = 0
        
        for batch in tqdm(loader, desc=f"Training", leave=False):
            view1, view2 = batch

            view1 = view1.to(DEVICE)
            view2 = view2.to(DEVICE)
            view1 = ensure_node_features(view1)
            view2 = ensure_node_features(view2)

            optimizer.zero_grad()
            z1, _, _ = model(view1.x, view1.edge_index, view1.batch)
            z2, _, _ = model(view2.x, view2.edge_index, view2.batch)
            loss = criterion(z1,z2)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            total_loss += loss.item() * view1.num_graphs
            total_graphs += view1.num_graphs

        return total_loss / total_graphs

    @torch.no_grad()
    def evaluate_CLR(model, loader, criterion):
        model.eval()

        total_loss = 0
        total_graphs = 0

        for view1, view2 in loader:
            view1 = view1.to(DEVICE)
            view2 = view2.to(DEVICE)
            view1 = ensure_node_features(view1)
            view2 = ensure_node_features(view2)

            z1, _, _ = model(view1.x, view1.edge_index, view1.batch)
            z2, _, _ = model(view2.x, view2.edge_index, view2.batch)
            loss = criterion(z1,z2)

            total_loss += loss.item() * view1.num_graphs
            total_graphs += view1.num_graphs
        return total_loss / total_graphs

    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0
    best_model_path = "models/best_CLR.pt"

    history = []
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch_CLR(CLR_model, CLR_train_loader, CLR_optimizer, CLR_criterion)
        val_loss = evaluate_CLR(CLR_model, CLR_val_loader, CLR_criterion)
        CLR_scheduler.step(val_loss)

        history.append({
            'epoch': epoch,
            'val_loss': val_loss,
        })

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(CLR_model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    @torch.no_grad()
    def compute_center(model, loader):
        model.eval()
        z_proj = []
        x_node = []
        x_graph = []
        for view1, view2 in loader:
            view1 = view1.to(DEVICE)
            view1 = ensure_node_features(view1)
            z_proj_batch, x_node_batch, x_graph_batch = model(view1.x, view1.edge_index, view1.batch)  # projector + normalize
            z_proj.append(z_proj_batch)

            x_node.append(x_node_batch)
            x_graph.append(x_graph_batch)

        z_proj_batch = torch.cat(z_proj, dim=0).mean(dim=0)
        x_node_batch = torch.cat(x_node, dim=0).mean(dim=0)
        x_graph_batch = torch.cat(x_graph, dim=0).mean(dim=0)
        return (z_proj_batch, x_node_batch, x_graph_batch)

    @torch.no_grad()
    def compute_anomaly_scores(model, loader, center_node, center_graph):
        model.eval()
        all_scores = []
        all_labels = []
        
        for view1, view2 in loader:
            view1 = view1.to(DEVICE)
            view1 = ensure_node_features(view1)
            z_proj, x_node, x_graph = model(view1.x, view1.edge_index, view1.batch)

            node_errors = torch.sum((x_node - center_node) ** 2, dim=1)  # [total_nós]
            node_errors_per_graph = global_mean_pool(node_errors.unsqueeze(1), view1.batch).squeeze(1)  # [num_grafos]

            graph_errors = torch.sum((x_graph - center_graph) ** 2, dim=1)  # [num_grafos]

            scores = (node_errors_per_graph + graph_errors).cpu().numpy()

            all_scores.extend(scores)
            all_labels.extend(view1.y.cpu().numpy())

        return np.asarray(all_scores), np.asarray(all_labels)


    # calcula centroide no treino
    CLR_model.load_state_dict(torch.load("models/best_CLR.pt", map_location=DEVICE))
    center_proj, center_node, center_graph = compute_center(CLR_model, CLR_train_loader)

    # avalia no teste
    test_scores, test_labels = compute_anomaly_scores(CLR_model, CLR_test_loader, center_node, center_graph)
    y_anom = anomaly_labels_from_y(test_labels, anomaly_class=anomaly_class_it)
    best_auc_CLR, flipped_CLR = auc_with_auto_flip(y_anom, test_scores)
    print(f"SimCLR Test AUC: {best_auc_CLR:.4f} | flipped_score={flipped_CLR}")

    print("\n=== Final Comparison ===")
    print(f"GAE  Test AUC: {best_auc:.4f}")
    print(f"SimCLR Test AUC: {best_auc_CLR:.4f}")
    print(np.unique(y, return_counts=True))

    return {
        "dataset": datasets_it,
        "gae_auc": best_auc,
        "clr_auc": best_auc_CLR
    }