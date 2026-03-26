from train import train

datasets = ["BZR", "DHFR", "COX2", "ENZYMES", "AIDS", "NCI1", "IMDB-BINARY"]

anomaly_class = {
    "BZR": 1,
    "DHFR": 0,
    "COX2": 1,
    "ENZYMES": 0,
    "AIDS": 1,
    "NCI1": 1,
    "IMDB-BINARY": 1
}
results = []

for dataset_name in datasets:
    result = train(datasets_it=dataset_name, anomaly_class_it=anomaly_class[dataset_name])
    results.append(result)
    print(f"Dataset: {dataset_name}")
    print(f"GAE  Test AUC: {result['gae_auc']:.4f}")
    print(f"SimCLR Test AUC: {result['clr_auc']:.4f}")
    print("-" * 40)

print("\n=== Resumo Geral ===")
print(f"{'Dataset':<15} {'GAE AUC':>10} {'SimCLR AUC':>12}")
print("-" * 40)
for dataset_name, result in zip(datasets, results):
    print(f"{dataset_name:<15} {result['gae_auc']:>10.4f} {result['clr_auc']:>12.4f}")