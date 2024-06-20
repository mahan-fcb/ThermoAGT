import pickle
import torch
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split

def load_dataset(file_path):
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    data_list = []
    for item in dataset:
        wild_type_data = Data(
            x=torch.tensor(item['wild_type']['node_features'], dtype=torch.float),
            edge_index=torch.tensor(item['wild_type']['edge_index'], dtype=torch.long),
            edge_attr=torch.tensor(item['wild_type']['edge_features'], dtype=torch.float),
            batch=torch.zeros(item['wild_type']['node_features'].shape[0], dtype=torch.long)
        )

        mutant_data = Data(
            x=torch.tensor(item['mutant']['node_features'], dtype=torch.float),
            edge_index=torch.tensor(item['mutant']['edge_index'], dtype=torch.long),
            edge_attr=torch.tensor(item['mutant']['edge_features'], dtype=torch.float),
            batch=torch.zeros(item['mutant']['node_features'].shape[0], dtype=torch.long)
        )

        data_list.append((wild_type_data, mutant_data, torch.tensor(item['ddg'], dtype=torch.float)))
    return data_list

def prepare_data(file_path):
    dataset = load_dataset(file_path)
    additional_data = []
    for wild_data, mutant_data, ddg in dataset:
        additional_data.append((mutant_data, wild_data, ddg * -1))
    datasets = dataset + additional_data

    train_data, val_data = train_test_split(datasets, test_size=0.1, random_state=42)

    batch_size = 256
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

