import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data, DataLoader
import pickle
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from model import ThermoAGTGA  # Ensure this is the correct path to your model definition

# Load the best saved model
def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ThermoAGTGA().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_val_loss = checkpoint['loss']
    print(f'Loaded model from epoch {epoch} with validation loss {best_val_loss:.4f}')
    return model, optimizer, device

# Load the test data
def load_test_data(file_path):
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

# Main function for prediction
def predict(model_path, test_data_path):
    model, optimizer, device = load_model(model_path)
    test_data = load_test_data(test_data_path)

    # Create DataLoader for the test data
    batch_size = 16
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Predicting
    model.eval()
    predictions = []
    true_ddg = []
    with torch.no_grad():
        for wild_data, mutant_data, ddg in test_loader:
            wild_data = wild_data.to(device)
            mutant_data = mutant_data.to(device)
            ddg = ddg.to(device)

            output = model(wild_data, mutant_data)
            predictions.extend(output.cpu().numpy())
            true_ddg.extend(ddg.cpu().numpy())

    # Calculate PCC and RMSE
    true_ddg_array = np.array(true_ddg)
    predictions_array = np.array(predictions)

    pcc, _ = pearsonr(true_ddg_array, predictions_array)
    rmse = np.sqrt(mean_squared_error(true_ddg_array, predictions_array))

    print(f'Test PCC: {pcc:.4f}')
    print(f'Test RMSE: {rmse:.4f}')

# Run the prediction
if __name__ == "__main__":
    predict('best_model_thermo_seq.pth', 'data_p553.pkl')

