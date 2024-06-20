import torch
import pickle
import pandas as pd
from torch_geometric.data import Data, DataLoader
from model import CrysCo  # Ensure this is the correct path to your model definition
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the saved model checkpoint
checkpoint_path = 'best_model_thermo_seq.pth'
checkpoint = torch.load(checkpoint_path)

# Initialize the model
model = CrysCo().to(device)

# Initialize the optimizer
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)

# Load the model and optimizer states
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
best_val_loss = checkpoint['loss']

print(f'Loaded model from epoch {epoch} with validation loss {best_val_loss:.4f}')

def load_and_preprocess_dataset(file_path):
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    
    processed_data = []
    for item in dataset:
        if isinstance(item, dict):
            wild_type_data = Data(
                x=torch.tensor(item['wild_type']['node_features'], dtype=torch.float),
                edge_index=torch.tensor(item['wild_type']['edge_index'], dtype=torch.long),
                edge_attr=torch.tensor(item['wild_type']['edge_features'], dtype=torch.float)
            )
            mutant_data = Data(
                x=torch.tensor(item['mutant']['node_features'], dtype=torch.float),
                edge_index=torch.tensor(item['mutant']['edge_index'], dtype=torch.long),
                edge_attr=torch.tensor(item['mutant']['edge_features'], dtype=torch.float)
            )
            ddg = torch.tensor(item['ddg'], dtype=torch.float)
            mutant_name = item['mutant_name']
            processed_data.append((wild_type_data, mutant_data, ddg, mutant_name))
    
    return processed_data

# Load the test data
test_data = load_and_preprocess_dataset('data_p553.pkl')

# Create DataLoader for the test data
batch_size = 16
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Predict and save results
model.eval()
predictions = []
true_ddg = []
result = []

with torch.no_grad():
    for wild_data, mutant_data, ddg, mutant_name in test_loader:
        wild_data = wild_data.to(device)
        mutant_data = mutant_data.to(device)
        ddg = ddg.to(device)
        output = model(wild_data, mutant_data)
        
        # Convert to numpy and append to lists
        pred_numpy = output.cpu().numpy()
        ddg_numpy = ddg.cpu().numpy()
        
        for i in range(len(mutant_name)):
            result.append((mutant_name[i], ddg_numpy[i], pred_numpy[i]))

# Create a DataFrame from the results
df = pd.DataFrame(result, columns=['Mutant Name', 'True DDG', 'Prediction'])

# Save the DataFrame to a CSV file
df.to_csv('predictions.csv', index=False)

print("Predictions saved to predictions.csv")

# Calculate PCC and RMSE
true_ddg_array = np.array([x[1] for x in result])
predictions_array = np.array([x[2] for x in result])

pcc, _ = pearsonr(true_ddg_array, predictions_array)
rmse = np.sqrt(mean_squared_error(true_ddg_array, predictions_array))

print(f'Test PCC: {pcc:.4f}')
print(f'Test RMSE: {rmse:.4f}')

