import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from model import ThermoAGTGA
from data_loader import prepare_data

# Load data
train_loader, val_loader = prepare_data('data_s2648.pkl')

# Training Loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ThermoAGTGA().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=20, min_lr=0.00001, threshold=0.0002)
criterion = torch.nn.L1Loss()
best_val_loss = float('inf')
best_model_path = 'best_model_thermo_seq.pth'

def train():
    model.train()
    total_loss = 0
    for wild_data, mutant_data, ddg in train_loader:
        wild_data = wild_data.to(device)
        mutant_data = mutant_data.to(device)
        ddg = ddg.to(device)

        optimizer.zero_grad()
        output = model(wild_data, mutant_data)
        loss = criterion(output, ddg)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

def evaluate(loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for wild_data, mutant_data, ddg in loader:
            wild_data = wild_data.to(device)
            mutant_data = mutant_data.to(device)
            ddg = ddg.to(device)

            output = model(wild_data, mutant_data)
            loss = criterion(output, ddg)
            total_loss += loss.item()

    return total_loss / len(loader)

# Training the model
num_epochs = 1000
for epoch in range(num_epochs):
    start_time = time.time()

    train_loss = train()
    val_loss = evaluate(val_loader)
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    end_time = time.time()
    epoch_duration = end_time - start_time

    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Learning_rate:{current_lr:.4f}, Time: {epoch_duration:.2f}s')

    # Checkpoint the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_val_loss,
        }, best_model_path)
        print(f'Best model saved at epoch {epoch+1} with validation loss: {best_val_loss:.4f}')

print("Training complete.")

