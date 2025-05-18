import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001, device='cuda'):
    """
    Train the autoencoder and save the best model based on validation loss
    """
    model = model.to(device)  # Move model to device
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data in train_loader:
            inputs, _ = data
            inputs = inputs.to(device)  # Move inputs to device
            recon_batch, z, mu, logvar = model(inputs)
            loss = criterion(recon_batch, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, _ = data
                inputs = inputs.to(device)  # Move inputs to device
                recon_batch, z, mu, logvar = model(inputs)
                val_loss += criterion(recon_batch, inputs).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    return train_losses, val_losses