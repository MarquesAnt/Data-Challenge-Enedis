import torch
import numpy as np  
from tqdm import tqdm
from models.loss import MaskedMSELoss

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    first_batch = True
    
    if first_batch:
        print(f" Training on device: {device}")
        print(f"  model device: {next(model.parameters()).device}")
        first_batch = False
    
    for x, mask, y in tqdm(loader, desc="Training", leave=False):
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        pred = model(x, mask)
        loss = criterion(pred, y, mask)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def validate(model, loader, criterion, device, scaler):
    model.eval()
    total_loss = 0
    total_mae = 0
    
    with torch.no_grad():
        for x, mask, y in loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            
            pred = model(x, mask)
            loss = criterion(pred, y, mask)
            
            # Calculer MAE sur les vraies valeurs (dénormalisées)
            pred_denorm = scaler.inverse_transform(pred.cpu().numpy().reshape(-1, 1))
            y_denorm = scaler.inverse_transform(y.cpu().numpy().reshape(-1, 1))
            mask_cpu = mask.cpu().numpy().reshape(-1, 1)
            
            # MAE seulement sur les valeurs manquantes
            missing_mask = (1 - mask_cpu).astype(bool).flatten()
            if missing_mask.sum() > 0:
                mae = np.abs(pred_denorm.flatten()[missing_mask] - 
                           y_denorm.flatten()[missing_mask]).mean()
                total_mae += mae
            
            total_loss += loss.item()
    
    return total_loss / len(loader), total_mae / len(loader)

def train_model(model, train_loader, val_loader, config, scaler):
    """Entraînement avec early stopping"""
    
    criterion = MaskedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device)
        
        # Validate
        val_loss, val_mae = validate(model, val_loader, criterion, config.device, scaler)
        
        # Scheduler
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.2f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Sauvegarder le meilleur modèle
            torch.save(model.state_dict(), 'best_bilstm_model.pt')
            print("  → New best model saved!")
        else:
            patience_counter += 1
            
        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Charger le meilleur modèle
    model.load_state_dict(torch.load('best_bilstm_model.pt'))
    return model