import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def train_ecg_model(model, train_loader, val_loader, epochs=10, lr=0.001, device='cpu'):
    """
    Entraîne le modèle ECG (CNN-BiLSTM-Attention).
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        # Phase d'entraînement
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += y_batch.size(0)
            correct_train += (predicted == y_batch).sum().item()
            
        train_acc = 100 * correct_train / total_train
        
        # Phase de validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs, _ = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += y_batch.size(0)
                correct_val += (predicted == y_batch).sum().item()
                
        val_acc = 100 * correct_val / total_val
        
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {train_loss/len(train_loader):.4f} - Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss/len(val_loader):.4f} - Acc: {val_acc:.2f}%")
              
    return model

def evaluate_with_uncertainty(model, test_loader, num_mc_samples=10, device='cpu'):
    """
    Évalue le modèle en utilisant Monte Carlo Dropout pour obtenir l'incertitude.
    """
    model.to(device)
    all_means = []
    all_variances = []
    all_targets = []
    all_preds = []
    
    print("Évaluation avec Monte Carlo Dropout (calcul de l'incertitude)...")
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        
        mean_prob, variance = model.mc_forward(X_batch, num_samples=num_mc_samples)
        
        preds = torch.argmax(mean_prob, dim=1)
        
        all_means.append(mean_prob.cpu().numpy())
        all_variances.append(variance.cpu().numpy())
        all_targets.append(y_batch.numpy())
        all_preds.append(preds.cpu().numpy())
        
    return np.concatenate(all_preds), np.concatenate(all_targets), np.concatenate(all_variances)

# ==============================================================================
# ENTRAINEMENT TABNET
# ==============================================================================
def train_tabnet(clf, X_train, y_train, X_valid, y_valid, max_epochs=50):
    """ Entraîne le modèle TabNet """
    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_name=['train', 'valid'],
        eval_metric=['accuracy'],
        max_epochs=max_epochs,
        patience=10,
        batch_size=256, virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )
    return clf
