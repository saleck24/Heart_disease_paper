import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut

# ==============================================================================
# DATASET: Heart Disease (Données Tabulaires)
# ==============================================================================
class HeartDiseaseDataset(Dataset):
    def __init__(self, data_path, is_train=True, fold=0, n_splits=5):
        """
        Charge les données tabulaires Heart Disease.
        Implémente la validation croisée Stratified K-Fold.
        """
        df = pd.read_csv(data_path)
        
        # La cible est généralement la colonne 'target'
        y = df['target'].values
        X = df.drop('target', axis=1).values
        
        # Normalisation simple
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        # Validation croisée
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(skf.split(X, y))
        
        train_idx, test_idx = splits[fold]
        
        if is_train:
            self.X = X[train_idx]
            self.y = y[train_idx]
        else:
            self.X = X[test_idx]
            self.y = y[test_idx]
            
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.LongTensor([self.y[idx]])[0]


# ==============================================================================
# DATASET: ECG (MIT-BIH et PTBDB) - Séries Temporelles
# ==============================================================================
class ECGDataset(Dataset):
    def __init__(self, data_path, is_train=True):
        """
        Charge les signaux ECG (MIT-BIH ou PTBDB) téléchargés depuis Kaggle.
        Chaque ligne du CSV est un battement (187 points) et la dernière colonne est le label.
        """
        df = pd.read_csv(data_path, header=None)
        
        # Séparation signal et label (dernière colonne)
        data = df.values
        X = data[:, :-1]
        y = data[:, -1]
        
        # Ajout d'une dimension pour le canal (nécessaire pour CNN 1D: [batch, channel, length])
        self.X = X.reshape(X.shape[0], 1, X.shape[1])
        self.y = y
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.LongTensor([self.y[idx]])[0]

def get_dataloaders(dataset, batch_size=64):
    """Crée les DataLoaders pour PyTorch"""
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

# ==============================================================================
# DATASET: ECG (MIT-BIH brut depuis PhysioNet) - Séries Temporelles (LOPO-CV)
# ==============================================================================
class RawECGDataset(Dataset):
    def __init__(self, processed_dir='data/mitdb/processed', is_train=True, test_patient='100'):
        """
        Charge les signaux ECG bruts extraits via wfdb.
        Sépare les données en utilisant l'ID du patient (Leave-One-Patient-Out).
        """
        X = np.load(f"{processed_dir}/X.npy")
        y = np.load(f"{processed_dir}/y.npy")
        groups = np.load(f"{processed_dir}/groups.npy")
        
        # Séparation en fonction du test_patient
        if is_train:
            mask = (groups != test_patient)
        else:
            mask = (groups == test_patient)
            
        X_split = X[mask]
        y_split = y[mask]
        
        # Ajout d'une dimension pour le canal
        self.X = X_split.reshape(X_split.shape[0], 1, X_split.shape[1])
        self.y = y_split
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.LongTensor([self.y[idx]])[0]
