import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_tabnet.tab_model import TabNetClassifier

# ==============================================================================
# MODELE ECG: CNN - BiLSTM - Attention
# ==============================================================================
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        # Attention weight computation
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_out):
        # lstm_out shape: [batch_size, seq_len, hidden_size * 2]
        attn_weights = F.softmax(self.attention(lstm_out), dim=1) # [batch, seq, 1]
        context = torch.sum(attn_weights * lstm_out, dim=1) # [batch, hidden_size * 2]
        return context, attn_weights

class ECGDeepModel(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super(ECGDeepModel, self).__init__()
        
        # 1D Convolutional Layers pour l'extraction de caractéristiques morphologiques
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # BiLSTM pour les dépendances temporelles
        # Après 2 MaxPool1d, la longueur de la séquence 187 devient 187/2/2 = 46
        self.bilstm = nn.LSTM(input_size=64, hidden_size=64, 
                              num_layers=1, batch_first=True, bidirectional=True)
        
        # Mécanisme d'attention temporel
        self.attention = TemporalAttention(hidden_size=64)
        
        # Classifieur avec Dropout de Monte Carlo (toujours activé pour incertitude)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: [batch, 1, 187]
        x = self.pool(F.relu(self.conv1(x))) # [batch, 32, 93]
        x = self.pool(F.relu(self.conv2(x))) # [batch, 64, 46]
        
        # Permutation pour LSTM: [batch, seq_len, features]
        x = x.permute(0, 2, 1) 
        
        lstm_out, _ = self.bilstm(x) # lstm_out: [batch, seq_len, 128]
        
        # Attention
        context, attn_weights = self.attention(lstm_out)
        
        # Pour le MC Dropout, on veut qu'il soit actif même en phase d'inférence
        # donc on utilise F.dropout(training=True) explicitement dans la fonction MC_inference
        # mais ici on utilise la couche standard.
        x = self.dropout(F.relu(self.fc1(context)))
        x = self.fc2(x)
        
        return x, attn_weights

    def mc_forward(self, x, num_samples=10):
        """ Monte Carlo Dropout pour l'estimation de l'incertitude """
        # Force l'activation du dropout
        self.train() 
        outputs = []
        with torch.no_grad():
            for _ in range(num_samples):
                out, _ = self.forward(x)
                outputs.append(F.softmax(out, dim=1))
                
        outputs = torch.stack(outputs)
        mean_prob = outputs.mean(dim=0)
        variance = outputs.var(dim=0).mean(dim=1) # Mesure de l'incertitude épistémique
        
        return mean_prob, variance


# ==============================================================================
# MODELE TABULAIRE: TabNet (Deep Learning interprétable)
# ==============================================================================
def get_tabnet_model():
    """ Initialise et retourne un modèle TabNet Classifier """
    clf = TabNetClassifier(
        n_d=16, n_a=16, n_steps=4,
        gamma=1.3, n_independent=2, n_shared=2,
        momentum=0.02, clip_value=2., lambda_sparse=1e-3,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        # On utilise StepLR à la place de ReduceLROnPlateau pour éviter
        # un bug de compatibilité entre pytorch-tabnet et PyTorch récent
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params=dict(step_size=10, gamma=0.9),
        verbose=1
    )
    return clf

