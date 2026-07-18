import torch
import shap
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# EXPLICABILITE TABULAIRE (SHAP sur TabNet)
# ==============================================================================
def explain_tabnet_shap(clf, X_test, feature_names=None):
    """
    Génère des explications SHAP pour le modèle TabNet.
    """
    print("Calcul des valeurs SHAP (cela peut prendre un peu de temps)...")
    # KernelExplainer est souvent nécessaire pour les modèles complexes comme TabNet
    # On utilise un échantillon de fond (background) pour réduire le temps de calcul
    background = X_test[:100] 
    explainer = shap.KernelExplainer(clf.predict_proba, background)
    
    # Explication sur un petit sous-ensemble de test
    shap_values = explainer.shap_values(X_test[:50])
    
    print("Affichage du Summary Plot SHAP...")
    shap.summary_plot(shap_values, X_test[:50], feature_names=feature_names)
    plt.show()

# ==============================================================================
# EXPLICABILITE ECG (Attention Visualisation)
# ==============================================================================
def visualize_ecg_attention(signal, attention_weights, title="ECG Attention Heatmap"):
    """
    Superpose les poids d'attention sur le signal ECG.
    
    signal: array 1D de longueur L (ex: 187)
    attention_weights: array 1D de la même longueur ou d'une longueur réduite
    """
    # Si la dimension de l'attention est plus petite (à cause du pooling), on l'interpole
    if len(attention_weights) != len(signal):
        # Interpolation simple (Nearest Neighbor) pour aligner sur 187 points
        attention_weights = np.repeat(attention_weights, len(signal) // len(attention_weights) + 1)[:len(signal)]
        
    # Normalisation pour l'affichage
    attention_weights = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min() + 1e-8)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Création du fond coloré (heatmap)
    extent = [0, len(signal), min(signal) - 0.1, max(signal) + 0.1]
    ax.imshow(attention_weights[np.newaxis, :], cmap='Reds', aspect='auto', alpha=0.5, extent=extent)
    
    # Tracé du signal ECG par-dessus
    ax.plot(signal, color='black', linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel('Temps')
    ax.set_ylabel('Amplitude')
    
    plt.tight_layout()
    plt.show()

def explain_ecg_prediction(model, dataset, idx, device='cpu'):
    """
    Sélectionne un signal, fait la prédiction et affiche l'attention.
    """
    model.to(device)
    model.eval()
    
    # Récupération du signal
    X, y = dataset[idx]
    X_batch = X.unsqueeze(0).to(device) # [1, 1, 187]
    
    with torch.no_grad():
        out, attn = model(X_batch)
        pred = torch.argmax(out, dim=1).item()
        
    attn = attn.squeeze().cpu().numpy()
    signal = X.squeeze().numpy()
    
    print(f"Véritable Classe: {y.item()} | Prédiction: {pred}")
    visualize_ecg_attention(signal, attn, title=f"Explication (Attention) - Pred: {pred}")
