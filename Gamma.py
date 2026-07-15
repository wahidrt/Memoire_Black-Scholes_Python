import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def calc_d1(S, K, T, r, sigma):
    """Calcule d1 pour le modèle de Black-Scholes."""
    if T <= 0:
        return np.inf if S >= K else -np.inf
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def gamma(S, K, T, r, sigma):
    """
    Calcule le Gamma (valable pour Call et Put).
    """
    if T <= 0: return 0.0 # À l'échéance, le Gamma est nul (sauf à S=K où il est infini)
    d1 = calc_d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def tracer_graphique_gamma():
    """
    Génère le graphique du Gamma en fonction du sous-jacent 
    pour illustrer la concentration du risque à la monnaie.
    """
    K = 100.0
    r = 0.05
    sigma = 0.20
    S_values = np.linspace(50, 150, 200)
    
    # On prend des maturités qui se rapprochent de 0 pour bien voir le "pic"
    maturites = [1.0, 0.5, 0.1, 0.02] 
    
    plt.figure(figsize=(10, 6))
    couleurs = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, T in enumerate(maturites):
        gammas = [gamma(S, K, T, r, sigma) for S in S_values]
        plt.plot(S_values, gammas, color=couleurs[i], 
                 label=f'Gamma (T={T} an(s))', linewidth=2)

    plt.axvline(K, color='black', linestyle=':', label='Strike (K=100)')

    plt.title('Gamma en fonction du prix du sous-jacent (S)', fontsize=14)
    plt.xlabel('Prix du sous-jacent (S)', fontsize=12)
    plt.ylabel('Gamma ($\Gamma$)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    
    plt.show()

# --- Exécution ---
if __name__ == "__main__":
    tracer_graphique_gamma()
