import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def calc_d1(S, K, T, r, sigma):
    """Calcule d1 pour le modèle de Black-Scholes."""
    if T <= 0:
        return np.inf if S >= K else -np.inf
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def vega(S, K, T, r, sigma):
    """
    Calcule le Vega (valable pour Call et Put).
    Renvoie la variation théorique pour 100% de variation de la volatilité.
    """
    if T <= 0: return 0.0 # À l'échéance, la volatilité n'a plus d'impact
    d1 = calc_d1(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)

def tracer_graphique_vega():
    """
    Génère le graphique du Vega en fonction du sous-jacent.
    Illustre la baisse du Vega à l'approche de la maturité.
    """
    K = 100.0
    r = 0.05
    sigma = 0.20
    S_values = np.linspace(50, 150, 200)
    
    # Maturités pour illustrer la perte de sensibilité avec le temps
    maturites = [1.0, 0.5, 0.25, 0.05] 
    
    plt.figure(figsize=(10, 6))
    couleurs = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, T in enumerate(maturites):
        # En pratique, on divise par 100 pour que l'axe Y représente 
        # l'impact en monnaie d'une hausse de 1% (100 bps) de volatilité.
        vegas = [vega(S, K, T, r, sigma) / 100 for S in S_values]
        plt.plot(S_values, vegas, color=couleurs[i], 
                 label=f'Vega (T={T} an(s))', linewidth=2)

    plt.axvline(K, color='black', linestyle=':', label='Strike (K=100)')

    plt.title('Vega en fonction du prix du sous-jacent (S)', fontsize=14)
    plt.xlabel('Prix du sous-jacent (S)', fontsize=12)
    plt.ylabel('Vega pour 1% de vol ($\mathcal{V} / 100$)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    
    plt.show()

# --- Exécution ---
if __name__ == "__main__":
    tracer_graphique_vega()
