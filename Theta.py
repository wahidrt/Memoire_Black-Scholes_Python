import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def calc_d1_d2(S, K, T, r, sigma):
    """Calcule d1 et d2 pour le modèle de Black-Scholes."""
    if T <= 0:
        return (np.inf, np.inf) if S >= K else (-np.inf, -np.inf)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def theta_call(S, K, T, r, sigma):
    """Calcule le Theta annuel d'une option Call."""
    if T <= 0: return 0.0
    d1, d2 = calc_d1_d2(S, K, T, r, sigma)
    terme1 = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    terme2 = r * K * np.exp(-r * T) * norm.cdf(d2)
    return terme1 - terme2

def theta_put(S, K, T, r, sigma):
    """Calcule le Theta annuel d'une option Put."""
    if T <= 0: return 0.0
    d1, d2 = calc_d1_d2(S, K, T, r, sigma)
    terme1 = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    terme2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
    return terme1 + terme2

def tracer_graphique_theta():
    """
    Génère le graphique du Theta (pour un Call) en fonction du sous-jacent.
    Illustre l'accélération de l'érosion temporelle à la monnaie
    lorsque l'échéance approche.
    """
    K = 100.0
    r = 0.05
    sigma = 0.20
    S_values = np.linspace(50, 150, 200)
    
    maturites = [1.0, 0.5, 0.25, 0.05] 
    
    plt.figure(figsize=(10, 6))
    couleurs = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, T in enumerate(maturites):
        # On divise par 365 pour obtenir le Theta journalier (1 jour de passage)
        thetas_c = [theta_call(S, K, T, r, sigma) / 365 for S in S_values]
        plt.plot(S_values, thetas_c, color=couleurs[i], 
                 label=f'Theta Call (T={T} an(s))', linewidth=2)

    plt.axvline(K, color='black', linestyle=':', label='Strike (K=100)')
    plt.axhline(0, color='black', linewidth=1)

    plt.title('Theta (journalier) d\'un Call en fonction du prix du sous-jacent (S)', fontsize=14)
    plt.xlabel('Prix du sous-jacent (S)', fontsize=12)
    plt.ylabel('Theta pour 1 jour écoulé ($\Theta / 365$)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Légende placée en bas pour ne pas cacher le "puits" à la monnaie
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    
    plt.show()

# --- Exécution ---
if __name__ == "__main__":
    tracer_graphique_theta()
