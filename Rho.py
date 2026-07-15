import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def calc_d2(S, K, T, r, sigma):
    """Calcule uniquement d2, car le Rho n'utilise pas directement d1."""
    if T <= 0:
        return np.inf if S >= K else -np.inf
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1 - sigma * np.sqrt(T)

def rho_call(S, K, T, r, sigma):
    """Calcule le Rho d'une option Call."""
    if T <= 0: return 0.0
    d2 = calc_d2(S, K, T, r, sigma)
    return K * T * np.exp(-r * T) * norm.cdf(d2)

def rho_put(S, K, T, r, sigma):
    """Calcule le Rho d'une option Put."""
    if T <= 0: return 0.0
    d2 = calc_d2(S, K, T, r, sigma)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2)

def tracer_graphique_rho():
    """
    Génère un graphique illustrant le Rho pour un Call et un Put.
    Met en évidence que la sensibilité est maximale dans la monnaie (ITM).
    """
    K = 100.0
    r = 0.05
    sigma = 0.20
    S_values = np.linspace(50, 150, 200)
    
    # Maturités pour observer l'effet du temps (le Rho augmente avec T)
    maturites = [1.0, 0.5, 0.1] 
    
    plt.figure(figsize=(12, 7))
    couleurs = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Tracé pour le Call
    for i, T in enumerate(maturites):
        # Division par 100 pour obtenir l'effet d'une hausse de 1% des taux
        rhos_c = [rho_call(S, K, T, r, sigma) / 100 for S in S_values]
        plt.plot(S_values, rhos_c, color=couleurs[i], 
                 label=f'Call (T={T} an(s))', linewidth=2)

    # Tracé pour le Put
    for i, T in enumerate(maturites):
        rhos_p = [rho_put(S, K, T, r, sigma) / 100 for S in S_values]
        plt.plot(S_values, rhos_p, color=couleurs[i], linestyle='--', 
                 label=f'Put (T={T} an(s))', linewidth=2)

    plt.axvline(K, color='black', linestyle=':', label='Strike (K=100)')
    plt.axhline(0, color='black', linewidth=1)

    plt.title('Rho des options Call et Put en fonction du prix du sous-jacent (S)', fontsize=14)
    plt.xlabel('Prix du sous-jacent (S)', fontsize=12)
    plt.ylabel('Rho pour 1% de variation du taux ($\\rho / 100$)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
    plt.tight_layout()
    
    plt.show()

# --- Exécution ---
if __name__ == "__main__":
    tracer_graphique_rho()
