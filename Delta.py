import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def calc_d1(S, K, T, r, sigma):
    """Calcule le paramètre d1 du modèle de Black-Scholes."""
    # Sécurité pour éviter la division par zéro à l'échéance (T=0)
    if T <= 0:
        return np.inf if S >= K else -np.inf
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def delta_call(S, K, T, r, sigma):
    """Calcule le Delta d'une option Call européenne."""
    if T <= 0: return 1.0 if S >= K else 0.0
    d1 = calc_d1(S, K, T, r, sigma)
    return norm.cdf(d1)

def delta_put(S, K, T, r, sigma):
    """Calcule le Delta d'une option Put européenne."""
    if T <= 0: return 0.0 if S >= K else -1.0
    d1 = calc_d1(S, K, T, r, sigma)
    return norm.cdf(d1) - 1

def tracer_graphique_delta():
    """
    Génère un graphique académique du Delta pour des Calls et des Puts
    en fonction du prix du sous-jacent, pour différentes maturités.
    """
    # Paramètres du marché et de l'option
    K = 100.0       # Strike
    r = 0.05        # Taux sans risque (5%)
    sigma = 0.20    # Volatilité (20%)
    
    # Plage de prix du sous-jacent (de 50 à 150)
    S_values = np.linspace(50, 150, 200)
    
    # Différentes maturités à observer (en années)
    maturites = [1.0, 0.5, 0.1, 0.01] 
    
    # Configuration de la figure
    plt.figure(figsize=(12, 7))
    
    # Couleurs pour différencier les maturités
    couleurs = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Tracé des courbes pour le Call
    for i, T in enumerate(maturites):
        deltas_c = [delta_call(S, K, T, r, sigma) for S in S_values]
        plt.plot(S_values, deltas_c, color=couleurs[i], 
                 label=f'Call (T={T} an(s))', linewidth=2)

    # Tracé des courbes pour le Put
    for i, T in enumerate(maturites):
        deltas_p = [delta_put(S, K, T, r, sigma) for S in S_values]
        plt.plot(S_values, deltas_p, color=couleurs[i], linestyle='--', 
                 label=f'Put (T={T} an(s))', linewidth=2)

    # Lignes de repère (Strike et l'axe horizontal 0)
    plt.axvline(K, color='black', linestyle=':', label='Strike (K=100)')
    plt.axhline(0, color='black', linewidth=1)

    # Habillage du graphique (idéal pour un document LaTeX/PDF)
    plt.title('Delta des options Call et Put en fonction du prix du sous-jacent (S)', fontsize=14)
    plt.xlabel('Prix du sous-jacent (S)', fontsize=12)
    plt.ylabel('Delta ($\Delta$)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Placement de la légende à l'extérieur pour ne pas surcharger les courbes
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
    
    plt.tight_layout() # Ajuste les marges pour inclure la légende
    
    # Affichage (ou plt.savefig('delta_graphe.png', dpi=300) pour l'exporter)
    plt.show()

# --- Exécution ---
if __name__ == "__main__":
    tracer_graphique_delta()
