import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt

# ==============================================================================
# 1. MODÈLE DE BLACK-SCHOLES (RÉFÉRENCE À VOLATILITÉ CONSTANTE)
# ==============================================================================
def bs_analytique_call(S, K, T, r, sigma):
    """Calcule le prix exact sous l'hypothèse de volatilité constante."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)


# ==============================================================================
# 2. MOTEUR STOCHASTIQUE DE HESTON (MONTE-CARLO PAR TRONCATURE COMPLÈTE)
# ==============================================================================
def heston_monte_carlo_call(S0, v0, K, T, r, kappa, theta, xi, rho, M, N):
    """
    Simule les trajectoires conjointes de l'actif et de sa variance sous Heston.
    M : Nombre de chemins (Trajectoires)
    N : Nombre de pas de temps
    """
    dt = T / N
    
    # Initialisation des matrices de stockage pour les trajectoires
    S = np.zeros((M, N + 1))
    v = np.zeros((M, N + 1))
    
    # Conditions initiales à t = 0
    S[:, 0] = S0
    v[:, 0] = v0
    
    # Matrice de variance-covariance pour générer les deux Browniens corrélés de pas dt
    # Cov(dW_S, dW_v) = rho * dt
    matrice_cov = [[dt, rho * dt], 
                   [rho * dt, dt]]
    
    # Boucle temporelle (Euler-Maruyama pas à pas)
    for t in range(1, N + 1):
        # Génération simultanée des incréments Browniens corrélés pour les M trajectoires
        increments = np.random.multivariate_normal([0, 0], matrice_cov, size=M)
        dW_S = increments[:, 0]
        dW_v = increments[:, 1]
        
        # Schéma de Troncature Complète (Full Truncation) pour la stabilité numérique.
        # Si le processus de variance devient négatif à cause de la discrétisation,
        # on utilise 0 sous la racine pour éviter un crash mathématique.
        v_positif = np.maximum(v[:, t-1], 0)
        
        # Évolution stochastique de la variance (Processus CIR de Heston)
        v[:, t] = v[:, t-1] + kappa * (theta - v_positif) * dt + xi * np.sqrt(v_positif) * dW_v
        
        # Évolution stochastique du prix de l'action
        S[:, t] = S[:, t-1] * np.exp((r - 0.5 * v_positif) * dt + np.sqrt(v_positif) * dW_S)
        
    # Calcul du Payoff actualisé à la maturité pour chaque trajectoire
    payoffs_finaux = np.maximum(S[:, -1] - K, 0)
    prix_option = np.exp(-r * T) * np.mean(payoffs_finaux)
    
    return prix_option


# ==============================================================================
# 3. CONFRONTATION ET PRODUCTION DU GRAPHIQUE COMPARATIF
# ==============================================================================
if __name__ == "__main__":
    
    # --- Paramètres Communs ---
    S0, T, r = 100.0, 1.0, 0.05
    strikes = np.linspace(80, 120, 15)  # Gamme de strikes du Call
    
    # --- Paramètres Black-Scholes ---
    sigma_bs = 0.20  # Volatilité fixe à 20%
    
    # --- Paramètres Heston ---
    v0 = 0.04     # Variance initiale (v0 = 0.2^2, équivalent à 20% de volatilité)
    theta = 0.04  # Moyenne de long terme de la variance (20% au carré)
    kappa = 2.0   # Vitesse élevée de retour à la moyenne
    xi = 0.35     # Volatilité de la volatilité importante
    rho = -0.75   # Effet de levier négatif fort (Marché d'actions classique)
    
    print("Calcul des profils de prix en cours (Monte-Carlo Heston vs Analytique BS)...")
    
    # Génération des listes de prix pour chaque Strike
    prix_bs = [bs_analytique_call(S0, K, T, r, sigma_bs) for K in strikes]
    prix_heston = [heston_monte_carlo_call(S0, v0, K, T, r, kappa, theta, xi, rho, M=30000, N=100) for K in strikes]
    
    # --- Création du Graphique de Tendance ---
    plt.figure(figsize=(10, 6))
    
    plt.plot(strikes, prix_heston, 'o-', color='darkgreen', linewidth=2, label='Modèle de Heston (Volatilité Stochastique & Levier)')
    plt.plot(strikes, prix_bs, 's--', color='blue', linewidth=1.5, label='Modèle de Black-Scholes (Volatilité Constante $\sigma=20\%$)')
    
    # Habillage textuel
    plt.title("Impact de la Volatilité Stochastique : Heston vs Black-Scholes", fontsize=12, fontweight='bold')
    plt.xlabel("Prix d'exercice / Strike (K)", fontsize=11)
    plt.ylabel("Prix du Call (€)", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.show()