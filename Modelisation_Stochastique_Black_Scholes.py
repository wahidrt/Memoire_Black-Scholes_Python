import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt

# ==============================================================================
# 1. FORMULE ANALYTIQUE DE BLACK-SCHOLES (RÉFÉRENCE)
# ==============================================================================
def bs_analytique_call(S, K, T, r, sigma):
    """
    Calcule la valeur exacte d'un Call Européen via la formule classique.
    Sert de cible fixe pour vérifier la convergence de la méthode de Monte-Carlo.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)


# ==============================================================================
# 2. MOTEUR STOCHASTIQUE (SIMULATION DE MONTE-CARLO)
# ==============================================================================
def simulation_monte_carlo_call(S0, K, T, r, sigma, N_max, pas_eval):
    """
    Simule des trajectoires de prix d'actifs et évalue la convergence de Monte-Carlo.
    """
    # Fixer la graine aléatoire pour que les graphiques soient stables et reproductibles
    np.random.seed(42)
    
    # --------------------------------------------------------------------------
    # Étape 2.1 : Simulation globale des prix finaux (Mouvement Brownien)
    # --------------------------------------------------------------------------
    # On génère un vecteur géant de N_max variables normales standard Z ~ N(0,1)
    Z = np.random.standard_normal(N_max)
    
    # Solution forte de l'EDS de Black-Scholes appliquée à la maturité T
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    
    # --------------------------------------------------------------------------
    # Étape 2.2 : Calcul des Payoffs actualisés
    # --------------------------------------------------------------------------
    # Évaluation du gain à maturité pour chaque trajectoire, puis actualisation à t=0
    payoffs_actualises = np.exp(-r * T) * np.maximum(ST - K, 0)
    
    # Initialisation des listes pour stocker l'historique de la convergence
    liste_N = np.arange(pas_eval, N_max + 1, pas_eval)
    prix_estimations = []
    ic_haut = []
    ic_bas = []
    
    # --------------------------------------------------------------------------
    # Étape 2.3 : Calcul de la convergence et des intervalles de confiance
    # --------------------------------------------------------------------------
    for n in liste_N:
        # On isole l'échantillon des 'n' premières simulations
        payoffs_partiels = payoffs_actualises[:n]
        
        # Moyenne empirique (notre estimation du prix à l'étape 'n')
        moyenne = np.mean(payoffs_partiels)
        
        # Écart-type des payoffs (mesure de la dispersion/variance du modèle)
        ecart_type = np.std(payoffs_partiels)
        
        # Erreur standard de l'estimateur (Théorème Central Limite)
        erreur_standard = ecart_type / np.sqrt(n)
        
        # Enregistrement du prix et des bornes de l'IC à 95% (+/- 1.96 * erreur)
        prix_estimations.append(moyenne)
        ic_haut.append(moyenne + 1.96 * erreur_standard)
        ic_bas.append(moyenne - 1.96 * erreur_standard)
        
    return liste_N, prix_estimations, ic_bas, ic_haut, prix_estimations[-1]


# ==============================================================================
# 3. SCRIPT D'ÉVALUATION ET DE VISUALISATION VISUELLE
# ==============================================================================
if __name__ == "__main__":
    
    # --- Configuration des paramètres financiers (Identiques à la partie EDP) ---
    S0    = 100.0   # Prix actuel de l'action (€)
    K     = 100.0   # Strike (€)
    T     = 1.0     # Maturité (1 an)
    r     = 0.05    # Taux sans risque (5%)
    sigma = 0.20    # Volatilité (20%)
    
    # --- Configuration de la simulation stochastique ---
    N_max    = 60000   # Nombre maximal de simulations de trajectoires
    pas_eval = 500     # Fréquence de calcul pour tracer la courbe de convergence
    
    # --- Lancement des calculs ---
    prix_exact = bs_analytique_call(S0, K, T, r, sigma)
    liste_N, prix_mc, ic_bas, ic_haut, prix_final_mc = simulation_monte_carlo_call(
        S0, K, T, r, sigma, N_max, pas_eval
    )
    
    # --- Impression des résultats dans le terminal ---
    print("\n" + "="*40)
    print("    RÉSULTATS DE COMPARAISON STOCHASTIQUE")
    print("="*40)
    print(f"Prix Théorique Exact (Black-Scholes) : {prix_exact:.4f} €")
    print(f"Prix Numérique (Monte-Carlo)         : {prix_final_mc:.4f} €")
    print(f"Erreur Absolue de l'estimateur       : {abs(prix_final_mc - prix_exact):.4f} €")
    print("="*40 + "\n")
    
    # --- Génération graphique du graphique de convergence ---
    plt.figure(figsize=(11, 6))
    
    # 1. Tracé de la ligne de référence (La vérité mathématique)
    plt.axhline(y=prix_exact, color='red', linestyle='--', linewidth=2, 
                label=f'Prix Analytique Exact ({prix_exact:.3f} €)')
    
    # 2. Dessin de l'enveloppe de l'Intervalle de Confiance (Zone bleutée transparente)
    plt.fill_between(liste_N, ic_bas, ic_haut, color='blue', alpha=0.12, 
                     label='Intervalle de Confiance à 95% (TCL)')
    
    # 3. Tracé de l'évolution de notre estimation Monte-Carlo
    plt.plot(liste_N, prix_mc, color='navy', linewidth=1.5, 
             label='Estimation cumulative de Monte-Carlo')
    
    # Habillage textuel et cosmétique du graphique
    plt.title("Convergence de l'Estimateur de Monte-Carlo & Réduction de l'Erreur", 
              fontsize=12, fontweight='bold')
    plt.xlabel("Nombre de simulations (N)", fontsize=11)
    plt.ylabel("Prix de l'Option calculé (€)", fontsize=11)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', fontsize=10)
    
    # On restreint légèrement l'axe Y autour du vrai prix pour apprécier la stabilisation
    plt.ylim(prix_exact - 1.5, prix_exact + 1.5)
    
    plt.tight_layout()
    plt.savefig("convergence_monte_carlo.png", dpi=300, bbox_inches="tight")
    plt.show()