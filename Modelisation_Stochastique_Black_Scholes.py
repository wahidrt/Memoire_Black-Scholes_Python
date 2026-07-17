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

    return (
        liste_N,
        np.asarray(prix_estimations),
        np.asarray(ic_bas),
        np.asarray(ic_haut),
        prix_estimations[-1],
    )


# ==============================================================================
# 3. SCRIPT D'ÉVALUATION ET DE VISUALISATION VISUELLE
# ==============================================================================
if __name__ == "__main__":

    # --- Configuration des paramètres financiers (identiques à la partie EDP) ---
    S0 = 100.0      # Prix actuel de l'action (€)
    K = 100.0       # Strike (€)
    r = 0.05        # Taux sans risque (5 %)
    sigma = 0.20    # Volatilité (20 %)

    # Plusieurs maturités T, exprimées en années.
    # Tu peux modifier directement cette liste pour ajouter ou retirer des courbes.
    maturites = [0.25, 0.50, 0.75, 1.00]

    # --- Configuration de la simulation stochastique ---
    N_max = 60000   # Nombre maximal de simulations de trajectoires
    pas_eval = 500  # Fréquence de calcul pour tracer la courbe de convergence

    # --- Préparation du graphique ---
    plt.figure(figsize=(12, 7))

    print("\n" + "=" * 72)
    print("      COMPARAISON STOCHASTIQUE POUR PLUSIEURS MATURITÉS")
    print("=" * 72)
    print(f"{'T (année)':>10} | {'Prix exact':>12} | {'Prix Monte-Carlo':>17} | {'Erreur absolue':>15}")
    print("-" * 72)

    # Une courbe de convergence est tracée pour chaque maturité T.
    for T in maturites:
        prix_exact = bs_analytique_call(S0, K, T, r, sigma)
        liste_N, prix_mc, ic_bas, ic_haut, prix_final_mc = simulation_monte_carlo_call(
            S0, K, T, r, sigma, N_max, pas_eval
        )

        erreur_absolue = abs(prix_final_mc - prix_exact)
        print(
            f"{T:10.2f} | {prix_exact:10.4f} € | "
            f"{prix_final_mc:15.4f} € | {erreur_absolue:13.4f} €"
        )

        # Courbe de convergence Monte-Carlo pour la maturité T.
        courbe, = plt.plot(
            liste_N,
            prix_mc,
            linewidth=1.6,
            label=f"Monte-Carlo : T = {T:.2f} an",
        )
        couleur = courbe.get_color()

        # Prix analytique exact correspondant à la même maturité.
        plt.axhline(
            y=prix_exact,
            color=couleur,
            linestyle="--",
            linewidth=1.3,
            alpha=0.9,
            label=f"Prix exact : T = {T:.2f} an ({prix_exact:.3f} €)",
        )

        # Intervalle de confiance à 95 % autour de l'estimation Monte-Carlo.
        plt.fill_between(
            liste_N,
            ic_bas,
            ic_haut,
            color=couleur,
            alpha=0.06,
        )

    print("=" * 72 + "\n")

    # --- Habillage du graphique ---
    plt.title(
        "Convergence de l'estimateur de Monte-Carlo pour plusieurs maturités",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel("Nombre de simulations (N)", fontsize=11)
    plt.ylabel("Prix du Call européen (€)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="best", fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(
        "convergence_monte_carlo_plusieurs_maturites.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
