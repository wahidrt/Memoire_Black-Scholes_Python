# Bibliothèque utilisée pour les calculs numériques
import numpy as np

# Bibliothèque utilisée pour tracer le graphique du Gamma
import matplotlib.pyplot as plt

# Fonction de densité de la loi normale centrée réduite
from scipy.stats import norm


def calc_d1(S, K, T, r, sigma):
    """
    Calcule le terme d1 intervenant dans le modèle de Black-Scholes.

    Paramètres
    ----------
    S : float
        Prix actuel du sous-jacent.
    K : float
        Prix d'exercice de l'option.
    T : float
        Temps restant avant l'échéance, exprimé en années.
    r : float
        Taux d'intérêt sans risque annuel.
    sigma : float
        Volatilité annuelle du sous-jacent.

    Retour
    ------
    float
        Valeur du terme d1.
    """

    # À l'échéance, la formule de d1 n'est plus définie à cause de sqrt(T).
    if T <= 0:
        return np.inf if S >= K else -np.inf

    # Formule de d1 dans le modèle de Black-Scholes.
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (
        sigma * np.sqrt(T)
    )


def gamma(S, K, T, r, sigma):
    """
    Calcule le Gamma d'une option européenne.

    Le Gamma mesure la variation du Delta lorsque le prix du sous-jacent
    varie. Il correspond donc à la dérivée seconde du prix de l'option par
    rapport au prix du sous-jacent.

    Dans le modèle de Black-Scholes, le Gamma est identique pour un Call et
    un Put ayant les mêmes paramètres.
    """

    # À l'échéance, on renvoie 0 dans le calcul numérique. Théoriquement,
    # le Gamma devient très concentré autour de S = K lorsque T tend vers 0.
    if T <= 0:
        return 0.0

    # Calcul du terme d1 nécessaire à la formule du Gamma.
    d1 = calc_d1(S, K, T, r, sigma)

    # Formule du Gamma : phi(d1) / (S * sigma * sqrt(T)),
    # où phi représente la densité de la loi normale centrée réduite.
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def tracer_graphique_gamma():
    """
    Trace le Gamma en fonction du prix du sous-jacent pour plusieurs maturités.

    Le graphique met en évidence la concentration du Gamma autour du strike
    lorsque l'option est proche de la monnaie et que l'échéance approche.
    """

    # Paramètres du marché et de l'option.
    K = 100.0       # Prix d'exercice de l'option
    r = 0.05        # Taux d'intérêt sans risque annuel : 5 %
    sigma = 0.20    # Volatilité annuelle : 20 %

    # Ensemble des prix du sous-jacent étudiés, de 50 à 150.
    S_values = np.linspace(50, 150, 200)

    # Maturités choisies pour observer la formation progressive du pic de Gamma.
    maturites = [1.0, 0.5, 0.1, 0.02]

    # Création de la figure.
    plt.figure(figsize=(10, 6))

    # Une couleur différente est associée à chaque maturité.
    couleurs = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Calcul et tracé du Gamma pour chacune des maturités.
    for i, T in enumerate(maturites):
        gammas = [gamma(S, K, T, r, sigma) for S in S_values]
        plt.plot(
            S_values,
            gammas,
            color=couleurs[i],
            label=f'Gamma (T={T} an(s))',
            linewidth=2
        )

    # Ligne verticale indiquant le strike, c'est-à-dire la zone à la monnaie.
    plt.axvline(K, color='black', linestyle=':', label='Strike (K=100)')

    # Mise en forme du graphique.
    plt.title('Gamma en fonction du prix du sous-jacent (S)', fontsize=14)
    plt.xlabel('Prix du sous-jacent (S)', fontsize=12)
    plt.ylabel('Gamma ($\Gamma$)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=10)

    # Ajustement automatique des marges.
    plt.tight_layout()

    # Affichage du graphique.
    plt.show()


# Ce bloc s'exécute uniquement lorsque Gamma.py est lancé directement.
if __name__ == "__main__":
    tracer_graphique_gamma()
