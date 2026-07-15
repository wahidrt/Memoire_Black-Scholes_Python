# Bibliothèque utilisée pour les calculs numériques et la création de tableaux
import numpy as np

# Bibliothèque utilisée pour tracer le graphique du Vega
import matplotlib.pyplot as plt

# Fonction de densité de la loi normale centrée réduite
from scipy.stats import norm


def calc_d1(S, K, T, r, sigma):
    """
    Calcule le terme d1 intervenant dans la formule de Black-Scholes.

    Paramètres
    ----------
    S : float
        Prix actuel du sous-jacent.
    K : float
        Prix d'exercice de l'option (strike).
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

    # Lorsque l'option arrive à échéance, la formule classique de d1
    # n'est plus définie à cause de la division par sqrt(T).
    if T <= 0:
        # À l'échéance, on distingue les cas S >= K et S < K.
        return np.inf if S >= K else -np.inf

    # Formule de d1 dans le modèle de Black-Scholes.
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (
        sigma * np.sqrt(T)
    )


def vega(S, K, T, r, sigma):
    """
    Calcule le Vega d'une option européenne dans le modèle de Black-Scholes.

    Le Vega mesure la sensibilité du prix de l'option à une variation de la
    volatilité. Sa formule est identique pour une option Call et une option Put.

    La valeur renvoyée correspond à une variation absolue de volatilité égale
    à 1, c'est-à-dire à 100 points de pourcentage. Pour obtenir l'effet d'une
    variation de 1 point de pourcentage, on divisera ensuite le résultat par 100.

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
        Vega théorique de l'option.
    """

    # À l'échéance, la volatilité n'a plus d'influence sur le prix de l'option.
    if T <= 0:
        return 0.0

    # Calcul du terme d1 nécessaire à la formule du Vega.
    d1 = calc_d1(S, K, T, r, sigma)

    # Formule du Vega : S * phi(d1) * sqrt(T),
    # où phi représente la densité de la loi normale centrée réduite.
    return S * norm.pdf(d1) * np.sqrt(T)


def tracer_graphique_vega():
    """
    Trace le Vega en fonction du prix du sous-jacent pour plusieurs maturités.

    Le graphique permet notamment d'observer que le Vega est généralement
    maximal lorsque l'option est proche de la monnaie, c'est-à-dire lorsque
    le prix du sous-jacent S est voisin du strike K. Il diminue également
    lorsque l'option se rapproche de son échéance.
    """

    # Paramètres du modèle de Black-Scholes.
    K = 100.0       # Prix d'exercice de l'option
    r = 0.05        # Taux d'intérêt sans risque annuel : 5 %
    sigma = 0.20    # Volatilité annuelle : 20 %

    # Ensemble des prix du sous-jacent étudiés, de 50 à 150.
    S_values = np.linspace(50, 150, 200)

    # Maturités choisies pour illustrer l'évolution du Vega dans le temps.
    # Les valeurs sont exprimées en années.
    maturites = [1.0, 0.5, 0.25, 0.05]

    # Création d'une figure de largeur 10 pouces et de hauteur 6 pouces.
    plt.figure(figsize=(10, 6))

    # Une couleur différente est associée à chaque maturité.
    couleurs = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Calcul et tracé du Vega pour chacune des maturités.
    for i, T in enumerate(maturites):
        # La formule renvoie le Vega pour une variation de volatilité égale à 1.
        # On divise donc par 100 pour représenter l'effet d'une hausse de
        # volatilité de 1 point de pourcentage, par exemple de 20 % à 21 %.
        vegas = [vega(S, K, T, r, sigma) / 100 for S in S_values]

        # Tracé de la courbe correspondant à la maturité T.
        plt.plot(
            S_values,
            vegas,
            color=couleurs[i],
            label=f'Vega (T={T} an(s))',
            linewidth=2
        )

    # Ligne verticale indiquant la position du strike K = 100.
    # Elle permet de repérer la zone où l'option est à la monnaie.
    plt.axvline(K, color='black', linestyle=':', label='Strike (K=100)')

    # Mise en forme du graphique.
    plt.title('Vega en fonction du prix du sous-jacent (S)', fontsize=14)
    plt.xlabel('Prix du sous-jacent (S)', fontsize=12)
    plt.ylabel('Vega pour 1% de vol ($\mathcal{V} / 100$)', fontsize=12)

    # Ajout d'une grille légère et de la légende des différentes courbes.
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=10)

    # Ajustement automatique des marges pour éviter le chevauchement des textes.
    plt.tight_layout()

    # Affichage du graphique à l'écran.
    plt.show()


# Ce bloc est exécuté uniquement lorsque le fichier Vega.py est lancé directement.
# Il ne s'exécute pas lorsque le fichier est importé comme module dans un autre code.
if __name__ == "__main__":
    tracer_graphique_vega()
