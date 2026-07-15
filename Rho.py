# Bibliothèque utilisée pour les calculs numériques
import numpy as np

# Bibliothèque utilisée pour tracer le graphique du Rho
import matplotlib.pyplot as plt

# Fonctions de la loi normale centrée réduite
from scipy.stats import norm


def calc_d2(S, K, T, r, sigma):
    """
    Calcule le terme d2 du modèle de Black-Scholes.

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
        Valeur du terme d2.
    """

    # À l'échéance, les formules de d1 et d2 ne sont plus définies
    # à cause de la division par sqrt(T).
    if T <= 0:
        return np.inf if S >= K else -np.inf

    # Calcul intermédiaire de d1.
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (
        sigma * np.sqrt(T)
    )

    # Relation entre d1 et d2 dans le modèle de Black-Scholes.
    return d1 - sigma * np.sqrt(T)


def rho_call(S, K, T, r, sigma):
    """
    Calcule le Rho d'une option Call européenne.

    Le Rho mesure la sensibilité du prix de l'option à une variation du taux
    d'intérêt sans risque. Pour un Call européen, il est généralement positif :
    une hausse des taux tend à augmenter la valeur du Call.

    La valeur renvoyée correspond à une variation absolue du taux égale à 1,
    c'est-à-dire à 100 points de pourcentage.
    """

    # À l'échéance, une variation du taux n'a plus d'effet sur l'option.
    if T <= 0:
        return 0.0

    # Calcul du terme d2 nécessaire à la formule du Rho.
    d2 = calc_d2(S, K, T, r, sigma)

    # Formule du Rho d'un Call européen.
    return K * T * np.exp(-r * T) * norm.cdf(d2)


def rho_put(S, K, T, r, sigma):
    """
    Calcule le Rho d'une option Put européenne.

    Pour un Put européen, le Rho est généralement négatif : une hausse des
    taux tend à diminuer la valeur actuelle du prix d'exercice et donc la
    valeur du Put.

    La valeur renvoyée correspond à une variation absolue du taux égale à 1,
    c'est-à-dire à 100 points de pourcentage.
    """

    # À l'échéance, une variation du taux n'a plus d'effet sur l'option.
    if T <= 0:
        return 0.0

    # Calcul du terme d2 nécessaire à la formule du Rho.
    d2 = calc_d2(S, K, T, r, sigma)

    # Formule du Rho d'un Put européen.
    return -K * T * np.exp(-r * T) * norm.cdf(-d2)


def tracer_graphique_rho():
    """
    Trace le Rho des Calls et des Puts en fonction du prix du sous-jacent.

    Plusieurs maturités sont représentées afin de montrer que la sensibilité
    aux taux d'intérêt est généralement plus forte lorsque l'échéance est
    éloignée.
    """

    # Paramètres du marché et de l'option.
    K = 100.0       # Prix d'exercice de l'option
    r = 0.05        # Taux d'intérêt sans risque annuel : 5 %
    sigma = 0.20    # Volatilité annuelle : 20 %

    # Ensemble des prix du sous-jacent étudiés, de 50 à 150.
    S_values = np.linspace(50, 150, 200)

    # Différentes maturités, exprimées en années.
    maturites = [1.0, 0.5, 0.1]

    # Création de la figure.
    plt.figure(figsize=(12, 7))

    # Une couleur différente est associée à chaque maturité.
    couleurs = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Tracé du Rho des Calls.
    for i, T in enumerate(maturites):
        # Les fonctions rho_call et rho_put renvoient la sensibilité pour une
        # variation absolue du taux égale à 1. La division par 100 permet donc
        # d'obtenir l'effet d'une hausse de 1 point de pourcentage, par exemple
        # un passage du taux de 5 % à 6 %.
        rhos_c = [rho_call(S, K, T, r, sigma) / 100 for S in S_values]

        plt.plot(
            S_values,
            rhos_c,
            color=couleurs[i],
            label=f'Call (T={T} an(s))',
            linewidth=2
        )

    # Tracé du Rho des Puts avec des courbes en pointillés.
    for i, T in enumerate(maturites):
        rhos_p = [rho_put(S, K, T, r, sigma) / 100 for S in S_values]

        plt.plot(
            S_values,
            rhos_p,
            color=couleurs[i],
            linestyle='--',
            label=f'Put (T={T} an(s))',
            linewidth=2
        )

    # Ligne verticale indiquant le strike et ligne horizontale indiquant Rho = 0.
    plt.axvline(K, color='black', linestyle=':', label='Strike (K=100)')
    plt.axhline(0, color='black', linewidth=1)

    # Mise en forme du graphique.
    plt.title(
        'Rho des options Call et Put en fonction du prix du sous-jacent (S)',
        fontsize=14
    )
    plt.xlabel('Prix du sous-jacent (S)', fontsize=12)
    plt.ylabel('Rho pour 1% de variation du taux ($\rho / 100$)', fontsize=12)
    plt.grid(True, alpha=0.3)

    # La légende est placée à l'extérieur pour ne pas masquer les courbes.
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)

    # Ajustement automatique des marges.
    plt.tight_layout()

    # Affichage du graphique.
    plt.show()


# Ce bloc s'exécute uniquement lorsque Rho.py est lancé directement.
if __name__ == "__main__":
    tracer_graphique_rho()
