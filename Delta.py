# Bibliothèque utilisée pour les calculs numériques et la création de tableaux
import numpy as np

# Bibliothèque utilisée pour tracer les courbes du Delta
import matplotlib.pyplot as plt

# Fonctions de la loi normale centrée réduite
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

    # À l'échéance, la formule classique de d1 n'est plus définie,
    # car elle contient une division par sqrt(T).
    if T <= 0:
        return np.inf if S >= K else -np.inf

    # Formule de d1 dans le modèle de Black-Scholes.
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (
        sigma * np.sqrt(T)
    )


def delta_call(S, K, T, r, sigma):
    """
    Calcule le Delta d'une option Call européenne.

    Le Delta d'un Call mesure la variation approximative du prix de l'option
    lorsque le prix du sous-jacent augmente d'une unité. Il est compris entre
    0 et 1 dans le modèle de Black-Scholes sans dividendes.
    """

    # À l'échéance, le Delta du Call vaut 1 si l'option termine dans la monnaie
    # et 0 si elle termine hors de la monnaie.
    if T <= 0:
        return 1.0 if S >= K else 0.0

    # Calcul du terme d1 puis application de la fonction de répartition normale.
    d1 = calc_d1(S, K, T, r, sigma)
    return norm.cdf(d1)


def delta_put(S, K, T, r, sigma):
    """
    Calcule le Delta d'une option Put européenne.

    Le Delta d'un Put mesure la variation approximative du prix de l'option
    lorsque le prix du sous-jacent augmente d'une unité. Il est compris entre
    -1 et 0 dans le modèle de Black-Scholes sans dividendes.
    """

    # À l'échéance, le Delta du Put vaut -1 si l'option termine dans la monnaie
    # et 0 si elle termine hors de la monnaie.
    if T <= 0:
        return 0.0 if S >= K else -1.0

    # La relation Delta_put = N(d1) - 1 découle de la parité Call-Put.
    d1 = calc_d1(S, K, T, r, sigma)
    return norm.cdf(d1) - 1


def tracer_graphique_delta():
    """
    Trace le Delta des Calls et des Puts en fonction du prix du sous-jacent.

    Plusieurs maturités sont représentées afin d'observer la transition du
    Delta lorsque l'option se rapproche de son échéance.
    """

    # Paramètres du marché et de l'option.
    K = 100.0       # Prix d'exercice de l'option
    r = 0.05        # Taux d'intérêt sans risque annuel : 5 %
    sigma = 0.20    # Volatilité annuelle : 20 %

    # Ensemble des prix du sous-jacent étudiés, de 50 à 150.
    S_values = np.linspace(50, 150, 200)

    # Différentes maturités, exprimées en années.
    maturites = [1.0, 0.5, 0.1, 0.01]

    # Création de la figure.
    plt.figure(figsize=(12, 7))

    # Une couleur différente est associée à chaque maturité.
    couleurs = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Tracé du Delta des Calls.
    for i, T in enumerate(maturites):
        deltas_c = [delta_call(S, K, T, r, sigma) for S in S_values]
        plt.plot(
            S_values,
            deltas_c,
            color=couleurs[i],
            label=f'Call (T={T} an(s))',
            linewidth=2
        )

    # Tracé du Delta des Puts avec des courbes en pointillés.
    for i, T in enumerate(maturites):
        deltas_p = [delta_put(S, K, T, r, sigma) for S in S_values]
        plt.plot(
            S_values,
            deltas_p,
            color=couleurs[i],
            linestyle='--',
            label=f'Put (T={T} an(s))',
            linewidth=2
        )

    # Ligne verticale indiquant le strike et ligne horizontale indiquant Delta = 0.
    plt.axvline(K, color='black', linestyle=':', label='Strike (K=100)')
    plt.axhline(0, color='black', linewidth=1)

    # Mise en forme du graphique.
    plt.title(
        'Delta des options Call et Put en fonction du prix du sous-jacent (S)',
        fontsize=14
    )
    plt.xlabel('Prix du sous-jacent (S)', fontsize=12)
    plt.ylabel('Delta ($\Delta$)', fontsize=12)
    plt.grid(True, alpha=0.3)

    # La légende est placée à l'extérieur pour ne pas masquer les courbes.
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)

    # Ajustement automatique des marges.
    plt.tight_layout()

    # Affichage du graphique.
    plt.show()


# Ce bloc s'exécute uniquement lorsque Delta.py est lancé directement.
if __name__ == "__main__":
    tracer_graphique_delta()
