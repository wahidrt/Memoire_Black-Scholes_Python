# Bibliothèque utilisée pour les calculs numériques
import numpy as np

# Bibliothèque utilisée pour tracer le graphique du Theta
import matplotlib.pyplot as plt

# Fonctions de la loi normale centrée réduite
from scipy.stats import norm


def calc_d1_d2(S, K, T, r, sigma):
    """
    Calcule les termes d1 et d2 du modèle de Black-Scholes.

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
    tuple
        Couple contenant les valeurs de d1 et d2.
    """

    # À l'échéance, les formules classiques de d1 et d2 ne sont plus définies
    # à cause de la division par sqrt(T).
    if T <= 0:
        return (np.inf, np.inf) if S >= K else (-np.inf, -np.inf)

    # Calcul du terme d1.
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (
        sigma * np.sqrt(T)
    )

    # Dans le modèle de Black-Scholes, d2 s'obtient directement à partir de d1.
    d2 = d1 - sigma * np.sqrt(T)

    return d1, d2


def theta_call(S, K, T, r, sigma):
    """
    Calcule le Theta annuel d'une option Call européenne.

    Le Theta mesure la variation du prix de l'option provoquée par le passage
    du temps, toutes choses égales par ailleurs. Pour une position acheteuse
    sur un Call, il est généralement négatif : la valeur temps de l'option
    diminue lorsque l'échéance approche.
    """

    # À l'échéance, il n'existe plus de valeur temps à éroder.
    if T <= 0:
        return 0.0

    # Calcul des termes d1 et d2 nécessaires à la formule du Theta.
    d1, d2 = calc_d1_d2(S, K, T, r, sigma)

    # Premier terme : contribution liée à la volatilité et à l'érosion du temps.
    terme1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

    # Deuxième terme : contribution liée à l'actualisation du strike.
    terme2 = r * K * np.exp(-r * T) * norm.cdf(d2)

    # Formule du Theta annuel d'un Call européen.
    return terme1 - terme2


def theta_put(S, K, T, r, sigma):
    """
    Calcule le Theta annuel d'une option Put européenne.

    Le Theta mesure la variation du prix du Put lorsque le temps restant avant
    l'échéance diminue. Son signe dépend notamment du prix du sous-jacent,
    du taux d'intérêt et de la position du Put par rapport au strike.
    """

    # À l'échéance, il n'existe plus de valeur temps à éroder.
    if T <= 0:
        return 0.0

    # Calcul des termes d1 et d2 nécessaires à la formule du Theta.
    d1, d2 = calc_d1_d2(S, K, T, r, sigma)

    # Premier terme commun aux Calls et aux Puts.
    terme1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

    # Deuxième terme propre au Put, utilisant la fonction N(-d2).
    terme2 = r * K * np.exp(-r * T) * norm.cdf(-d2)

    # Formule du Theta annuel d'un Put européen.
    return terme1 + terme2


def tracer_graphique_theta():
    """
    Trace le Theta journalier d'un Call en fonction du prix du sous-jacent.

    Plusieurs maturités sont représentées afin d'illustrer l'accélération de
    l'érosion temporelle, notamment lorsque l'option est proche de la monnaie
    et que l'échéance approche.
    """

    # Paramètres du marché et de l'option.
    K = 100.0       # Prix d'exercice de l'option
    r = 0.05        # Taux d'intérêt sans risque annuel : 5 %
    sigma = 0.20    # Volatilité annuelle : 20 %

    # Ensemble des prix du sous-jacent étudiés, de 50 à 150.
    S_values = np.linspace(50, 150, 200)

    # Différentes maturités, exprimées en années.
    maturites = [1.0, 0.5, 0.25, 0.05]

    # Création de la figure.
    plt.figure(figsize=(10, 6))

    # Une couleur différente est associée à chaque maturité.
    couleurs = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Calcul et tracé du Theta pour chacune des maturités.
    for i, T in enumerate(maturites):
        # La fonction theta_call renvoie un Theta annuel.
        # La division par 365 donne une approximation du Theta journalier,
        # c'est-à-dire l'effet théorique du passage d'une journée.
        thetas_c = [theta_call(S, K, T, r, sigma) / 365 for S in S_values]

        plt.plot(
            S_values,
            thetas_c,
            color=couleurs[i],
            label=f'Theta Call (T={T} an(s))',
            linewidth=2
        )

    # Ligne verticale indiquant le strike et ligne horizontale indiquant Theta = 0.
    plt.axvline(K, color='black', linestyle=':', label='Strike (K=100)')
    plt.axhline(0, color='black', linewidth=1)

    # Mise en forme du graphique.
    plt.title(
        'Theta (journalier) d\'un Call en fonction du prix du sous-jacent (S)',
        fontsize=14
    )
    plt.xlabel('Prix du sous-jacent (S)', fontsize=12)
    plt.ylabel('Theta pour 1 jour écoulé ($\Theta / 365$)', fontsize=12)
    plt.grid(True, alpha=0.3)

    # La légende est placée en bas à droite pour ne pas masquer le creux du Theta.
    plt.legend(loc='lower right', fontsize=10)

    # Ajustement automatique des marges.
    plt.tight_layout()

    # Affichage du graphique.
    plt.show()


# Ce bloc s'exécute uniquement lorsque Theta.py est lancé directement.
if __name__ == "__main__":
    tracer_graphique_theta()
