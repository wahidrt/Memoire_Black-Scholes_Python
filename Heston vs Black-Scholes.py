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
def heston_monte_carlo_calls(
    S0,
    v0,
    strikes,
    T,
    r,
    kappa,
    theta,
    xi,
    rho,
    M,
    N,
    graine=42,
):
    """
    Simule les trajectoires conjointes de l'actif et de sa variance sous Heston,
    puis calcule les prix des Calls pour toute une liste de strikes.

    M : nombre de trajectoires simulées
    N : nombre de pas de temps
    """
    rng = np.random.default_rng(graine)
    dt = T / N
    racine_dt = np.sqrt(dt)

    # Seules les valeurs courantes sont conservées afin de limiter la mémoire.
    S = np.full(M, S0, dtype=float)
    v = np.full(M, v0, dtype=float)

    # Boucle temporelle : schéma d'Euler-Maruyama avec troncature complète.
    for _ in range(N):
        Z1 = rng.standard_normal(M)
        Z2 = rng.standard_normal(M)

        # Construction de deux accroissements browniens corrélés.
        dW_S = racine_dt * Z1
        dW_v = racine_dt * (rho * Z1 + np.sqrt(1 - rho ** 2) * Z2)

        v_positif = np.maximum(v, 0)

        # Évolution de la variance selon le processus CIR de Heston.
        v = (
            v
            + kappa * (theta - v_positif) * dt
            + xi * np.sqrt(v_positif) * dW_v
        )

        # Évolution du prix de l'actif.
        S = S * np.exp(
            (r - 0.5 * v_positif) * dt
            + np.sqrt(v_positif) * dW_S
        )

    # Une même simulation terminale est utilisée pour tous les strikes.
    strikes = np.asarray(strikes, dtype=float)
    payoffs_finaux = np.maximum(S[:, np.newaxis] - strikes[np.newaxis, :], 0)

    return np.exp(-r * T) * np.mean(payoffs_finaux, axis=0)


# ==============================================================================
# 3. CONFRONTATION ET PRODUCTION DU GRAPHIQUE COMPARATIF
# ==============================================================================
if __name__ == "__main__":

    # --- Paramètres communs ---
    S0, r = 100.0, 0.05
    strikes = np.linspace(80, 120, 15)

    # Plusieurs maturités, exprimées en années.
    maturites = [0.25, 0.50, 0.75, 1.00]

    # --- Paramètres Black-Scholes ---
    sigma_bs = 0.20

    # --- Paramètres Heston ---
    v0 = 0.04
    theta = 0.04
    kappa = 2.0
    xi = 0.35
    rho = -0.75

    # --- Paramètres de Monte-Carlo ---
    M = 30000
    N = 100

    print(
        "Calcul des profils de prix pour plusieurs maturités "
        "(Monte-Carlo Heston vs analytique Black-Scholes)..."
    )

    # --- Création du graphique comparatif ---
    plt.figure(figsize=(12, 7))
    couleurs = plt.cm.viridis(np.linspace(0.10, 0.90, len(maturites)))

    for indice, (T, couleur) in enumerate(zip(maturites, couleurs)):
        # Prix Black-Scholes pour tous les strikes à la maturité T.
        prix_bs = bs_analytique_call(S0, strikes, T, r, sigma_bs)

        # Prix Heston pour tous les strikes à la même maturité T.
        prix_heston = heston_monte_carlo_calls(
            S0,
            v0,
            strikes,
            T,
            r,
            kappa,
            theta,
            xi,
            rho,
            M,
            N,
            graine=42 + indice,
        )

        # Même couleur pour une maturité donnée :
        # trait continu pour Heston et pointillé pour Black-Scholes.
        plt.plot(
            strikes,
            prix_heston,
            "o-",
            color=couleur,
            linewidth=2,
            markersize=4,
            label=f"Heston — T = {T:.2f} an",
        )
        plt.plot(
            strikes,
            prix_bs,
            "s--",
            color=couleur,
            linewidth=1.5,
            markersize=4,
            label=f"Black-Scholes — T = {T:.2f} an",
        )

    # --- Habillage du graphique ---
    plt.title(
        "Heston vs Black-Scholes pour plusieurs maturités",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Prix d'exercice / Strike (K)", fontsize=11)
    plt.ylabel("Prix du Call (€)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig(
        "Heston_vs_Black_Scholes_plusieurs_T.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
