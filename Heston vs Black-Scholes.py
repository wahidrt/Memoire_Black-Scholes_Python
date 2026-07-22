import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import quad
from scipy.stats import norm


# ================================================================
# 1. PRIX DE BLACK-SCHOLES
# ================================================================
def bs_call(S, K, tau, r, sigma):
    """Prix d'un Call européen; K peut être un tableau."""
    K = np.asarray(K, dtype=float)
    if tau <= 0:
        return np.maximum(S - K, 0.0)
    d1 = (
        np.log(S / K) + (r + 0.5 * sigma**2) * tau
    ) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)


# ================================================================
# 2. RÉFÉRENCE SEMI-ANALYTIQUE DE HESTON
# ================================================================
def heston_cf(u, S, v, tau, r, kappa, theta, xi, rho):
    """Fonction caractéristique de log(S_T) sous la mesure risque-neutre."""
    q = kappa - 1j * rho * xi * u
    d = np.sqrt(q * q + xi**2 * (u * u + 1j * u))

    # Le choix Re(d) >= 0 stabilise la branche de la racine complexe.
    if np.real(d) < 0:
        d = -d

    g = (q - d) / (q + d)
    exp_d = np.exp(-d * tau)
    A = (
        1j * u * r * tau
        + (kappa * theta / xi**2)
        * (
            (q - d) * tau
            - 2.0 * np.log((1.0 - g * exp_d) / (1.0 - g))
        )
    )
    B = (
        (q - d) / xi**2
        * (1.0 - exp_d) / (1.0 - g * exp_d)
    )
    return np.exp(1j * u * np.log(S) + A + B * v)


def heston_fourier_call(
    S, K, v, tau, r, kappa, theta, xi, rho, borne=200.0
):
    """Prix de Heston obtenu par deux intégrales de Fourier."""
    if tau <= 0:
        return max(S - K, 0.0)

    log_K = np.log(K)

    def integrande_P1(u):
        phi = heston_cf(
            u - 1j, S, v, tau, r, kappa, theta, xi, rho
        )
        denominateur = 1j * u * S * np.exp(r * tau)
        return np.real(
            np.exp(-1j * u * log_K) * phi / denominateur
        )

    def integrande_P2(u):
        phi = heston_cf(
            u, S, v, tau, r, kappa, theta, xi, rho
        )
        return np.real(
            np.exp(-1j * u * log_K) * phi / (1j * u)
        )

    options_quad = dict(
        epsabs=1e-9, epsrel=1e-9, limit=300
    )
    I1 = quad(integrande_P1, 0.0, borne, **options_quad)[0]
    I2 = quad(integrande_P2, 0.0, borne, **options_quad)[0]
    P1 = 0.5 + I1 / np.pi
    P2 = 0.5 + I2 / np.pi
    return S * P1 - K * np.exp(-r * tau) * P2


def heston_fourier_calls(
    S, strikes, v, tau, r, kappa, theta, xi, rho
):
    return np.array([
        heston_fourier_call(
            S, K, v, tau, r, kappa, theta, xi, rho
        )
        for K in np.asarray(strikes, dtype=float)
    ])


# ================================================================
# 3. MONTE-CARLO : TRONCATURE COMPLÈTE ET RÉDUCTION DE VARIANCE
# ================================================================
def observations_corrigees(S_T, S_t, strikes, tau, r):
    """Moyennes antithétiques corrigées par la variable de contrôle."""
    M = len(S_T)
    moitie = M // 2
    actualisation = np.exp(-r * tau)

    X = actualisation * np.maximum(
        S_T[:, None] - np.asarray(strikes)[None, :], 0.0
    )
    Y = actualisation * S_T

    # Les trajectoires de la seconde moitié sont les antithétiques
    # des trajectoires de la première moitié.
    X_paire = 0.5 * (X[:moitie] + X[moitie:])
    Y_paire = 0.5 * (Y[:moitie] + Y[moitie:])

    Y_centre = Y_paire - np.mean(Y_paire)
    denominateur = np.dot(Y_centre, Y_centre)
    beta = (
        Y_centre
        @ (X_paire - np.mean(X_paire, axis=0))
        / denominateur
    )

    # E[e^{-r tau} S_T] = S_t sous la mesure risque-neutre.
    return X_paire - (Y_paire[:, None] - S_t) * beta[None, :]


def heston_monte_carlo_calls(
    S_t,
    v_t,
    strikes,
    tau,
    r,
    kappa,
    theta,
    xi,
    rho,
    M=30_000,
    pas_par_an=100,
    graine=42,
):
    """Prix, erreur-type et intervalle ponctuel à 95 % pour chaque strike."""
    if M % 2 != 0:
        raise ValueError("M doit être pair pour les variables antithétiques")

    strikes = np.asarray(strikes, dtype=float)
    if tau <= 0:
        prix = np.maximum(S_t - strikes, 0.0)
        zeros = np.zeros_like(prix)
        return prix, zeros, prix, prix

    rng = np.random.default_rng(graine)
    N = max(1, int(np.ceil(pas_par_an * tau)))
    dt = tau / N
    racine_dt = np.sqrt(dt)
    moitie = M // 2

    S = np.full(M, S_t, dtype=float)
    v = np.full(M, v_t, dtype=float)

    for _ in range(N):
        # Une seule moitié est tirée; l'autre reçoit les opposés.
        Z1_moitie = rng.standard_normal(moitie)
        Z2_moitie = rng.standard_normal(moitie)
        Z1 = np.concatenate((Z1_moitie, -Z1_moitie))
        Z2 = np.concatenate((Z2_moitie, -Z2_moitie))

        v_positif = np.maximum(v, 0.0)
        dW_S = racine_dt * Z1
        dW_v = racine_dt * (
            rho * Z1 + np.sqrt(1.0 - rho**2) * Z2
        )

        v = (
            v
            + kappa * (theta - v_positif) * dt
            + xi * np.sqrt(v_positif) * dW_v
        )
        S *= np.exp(
            (r - 0.5 * v_positif) * dt
            + np.sqrt(v_positif) * dW_S
        )

    Z = observations_corrigees(S, S_t, strikes, tau, r)
    prix = np.mean(Z, axis=0)
    erreur_type = np.std(Z, axis=0, ddof=1) / np.sqrt(moitie)
    borne_basse = prix - 1.96 * erreur_type
    borne_haute = prix + 1.96 * erreur_type
    return prix, erreur_type, borne_basse, borne_haute


# ================================================================
# 4. CONVERGENCE TEMPORELLE AVEC MOUVEMENTS BROWNIENS COMMUNS
# ================================================================
def observations_sur_maillage_couple(
    S_t,
    v_t,
    strikes,
    tau,
    r,
    kappa,
    theta,
    xi,
    rho,
    Z1_fin,
    Z2_fin,
    N,
):
    """Observations antithétiques sur un sous-maillage du maillage fin."""
    N_fin, moitie = Z1_fin.shape
    bloc = N_fin // N
    if N_fin % N != 0:
        raise ValueError("N doit diviser le nombre de pas du maillage fin")

    # Somme normalisée : un bloc reste de loi N(0,1).
    Z1_blocs = (
        Z1_fin.reshape(N, bloc, moitie).sum(axis=1) / np.sqrt(bloc)
    )
    Z2_blocs = (
        Z2_fin.reshape(N, bloc, moitie).sum(axis=1) / np.sqrt(bloc)
    )

    dt = tau / N
    racine_dt = np.sqrt(dt)
    S = np.full(2 * moitie, S_t, dtype=float)
    v = np.full(2 * moitie, v_t, dtype=float)

    for Z1_moitie, Z2_moitie in zip(Z1_blocs, Z2_blocs):
        Z1 = np.concatenate((Z1_moitie, -Z1_moitie))
        Z2 = np.concatenate((Z2_moitie, -Z2_moitie))
        v_positif = np.maximum(v, 0.0)
        dW_S = racine_dt * Z1
        dW_v = racine_dt * (
            rho * Z1 + np.sqrt(1.0 - rho**2) * Z2
        )
        v = (
            v
            + kappa * (theta - v_positif) * dt
            + xi * np.sqrt(v_positif) * dW_v
        )
        S *= np.exp(
            (r - 0.5 * v_positif) * dt
            + np.sqrt(v_positif) * dW_S
        )

    return observations_corrigees(S, S_t, strikes, tau, r)


def etude_convergence_temporelle(parametres):
    """Compare N = 25, 50, 100 et 200 avec les mêmes browniens."""
    rng = np.random.default_rng(54321)
    moitie = 30_000              # M = 60 000 trajectoires
    Z1_fin = rng.standard_normal((200, moitie))
    Z2_fin = rng.standard_normal((200, moitie))
    observations = {}

    for N in [25, 50, 100, 200]:
        observations[N] = observations_sur_maillage_couple(
            strikes=[100.0],
            Z1_fin=Z1_fin,
            Z2_fin=Z2_fin,
            N=N,
            **parametres,
        )[:, 0]

    reference = observations[200]
    for N in [25, 50, 100, 200]:
        Z = observations[N]
        difference = Z - reference
        demi_ic_difference = (
            1.96 * np.std(difference, ddof=1) / np.sqrt(moitie)
        )
        print(
            f"N={N:3d} | prix={np.mean(Z):.6f} | "
            f"écart à N=200={np.mean(difference):.6f} | "
            f"demi-IC={demi_ic_difference:.6f}"
        )


# ================================================================
# 5. EXPÉRIENCE, TESTS ET GRAPHIQUES
# ================================================================
if __name__ == "__main__":
    S_t, v_t, r = 100.0, 0.04, 0.05
    kappa, theta, xi, rho = 2.0, 0.04, 0.35, -0.75
    sigma_bs = 0.20
    T = 1.0
    strikes = np.linspace(80.0, 120.0, 21)
    dates = [0.00, 0.25, 0.50, 0.75]
    M = 30_000
    pas_par_an = 100
    resultats = []

    for indice, t in enumerate(dates):
        tau = T - t
        prix_bs = bs_call(S_t, strikes, tau, r, sigma_bs)
        prix_fourier = heston_fourier_calls(
            S_t, strikes, v_t, tau, r,
            kappa, theta, xi, rho,
        )
        prix_mc, se_mc, ic_bas, ic_haut = heston_monte_carlo_calls(
            S_t=S_t,
            v_t=v_t,
            strikes=strikes,
            tau=tau,
            r=r,
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho,
            M=M,
            pas_par_an=pas_par_an,
            graine=42 + indice,
        )

        rmse = np.sqrt(np.mean((prix_mc - prix_fourier) ** 2))
        couverture = np.mean(
            (prix_fourier >= ic_bas) & (prix_fourier <= ic_haut)
        )
        print(
            f"t={t:.2f} | RMSE={rmse:.6f} | "
            f"couverture ponctuelle={couverture:.1%}"
        )
        resultats.append(
            (t, tau, prix_bs, prix_fourier, prix_mc, se_mc)
        )

    # Contrôle de la quadrature pour trois strikes.
    for K in [80.0, 100.0, 120.0]:
        valeurs = [
            heston_fourier_call(
                S_t, K, v_t, 1.0, r,
                kappa, theta, xi, rho, borne=U,
            )
            for U in [100.0, 150.0, 200.0]
        ]
        print(f"K={K:.0f} | quadrature : {valeurs}")

    # Bornes d'arbitrage, décroissance et convexité en strike.
    for _, tau, _, prix_fourier, _, _ in resultats:
        borne_basse = np.maximum(
            S_t - strikes * np.exp(-r * tau), 0.0
        )
        assert np.all(prix_fourier >= borne_basse - 1e-8)
        assert np.all(prix_fourier <= S_t + 1e-8)
        assert np.all(np.diff(prix_fourier) <= 1e-8)
        assert np.all(np.diff(prix_fourier, n=2) >= -1e-8)

    # Limite vers Black-Scholes lorsque xi est presque nul.
    prix_xi_faible = heston_fourier_call(
        S_t, 100.0, v_t, 1.0, r,
        kappa, theta, 1e-4, rho,
    )
    prix_bs_atm = bs_call(S_t, 100.0, 1.0, r, sigma_bs)
    print("Test xi -> 0 :", abs(prix_xi_faible - prix_bs_atm))

    # Convergence de l'erreur statistique en M.
    for M_test in [7_500, 15_000, 30_000, 60_000]:
        prix, se, _, _ = heston_monte_carlo_calls(
            S_t, v_t, [100.0], 1.0, r,
            kappa, theta, xi, rho,
            M=M_test, pas_par_an=100, graine=12345,
        )
        print(
            f"M={M_test:6d} | prix={prix[0]:.6f} | "
            f"demi-IC={1.96 * se[0]:.6f}"
        )

    parametres_convergence = dict(
        S_t=S_t,
        v_t=v_t,
        tau=1.0,
        r=r,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
    )
    etude_convergence_temporelle(parametres_convergence)

    # Figure 1 : comparaison lisible des niveaux de prix.
    fig, ax = plt.subplots(figsize=(13, 8))
    couleurs = plt.cm.viridis(
        np.linspace(0.10, 0.90, len(resultats))
    )
    for couleur, resultat in zip(couleurs, resultats):
        t, tau, prix_bs, prix_fourier, prix_mc, se_mc = resultat
        ax.plot(
            strikes, prix_fourier, color=couleur, linewidth=2.2,
        )
        ax.scatter(
            strikes, prix_mc, s=22, marker="o",
            facecolors="white", edgecolors=[couleur],
            linewidths=1.1, alpha=0.9, zorder=3,
        )
        ax.plot(
            strikes, prix_bs, "--", color=couleur, linewidth=1.7,
        )

    legende_methodes = [
        Line2D([0], [0], color="black", linewidth=2.2,
               label="Heston (Fourier)"),
        Line2D([0], [0], color="black", linewidth=1.7,
               linestyle="--", label="Black-Scholes"),
        Line2D([0], [0], color="black", marker="o",
               markerfacecolor="white", linestyle="None",
               label="Heston (Monte-Carlo)"),
    ]
    legende_dates = [
        Line2D([0], [0], color=couleur, linewidth=3,
               label=f"t={resultat[0]:.2f}")
        for couleur, resultat in zip(couleurs, resultats)
    ]
    premiere_legende = ax.legend(
        handles=legende_methodes, title="Méthode",
        loc="upper right", fontsize=9,
    )
    ax.add_artist(premiere_legende)
    ax.legend(
        handles=legende_dates, title="Date d'évaluation",
        loc="lower left", fontsize=9,
    )
    ax.set_xlabel("Prix d'exercice K")
    ax.set_ylabel("Prix du Call (euro)")
    ax.set_title("Heston et Black-Scholes -- échéance fixe T=1")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(
        "Heston_vs_Black_Scholes_au_fil_du_temps.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Figure 2 : écart de modèle et incertitude Monte-Carlo visible.
    fig, ax = plt.subplots(figsize=(11, 7))
    for couleur, resultat in zip(couleurs, resultats):
        t, tau, prix_bs, prix_fourier, prix_mc, se_mc = resultat
        ax.plot(
            strikes, prix_fourier - prix_bs,
            color=couleur, linewidth=2.2,
        )
        ax.errorbar(
            strikes, prix_mc - prix_bs, yerr=1.96 * se_mc,
            fmt="o", color=couleur, markersize=3.8,
            capsize=3, elinewidth=1.2, capthick=1.2,
            alpha=0.85, zorder=3,
        )

    legende_methodes = [
        Line2D([0], [0], color="black", linewidth=2.2,
               label="Fourier - Black-Scholes"),
        ax.errorbar(
            [np.nan], [np.nan], yerr=[0.1], fmt="o",
            color="black", markersize=3.8, capsize=3,
            elinewidth=1.2, label=r"Monte-Carlo $\pm$ IC à 95 %",
        ),
    ]
    premiere_legende = ax.legend(
        handles=legende_methodes, title="Quantité représentée",
        loc="lower left", fontsize=9,
    )
    ax.add_artist(premiere_legende)
    ax.legend(
        handles=legende_dates, title="Date d'évaluation",
        loc="upper right", fontsize=9,
    )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Prix d'exercice K")
    ax.set_ylabel("Écart de prix (euro)")
    ax.set_title("Écart Heston - Black-Scholes")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(
        "Ecart_Heston_moins_Black_Scholes_au_fil_du_temps.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
