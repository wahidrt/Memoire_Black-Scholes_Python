import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt


# ==============================================================================
# 1. MODÈLE DE BLACK-SCHOLES
# ==============================================================================
def bs_analytique_call(S_t, K, tau, r, sigma):
    """Prix du Call à la date t, avec tau = T - t années avant l'échéance."""
    K = np.asarray(K, dtype=float)

    if tau <= 0:
        return np.maximum(S_t - K, 0.0)

    d1 = (
        np.log(S_t / K) + (r + 0.5 * sigma**2) * tau
    ) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    return S_t * si.norm.cdf(d1) - K * np.exp(-r * tau) * si.norm.cdf(d2)


# ==============================================================================
# 2. MODÈLE DE HESTON PAR MONTE-CARLO
# ==============================================================================
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
    M,
    pas_par_an,
    graine=42,
):
    """
    Calcule à la date t les prix des Calls sous Heston.

    La simulation part de l'état courant (S_t, v_t) et va jusqu'à l'échéance
    pendant la durée résiduelle tau = T - t.
    """
    strikes = np.asarray(strikes, dtype=float)

    if tau <= 0:
        return np.maximum(S_t - strikes, 0.0)

    rng = np.random.default_rng(graine)
    N = max(1, int(np.ceil(pas_par_an * tau)))
    dt = tau / N
    racine_dt = np.sqrt(dt)

    S = np.full(M, S_t, dtype=float)
    v = np.full(M, v_t, dtype=float)

    for _ in range(N):
        Z1 = rng.standard_normal(M)
        Z2 = rng.standard_normal(M)

        dW_S = racine_dt * Z1
        dW_v = racine_dt * (rho * Z1 + np.sqrt(1.0 - rho**2) * Z2)

        # Troncature complète pour éviter une variance négative dans les racines.
        v_positif = np.maximum(v, 0.0)

        v = (
            v
            + kappa * (theta - v_positif) * dt
            + xi * np.sqrt(v_positif) * dW_v
        )

        S *= np.exp(
            (r - 0.5 * v_positif) * dt
            + np.sqrt(v_positif) * dW_S
        )

    payoffs = np.maximum(S[:, np.newaxis] - strikes[np.newaxis, :], 0.0)
    return np.exp(-r * tau) * np.mean(payoffs, axis=0)


# ==============================================================================
# 3. COMPARAISON À PLUSIEURS DATES t POUR UNE MÊME ÉCHÉANCE T
# ==============================================================================
if __name__ == "__main__":
    # État observé du marché à chaque date t.
    # On le garde constant ici afin d'isoler uniquement l'effet du temps restant.
    S_t = 100.0
    v_t = 0.04
    r = 0.05

    strikes = np.linspace(80.0, 120.0, 21)

    # Une seule option, d'échéance finale T = 1 an.
    maturite_finale = 1.0
    dates_observation = [0.00, 0.25, 0.50, 0.75]

    # Paramètres Black-Scholes.
    sigma_bs = 0.20

    # Paramètres Heston.
    theta = 0.04
    kappa = 2.0
    xi = 0.35
    rho = -0.75

    # Paramètres de Monte-Carlo.
    M = 30_000
    pas_par_an = 200

    resultats = []

    print("Comparaison Heston / Black-Scholes au fil du temps")
    print(f"Échéance fixe : T = {maturite_finale:.2f} an")

    for indice, t in enumerate(dates_observation):
        tau = maturite_finale - t

        prix_bs = bs_analytique_call(S_t, strikes, tau, r, sigma_bs)
        prix_heston = heston_monte_carlo_calls(
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

        ecart = prix_heston - prix_bs
        resultats.append((t, tau, prix_bs, prix_heston, ecart))

        print(
            f"t = {t:.2f} | T-t = {tau:.2f} | "
            f"écart absolu moyen = {np.mean(np.abs(ecart)):.4f}"
        )

    # --------------------------------------------------------------------------
    # Figure 1 : comparaison des prix à chaque date d'observation.
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)

    for ax, (t, tau, prix_bs, prix_heston, ecart) in zip(
        axes.ravel(), resultats
    ):
        ax.plot(
            strikes,
            prix_heston,
            "o-",
            linewidth=2,
            markersize=4,
            label="Heston",
        )
        ax.plot(
            strikes,
            prix_bs,
            "s--",
            linewidth=1.7,
            markersize=4,
            label="Black-Scholes",
        )
        ax.fill_between(
            strikes,
            prix_heston,
            prix_bs,
            alpha=0.15,
            label="Écart entre les modèles",
        )
        ax.set_title(f"t = {t:.2f} an   (T - t = {tau:.2f} an)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Heston vs Black-Scholes au fil du temps — échéance fixe T = 1 an",
        fontsize=14,
        fontweight="bold",
    )
    fig.supxlabel("Prix d'exercice / Strike (K)")
    fig.supylabel("Prix du Call (€)")
    fig.tight_layout(rect=(0.03, 0.03, 1.0, 0.95))
    fig.savefig(
        "Heston_vs_Black_Scholes_au_fil_du_temps.png",
        dpi=300,
        bbox_inches="tight",
    )

    # --------------------------------------------------------------------------
    # Figure 2 : différence signée Heston - Black-Scholes selon t et K.
    # --------------------------------------------------------------------------
    plt.figure(figsize=(11, 6))

    for t, tau, prix_bs, prix_heston, ecart in resultats:
        plt.plot(
            strikes,
            ecart,
            "o-",
            linewidth=1.8,
            markersize=4,
            label=f"t = {t:.2f} (T-t = {tau:.2f})",
        )

    plt.axhline(0.0, color="black", linewidth=1)
    plt.title(
        "Écart de prix Heston − Black-Scholes au fil du temps",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Prix d'exercice / Strike (K)")
    plt.ylabel("Écart de prix (€)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "Ecart_Heston_moins_Black_Scholes_au_fil_du_temps.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
