import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def prix_call_black_scholes(S0, K, T, r, sigma):
    """Prix analytique d'un call européen sans dividende."""
    d1 = (
        np.log(S0 / K) + (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def statistiques_cumulatives(echantillon, tailles):
    """Moyenne, erreur standard et IC à 95 % pour plusieurs préfixes."""
    sommes = np.cumsum(echantillon)
    sommes_carres = np.cumsum(echantillon**2)
    indices = tailles - 1
    moyennes = sommes[indices] / tailles
    variances = (
        sommes_carres[indices] - tailles * moyennes**2
    ) / (tailles - 1)
    variances = np.maximum(variances, 0.0)
    erreurs_standard = np.sqrt(variances / tailles)
    marge = norm.ppf(0.975) * erreurs_standard
    return moyennes, erreurs_standard, moyennes - marge, moyennes + marge


def comparaison_monte_carlo(
    S0,
    K,
    T,
    r,
    sigma,
    nombre_paires=30000,
    pas_paires=250,
    graine=42,
):
    """Compare Monte-Carlo standard et variables antithétiques."""
    if nombre_paires < 2 or pas_paires < 1:
        raise ValueError("Le nombre de paires et le pas sont invalides.")

    actualisation = np.exp(-r * T)
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T)

    # Estimateur antithétique : une observation indépendante par paire.
    rng_anti = np.random.default_rng(graine)
    z = rng_anti.standard_normal(nombre_paires)
    st_plus = S0 * np.exp(drift + diffusion * z)
    st_moins = S0 * np.exp(drift - diffusion * z)
    observations_anti = 0.5 * actualisation * (
        np.maximum(st_plus - K, 0.0)
        + np.maximum(st_moins - K, 0.0)
    )

    # Estimateur standard à coût identique : deux tirages par paire.
    rng_standard = np.random.default_rng(graine + 1)
    z_standard = rng_standard.standard_normal(2 * nombre_paires)
    st_standard = S0 * np.exp(drift + diffusion * z_standard)
    observations_standard = actualisation * np.maximum(
        st_standard - K, 0.0
    )

    debut = min(max(2, pas_paires), nombre_paires)
    tailles_paires = np.arange(
        debut, nombre_paires + 1, pas_paires, dtype=int
    )
    if tailles_paires.size == 0 or tailles_paires[-1] != nombre_paires:
        tailles_paires = np.append(tailles_paires, nombre_paires)
    couts = 2 * tailles_paires

    moyenne_anti, se_anti, bas_anti, haut_anti = (
        statistiques_cumulatives(observations_anti, tailles_paires)
    )
    moyenne_std, se_std, bas_std, haut_std = (
        statistiques_cumulatives(observations_standard, couts)
    )

    return {
        "couts": couts,
        "moyenne_standard": moyenne_std,
        "se_standard": se_std,
        "ic_standard": (bas_std, haut_std),
        "moyenne_anti": moyenne_anti,
        "se_anti": se_anti,
        "ic_anti": (bas_anti, haut_anti),
    }


if __name__ == "__main__":
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    prix_exact = prix_call_black_scholes(S0, K, T, r, sigma)
    resultats = comparaison_monte_carlo(S0, K, T, r, sigma)

    couts = resultats["couts"]
    prix_std = resultats["moyenne_standard"]
    prix_anti = resultats["moyenne_anti"]
    se_std = resultats["se_standard"]
    se_anti = resultats["se_anti"]
    bas_anti, haut_anti = resultats["ic_anti"]
    bas_std, haut_std = resultats["ic_standard"]

    print(f"Prix analytique                 : {prix_exact:.6f}")
    print(
        f"Monte-Carlo standard            : {prix_std[-1]:.6f} "
        f"(SE = {se_std[-1]:.6f})"
    )
    print(
        f"IC standard à 95 %              : "
        f"[{bas_std[-1]:.6f}, {haut_std[-1]:.6f}]"
    )
    print(
        f"Monte-Carlo antithétique        : {prix_anti[-1]:.6f} "
        f"(SE = {se_anti[-1]:.6f})"
    )
    print(
        f"IC antithétique à 95 %          : "
        f"[{bas_anti[-1]:.6f}, {haut_anti[-1]:.6f}]"
    )
    print(
        "Facteur de réduction de variance : "
        f"{(se_std[-1] / se_anti[-1])**2:.3f}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].axhline(
        prix_exact,
        color="black",
        linestyle="--",
        label=f"Prix analytique = {prix_exact:.4f}",
    )
    axes[0].plot(couts, prix_std, color="tab:orange", label="Standard")
    axes[0].plot(couts, prix_anti, color="tab:blue", label="Antithétique")
    axes[0].fill_between(
        couts,
        bas_anti,
        haut_anti,
        color="tab:blue",
        alpha=0.15,
        label="IC antithétique à 95 %",
    )
    axes[0].set_xlabel("Nombre total de prix terminaux simulés")
    axes[0].set_ylabel("Prix estimé")
    axes[0].set_title("Convergence des estimateurs")
    axes[0].grid(linestyle=":", alpha=0.6)
    axes[0].legend()

    reference = se_anti[0] * np.sqrt(couts[0] / couts)
    axes[1].loglog(couts, se_std, label="Erreur standard")
    axes[1].loglog(couts, se_anti, label="Erreur antithétique")
    axes[1].loglog(
        couts,
        reference,
        "k--",
        label=r"Référence en $N^{-1/2}$",
    )
    axes[1].set_xlabel("Nombre total de prix terminaux simulés")
    axes[1].set_ylabel("Erreur standard estimée")
    axes[1].set_title("Vitesse de convergence statistique")
    axes[1].grid(linestyle=":", alpha=0.6)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(
        "convergence_monte_carlo.png",
        dpi=300,
        bbox_inches="tight",
    )