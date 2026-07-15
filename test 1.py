import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Solution Analytique de Référence
def bs_analytique_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)

# 2. Résolution EDP (Schéma Implicite)
def black_scholes_mdf_implicite(S0, K, T, r, sigma, S_max, M, N):
    dt, dS = T / N, S_max / M
    S_space = np.linspace(0, S_max, M + 1)
    t_space = np.linspace(0, T, N + 1)
    
    # Grille des prix
    V = np.zeros((M + 1, N + 1))
    
    # Condition terminale (Payoff)
    V[:, -1] = np.maximum(S_space - K, 0)
    
    # Conditions aux limites
    V[0, :] = 0
    V[-1, :] = S_max - K * np.exp(-r * (T - t_space))
    
    # Coefficients de la matrice tridiagonale
    i = np.arange(1, M)
    alpha = 0.5 * dt * (r * i - (sigma ** 2) * (i ** 2))
    beta = 1 + dt * ((sigma ** 2) * (i ** 2) + r)
    gamma = -0.5 * dt * (r * i + (sigma ** 2) * (i ** 2))
    
    # Construction et inversion de la matrice A
    A = np.diag(beta) + np.diag(alpha[1:], -1) + np.diag(gamma[:-1], 1)
    A_inv = np.linalg.inv(A)
    
    # Récurrence rétrograde
    for n in reversed(range(0, N)):
        B = V[1:M, n + 1].copy()
        B[0] -= alpha[0] * V[0, n]
        B[-1] -= gamma[-1] * V[-1, n]
        V[1:M, n] = A_inv.dot(B)
        
    prix_mdf = np.interp(S0, S_space, V[:, 0])
    return S_space, t_space, V, prix_mdf

# --- Paramètres de marché et calculs ---
S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
S_max_affichage = 200  # Limite pour un joli zoom 3D
M, N = 100, 100

S_grid, t_grid, V_matrice, prix_mdf = black_scholes_mdf_implicite(S0, K, T, r, sigma, S_max_affichage, M, N)
prix_exact = bs_analytique_call(S0, K, T, r, sigma)

# --- Affichage des résultats numériques ---
print("=== RÉSULTATS ===")
print(f"Prix Exact (BS)  : {prix_exact:.4f} €")
print(f"Prix Numérique   : {prix_mdf:.4f} €")
print(f"Erreur Absolue   : {abs(prix_mdf - prix_exact):.4f} €")

# --- Génération de la surface 3D ---
X, Y = np.meshgrid(t_grid, S_grid)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, V_matrice, cmap='viridis', edgecolor='none', alpha=0.9)

ax.set_title("Surface de prix du Call Européen (EDP Implicite)", fontsize=13, fontweight='bold', pad=20)
ax.set_xlabel("Temps (t)", fontsize=11, labelpad=10)
ax.set_ylabel("Prix du Sous-jacent (S)", fontsize=11, labelpad=10)
ax.set_zlabel("Prix de l'Option (V)", fontsize=11, labelpad=10)

ax.view_init(elev=25, azim=-120)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Valeur de l'option (€)")

plt.tight_layout()

# Sauvegarde pour ton rapport LaTeX (optionnel)
plt.savefig('surface_edp.pdf')

plt.show()
