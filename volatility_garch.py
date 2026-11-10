# %%
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt

from statsmodels.tsa.stattools import adfuller
from arch import arch_model

# %%
NB_TRADING_DAYS = 252

# %% [markdown]
# # Récupération et nettoyage des données

# %% [markdown]
# ## Téléchargement des données quotidiennes du S&P500
# 
# Dates : 01/01/2018 $\to$ 31/12/2023

# %%
ticker = "^GSPC"
df_lf_long: pd.DataFrame = yf.download(ticker, start="2018-01-01", end="2023-12-31")
df_lf_long.columns = df_lf_long.columns.droplevel("Ticker")

# %% [markdown]
# ## Calcul du *log return*

# %%
df_lf_long = (
    pd.DataFrame({"Price" : df_lf_long["Close"]})
    .reset_index(drop=False)
    .dropna()
)

df_lf_long["Log_Return"] = np.log(df_lf_long["Price"] / df_lf_long["Price"].shift(1))

df_lf_long = df_lf_long.dropna()

# %%
fig, ax = plt.subplots()

fig.set_size_inches(12, 5)
ax.plot(df_lf_long["Date"], df_lf_long["Log_Return"], label="Log Returns")
ax.axhline(0, color="black")
ax.set_title("Log-returns quotidiens du S&P 500 (2018-2023)")
ax.set_xlabel("Date")
ax.set_ylabel("Log Return")
plt.show()

# %% [markdown]
# ### Vérification de la stationnarité via le test ADF

# %%
adf_result = adfuller(df_lf_long["Log_Return"], result_object=False)

print("--- Résultats du Test ADF ---")
print(f"Statistique ADF : {adf_result[0]:.4f}")
print(f"p-value         : {adf_result[1]:.4e}")
print("Valeurs critiques :")
for key, value in adf_result[4].items():
    print(f"   {key}: {value:.4f}")

# %% [markdown]
# La valeur obtenue est inférieure à la valeur critique à 1%, et la p-value est très petite, ce qui conforte dans l'hypothèse de la stationnarité : on rejette l'hypothèse nulle selon laquelle on aurait $\gamma = 0$ dans le modèle ADF.

# %% [markdown]
# # Volatilité réalisée

# %% [markdown]
# ## Téléchargement des données haute fréquence

# %%
ticker = "^GSPC"
df_hf_1mo: pd.DataFrame = yf.download(ticker, period="1mo", interval="5m")
df_lf_1mo: pd.DataFrame = yf.download(ticker, period="1mo")
df_hf_1mo.columns = df_hf_1mo.columns.droplevel("Ticker")
df_lf_1mo.columns = df_lf_1mo.columns.droplevel("Ticker")

# %% [markdown]
# ## Calculs de volatilité

# %% [markdown]
# ### Calcul de $RV_t$

# %%
df_hf_1mo = (
    pd.DataFrame({"Price" : df_hf_1mo["Close"]})
    .reset_index(drop=False)
    .dropna()
)
df_hf_1mo["Date"] = df_hf_1mo["Datetime"].dt.floor("D")
df_hf_1mo["Log_Return"] = np.log(df_hf_1mo["Price"] / df_hf_1mo["Price"].shift(1))
df_hf_1mo = df_hf_1mo.dropna()
df_sq_returns = (df_hf_1mo["Log_Return"]**2).groupby(df_hf_1mo["Date"]).sum().reset_index()
df_volatility = pd.DataFrame(
    {
        "Date": pd.to_datetime(df_sq_returns["Date"]).dt.tz_localize(None).astype("datetime64[ns]"),
        "realized_volatility" : df_sq_returns["Log_Return"]
    }
)

# %% [markdown]
# ### Volatilité annualisée

# %% [markdown]
# On ramène la volatilité réalisée au nombre de jours de bourse. On prend la racine carrée pour avoir l'écart-type plutôt que la variance.

# %%
annualized_volatility = np.sqrt(df_volatility["realized_volatility"].mean() * NB_TRADING_DAYS)

# %% [markdown]
# ### Volatilité historique

# %%
df_lf_1mo = (
    pd.DataFrame({"Price" : df_lf_1mo["Close"]})
    .reset_index(drop=False)
    .dropna()
)
df_lf_1mo["Log_Return"] = np.log(df_lf_1mo["Price"] / df_lf_1mo["Price"].shift(1))
df_lf_1mo = df_lf_1mo.dropna()

# %%
historical_volatility = np.sqrt(df_lf_1mo["Log_Return"].std()**2 * NB_TRADING_DAYS)

# %%
print(f"--- Comparaison sur les 30 derniers jours ---")
print(f"Volatilité Réalisée (5 min) annualisée   : {annualized_volatility:.2f}")
print(f"Volatilité Historique (Clôtures) annualisée : {historical_volatility:.2f}")

# %% [markdown]
# # Prédiction de la volatilité - modèle GARCH(1,1)

# %% [markdown]
# Astuce : multiplier les rendements par 100 améliore la convergence numérique de l'optimiseur.

# %%
model = arch_model(df_lf_long["Log_Return"], vol="Garch", p=1, q=1, mean="Constant", rescale=True)

# %%
results = model.fit(disp="off")

# %%
print(results.summary())

# %% [markdown]
# Commentaire : la régression du modèle GARCH(1,1) sur les données historiques fournit une valeur de $\omega$ à 0.1744, $\alpha$ à 0.5269 et $\beta$ à 0.4731. On est proche d'un modèle IGARCH.  
# 
# Cette observation peut s'expliquer par le fait que la période contient des chocs extrêmes (comme la crise du CoViD-19), où la volatilité a explosé avant de revenir très lentement à son niveau habituel.

# %% [markdown]
# # Evaluation du modèle GARCH(1,1) hors échantillon

# %% [markdown]
# ## Séparation des données

# %% [markdown]
# Données d'entraînement *in sample*

# %%
ticker = "^GSPC"
df_sample = yf.download(ticker, start="2018-01-01", end="2026-10-11", multi_level_index=False)

df_sample = (
    pd.DataFrame({"Price" : df_sample["Close"]})
    .reset_index(drop=False)
    .dropna()
)

df_sample["Log_Return"] = np.log(df_lf_long["Price"] / df_lf_long["Price"].shift(1))

df_sample = df_sample.dropna()

# %% [markdown]
# Données de test *out of sample*

# %%
ticker = "^GSPC"
df_test = yf.download(ticker, start="2026-10-12", end="2026-11-10", multi_level_index=False)

df_test = (
    pd.DataFrame({"Price" : df_test["Close"]})
    .reset_index(drop=False)
    .dropna()
)

df_test["Log_Return"] = np.log(df_test["Price"] / df_test["Price"].shift(1))

df_test = df_test.dropna()

# %% [markdown]
# Ajustement du modèle sur les données d'entraînement

# %%
model = arch_model(df_sample["Log_Return"], vol="Garch", p=1, q=1, mean="Constant", rescale=True)
results = model.fit(disp="off")

# %% [markdown]
# ## Forecasting de volatilité

# %%
omega = results.params["omega"]
alpha = results.params["alpha[1]"]
beta = results.params["beta[1]"]
scale = results.scale

# %%
last_var_in_sample = results.conditional_volatility.iloc[-1] ** 2

# %%
forecasted_vars = []
log_returns_scaled  = df_test["Log_Return"] * scale
current_var = last_var_in_sample

for log_return in log_returns_scaled:
    pred_var = omega + alpha*log_return**2 + beta*current_var
    forecasted_vars.append(pred_var)
    current_var = pred_var

df_test["forecasted_volatility"] = np.sqrt(forecasted_vars) / scale

# %% [markdown]
# ## Comparaison à une volatilité de référence

# %%
df_test = df_test.merge(right=df_volatility, how="inner", on="Date")
df_test["realized_sigma"] = np.sqrt(df_test["realized_volatility"])

# %%
fig, ax = plt.subplots()
fig.set_size_inches(12, 5)
ax.set_title("Back-testing du modèle GARCH(1,1) sur le S&P-500 entre le 13/10/2026 et le 09/11/2026")
ax.plot(df_test["Date"], df_test["realized_sigma"], label=r"$\sqrt{RV_t}$")
ax.plot(df_test["Date"], df_test["forecasted_volatility"], label=r"$\hat{\sigma}_t$")
ax.legend()
ax.set_xlabel("Date")
plt.show()

# %% [markdown]
# ### Calcul des métriques d'erreurs

# %%
root_mean_squared_error = np.sqrt(
    np.mean(
        (df_test["realized_sigma"] - df_test["forecasted_volatility"]) ** 2
    )
)

# %%
mean_absolute_error = np.mean(
    np.abs(
        df_test["realized_sigma"] - df_test["forecasted_volatility"]
    )
)

# %%
print(f"Erreur quadratique moyenne : {root_mean_squared_error:.2e}")
print(f"Erreur absolue moyenne : {mean_absolute_error:.2e}")


