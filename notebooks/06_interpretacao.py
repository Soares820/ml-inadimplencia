# %% [markdown]
# # Passo 6 — Interpretação (feature importance, SHAP)
# Previsão de inadimplência de cartão de crédito.
#
# Em crédito, um modelo "caixa-preta" não basta: cliente e regulador têm o
# direito de entender por que um pedido foi negado. A importância nativa do
# XGBoost (Passo 4) diz o que importa **em média**. SHAP vai além: explica
# **uma previsão específica**, com sinal (a variável empurrou o risco pra cima
# ou pra baixo) — baseado em teoria dos jogos (valores de Shapley).

# %%
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Carregar o modelo tuned (Passo 5) e os dados processados

# %%
train_df = pd.read_csv("../data/processed/train.csv")
test_df = pd.read_csv("../data/processed/test.csv")
pipeline = joblib.load("../models/preprocessor.pkl")
modelo = joblib.load("../models/melhor_modelo_tuned.pkl")

target = "default"
X_test, y_test = test_df.drop(columns=[target]), test_df[target]
X_test_proc = pipeline.transform(X_test)
feature_names = [f.replace("num__", "").replace("cat__", "")
                  for f in pipeline.named_steps["preprocessor"].get_feature_names_out()]

proba_test = modelo.predict_proba(X_test_proc)[:, 1]
print("Modelo:", type(modelo).__name__, "| Amostras de teste:", X_test_proc.shape[0])

# %% [markdown]
# ## 2. Calcular valores SHAP no conjunto de teste
# `TreeExplainer` é exato e rápido para modelos de árvore (XGBoost) — sem
# precisar de amostragem/aproximação.

# %%
explainer = shap.TreeExplainer(modelo)
shap_values = explainer(X_test_proc)
shap_values.feature_names = feature_names

# %% [markdown]
# ## 3. Importância global via SHAP (beeswarm)
# Cada ponto é um cliente do teste. Posição no eixo x = quanto aquela variável
# empurrou a previsão daquele cliente (positivo = mais risco). Cor = valor da
# variável (vermelho = alto, azul = baixo).

# %%
plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values, X_test_proc, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig("../data/shap_summary.png", dpi=100, bbox_inches="tight")
plt.close()
print("Salvo: data/shap_summary.png")

# %% [markdown]
# ## 4. Importância média |SHAP| — compara com a importância nativa do XGBoost

# %%
importancia_shap = pd.Series(
    np.abs(shap_values.values).mean(axis=0), index=feature_names
).sort_values(ascending=False)
print("\nTop 10 features por |SHAP| médio:")
print(importancia_shap.head(10).round(4))

# %% [markdown]
# ## 5. Explicação individual: um cliente de alto risco e um de baixo risco
# Pegamos casos reais do teste — o de maior e o de menor probabilidade prevista
# — para ver quais variáveis pesaram em cada decisão.

# %%
idx_risco = int(np.argmax(proba_test))
idx_seguro = int(np.argmin(proba_test))

print(f"\nCliente de MAIOR risco previsto (linha {idx_risco}):")
print(f"  Probabilidade prevista: {proba_test[idx_risco]:.4f} | Rótulo real: {y_test.iloc[idx_risco]}")
print(X_test.iloc[idx_risco][["LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "BILL_AMT1", "PAY_AMT1"]])

print(f"\nCliente de MENOR risco previsto (linha {idx_seguro}):")
print(f"  Probabilidade prevista: {proba_test[idx_seguro]:.4f} | Rótulo real: {y_test.iloc[idx_seguro]}")
print(X_test.iloc[idx_seguro][["LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "BILL_AMT1", "PAY_AMT1"]])

# %%
plt.figure()
shap.plots.waterfall(shap_values[idx_risco], show=False)
plt.tight_layout()
plt.savefig("../data/shap_waterfall_alto_risco.png", dpi=100, bbox_inches="tight")
plt.close()

plt.figure()
shap.plots.waterfall(shap_values[idx_seguro], show=False)
plt.tight_layout()
plt.savefig("../data/shap_waterfall_baixo_risco.png", dpi=100, bbox_inches="tight")
plt.close()
print("\nSalvos: data/shap_waterfall_alto_risco.png, data/shap_waterfall_baixo_risco.png")

# %% [markdown]
# ## 6. Conclusões do Passo 6
# - `PAY_0` domina tanto a importância nativa do XGBoost quanto o SHAP — dois
#   métodos diferentes concordando é um bom sinal de que o sinal é real, não
#   artefato do algoritmo.
# - O waterfall individual mostra que a **mesma variável pode empurrar em
#   direções opostas** dependendo do cliente (ex.: `PAY_0` em dia reduz risco;
#   `PAY_0` atrasado aumenta) — é exatamente o que a importância global (uma
#   média) não consegue mostrar.
# - Essa é a peça que falta para justificar uma decisão de crédito a um
#   cliente ou auditor: não "o modelo achou", mas "essas N variáveis, com
#   esses valores, pesaram assim".
#
# → Próximo: **Passo 7 — Deploy (FastAPI + endpoint de previsão).**
