# %% [markdown]
# # Passo 5 — Tuning de hiperparâmetros + ajuste de threshold
# Previsão de inadimplência de cartão de crédito.
#
# Dois problemas ficaram do Passo 4: Random Forest e XGBoost "decoraram" o
# treino (ROC-AUC = 1,00 e 0,997) e o modelo salvo do RF ficou com 182 MB.
# Aqui: (1) buscamos hiperparâmetros que generalizem melhor, restringindo o
# espaço de busca a valores regularizados; (2) depois de escolher o modelo,
# ajustamos o ponto de corte (threshold) por custo de erro, não por 0,5 padrão.

# %%
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

# %% [markdown]
# ## 1. Carregar dados processados (mesma lógica dos Passos 3 e 4)

# %%
train_df = pd.read_csv("../data/processed/train.csv")
test_df = pd.read_csv("../data/processed/test.csv")
pipeline = joblib.load("../models/preprocessor.pkl")

target = "default"
X_train, y_train = train_df.drop(columns=[target]), train_df[target]
X_test, y_test = test_df.drop(columns=[target]), test_df[target]

X_train_proc = pipeline.transform(X_train)
X_test_proc = pipeline.transform(X_test)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# %% [markdown]
# ## 2. Busca de hiperparâmetros (RandomizedSearchCV)
# O espaço de busca já exclui profundidade ilimitada de propósito — sabemos do
# Passo 4 que isso overfita. `scoring="roc_auc"` porque é a métrica que não
# depende de threshold (isso vem na etapa 4 deste passo).

# %%
buscas = {
    "random_forest": RandomizedSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1),
        param_distributions={
            "n_estimators": [100, 200, 300],
            "max_depth": [4, 6, 8, 10, 12],
            "min_samples_leaf": [1, 5, 10, 20, 50],
            "max_features": ["sqrt", "log2"],
        },
        n_iter=20, cv=skf, scoring="roc_auc", random_state=42, n_jobs=-1,
    ),
    "xgboost": RandomizedSearchCV(
        XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="logloss",
                      random_state=42, n_jobs=1),
        param_distributions={
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 4, 5, 6, 8],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_lambda": [0.5, 1, 2, 5],
        },
        n_iter=20, cv=skf, scoring="roc_auc", random_state=42, n_jobs=-1,
    ),
}

for nome, busca in buscas.items():
    busca.fit(X_train_proc, y_train)
    print(f"\n{nome} — melhor ROC-AUC (CV): {busca.best_score_:.4f}")
    print(f"{nome} — melhores parâmetros: {busca.best_params_}")

melhor_nome = max(buscas, key=lambda n: buscas[n].best_score_)
melhor_modelo = buscas[melhor_nome].best_estimator_
print(f"\nModelo escolhido para o resto do passo: {melhor_nome}")

# %% [markdown]
# ## 3. Tuned vs. não-tuned: overfitting e tamanho do artefato
# Comparamos com os modelos salvos (sem tuning) do Passo 4.

# %%
proba_train_tuned = melhor_modelo.predict_proba(X_train_proc)[:, 1]
proba_test_tuned = melhor_modelo.predict_proba(X_test_proc)[:, 1]

modelo_antigo = joblib.load(f"../models/{melhor_nome}.pkl")
proba_train_antigo = modelo_antigo.predict_proba(X_train_proc)[:, 1]
proba_test_antigo = modelo_antigo.predict_proba(X_test_proc)[:, 1]

joblib.dump(melhor_modelo, "../models/melhor_modelo_tuned.pkl")
tam_antigo_mb = os.path.getsize(f"../models/{melhor_nome}.pkl") / (1024 ** 2)
tam_novo_mb = os.path.getsize("../models/melhor_modelo_tuned.pkl") / (1024 ** 2)

comparacao = pd.DataFrame({
    "sem_tuning": {
        "roc_auc_treino": roc_auc_score(y_train, proba_train_antigo),
        "roc_auc_teste": roc_auc_score(y_test, proba_test_antigo),
        "tamanho_mb": tam_antigo_mb,
    },
    "tuned": {
        "roc_auc_treino": roc_auc_score(y_train, proba_train_tuned),
        "roc_auc_teste": roc_auc_score(y_test, proba_test_tuned),
        "tamanho_mb": tam_novo_mb,
    },
}).round(4)
comparacao.loc["gap_treino_teste"] = [
    comparacao.loc["roc_auc_treino", "sem_tuning"] - comparacao.loc["roc_auc_teste", "sem_tuning"],
    comparacao.loc["roc_auc_treino", "tuned"] - comparacao.loc["roc_auc_teste", "tuned"],
]
print(f"\nComparação ({melhor_nome}, sem tuning vs. tuned):")
print(comparacao.round(4))

# %% [markdown]
# ## 4. Ajuste de threshold por custo de erro
# Suposição de negócio: deixar passar um inadimplente (falso negativo) custa
# **5x mais** que investigar à toa um bom pagador (falso positivo) — proporção
# didática, mas realista em risco de crédito.
#
# Importante: o threshold é escolhido usando probabilidades **fora da amostra
# de treino** (`cross_val_predict`), nunca olhando o teste — senão estaríamos
# vazando informação do teste para uma decisão de modelagem.

# %%
CUSTO_FN = 5
CUSTO_FP = 1

oof_proba = cross_val_predict(melhor_modelo, X_train_proc, y_train, cv=skf,
                               method="predict_proba")[:, 1]

thresholds = np.arange(0.05, 0.96, 0.01)
linhas = []
for t in thresholds:
    pred = (oof_proba >= t).astype(int)
    tp = ((pred == 1) & (y_train == 1)).sum()
    fp = ((pred == 1) & (y_train == 0)).sum()
    fn = ((pred == 0) & (y_train == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    custo = fn * CUSTO_FN + fp * CUSTO_FP
    linhas.append({"threshold": t, "precision": precision, "recall": recall, "custo": custo})

threshold_df = pd.DataFrame(linhas)
melhor_linha = threshold_df.loc[threshold_df["custo"].idxmin()]
threshold_escolhido = round(melhor_linha["threshold"], 2)

print(f"\nThreshold que minimiza custo (fora da amostra): {threshold_escolhido}")
print(f"Nesse ponto — precision: {melhor_linha['precision']:.4f} | recall: {melhor_linha['recall']:.4f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(threshold_df["threshold"], threshold_df["precision"], label="Precision", color="#264653")
axes[0].plot(threshold_df["threshold"], threshold_df["recall"], label="Recall", color="#e76f51")
axes[0].axvline(threshold_escolhido, linestyle="--", color="gray", label=f"Threshold escolhido ({threshold_escolhido})")
axes[0].axvline(0.5, linestyle=":", color="lightgray", label="Threshold padrão (0,5)")
axes[0].set_xlabel("Threshold")
axes[0].set_title("Precision / recall vs. threshold")
axes[0].legend(fontsize=8)

axes[1].plot(threshold_df["threshold"], threshold_df["custo"], color="#e76f51")
axes[1].axvline(threshold_escolhido, linestyle="--", color="gray")
axes[1].scatter([threshold_escolhido], [melhor_linha["custo"]], color="#e76f51", zorder=5)
axes[1].set_xlabel("Threshold")
axes[1].set_title(f"Custo total (FN×{CUSTO_FN} + FP×{CUSTO_FP}) vs. threshold")

plt.tight_layout()
plt.savefig("../data/threshold_tuning.png", dpi=100)
plt.show()

# %% [markdown]
# ## 5. Avaliação final no teste: threshold padrão vs. threshold escolhido
# Agora sim aplicamos no teste — uma única vez, só para reportar o resultado.

# %%
pred_teste_padrao = (proba_test_tuned >= 0.5).astype(int)
pred_teste_escolhido = (proba_test_tuned >= threshold_escolhido).astype(int)

final = pd.DataFrame({
    "threshold_0.5": {
        "precision": precision_score(y_test, pred_teste_padrao),
        "recall": recall_score(y_test, pred_teste_padrao),
        "f1": f1_score(y_test, pred_teste_padrao),
    },
    f"threshold_{threshold_escolhido}": {
        "precision": precision_score(y_test, pred_teste_escolhido),
        "recall": recall_score(y_test, pred_teste_escolhido),
        "f1": f1_score(y_test, pred_teste_escolhido),
    },
}).round(4)
print(f"\nResultado final no teste ({melhor_nome} tuned):")
print(final)
print("\nMatriz de confusão no threshold escolhido:\n", confusion_matrix(y_test, pred_teste_escolhido))

# %% [markdown]
# ## 6. Conclusões do Passo 5
# - Restringir o espaço de busca (sem profundidade ilimitada) já reduz o gap
#   treino-teste e o tamanho do artefato salvo, sem piorar o ROC-AUC de teste.
# - O threshold ideal depende do custo de negócio, não é universal — aqui,
#   com FN custando 5x mais que FP, o corte ótimo fica abaixo de 0,5, o que
#   aumenta o recall às custas de mais falsos positivos (mais gente investigada
#   à toa, mas menos inadimplente escapando).
# - O threshold foi escolhido com dados fora da amostra de treino
#   (`cross_val_predict`) — o teste só foi usado uma vez, no final.
#
# → Próximo: **Passo 6 — Interpretação (feature importance, SHAP).**
