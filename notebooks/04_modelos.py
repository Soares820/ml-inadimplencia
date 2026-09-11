# %% [markdown]
# # Passo 4 — Modelos (Random Forest, XGBoost) + Validação Cruzada
# Previsão de inadimplência de cartão de crédito.
#
# Objetivo: sair de um único modelo linear (Passo 3) para ensembles, e validar
# com k-fold em vez de confiar num único split treino/teste — que pode ser
# "sortudo" ou "azarado".

# %%
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, roc_curve

# %% [markdown]
# ## 1. Carregar dados processados (mesma lógica do Passo 3)

# %%
train_df = pd.read_csv("../data/processed/train.csv")
test_df = pd.read_csv("../data/processed/test.csv")
pipeline = joblib.load("../models/preprocessor.pkl")

target = "default"
X_train, y_train = train_df.drop(columns=[target]), train_df[target]
X_test, y_test = test_df.drop(columns=[target]), test_df[target]

X_train_proc = pipeline.transform(X_train)
X_test_proc = pipeline.transform(X_test)
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print("Treino:", X_train_proc.shape, "| Teste:", X_test_proc.shape)
print("scale_pos_weight (proporção 0/1 no treino):", round(scale_pos_weight, 3))

# %% [markdown]
# ## 2. Validação cruzada estratificada (5-fold) no treino
# Em vez de um único split, revezamos qual pedaço do treino vira validação —
# a média das 5 rodadas é uma estimativa mais confiável do que um número só.
# Todos os três modelos tratam o desbalanceamento (`class_weight`/`scale_pos_weight`).

# %%
modelos = {
    "logistic": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
    "xgboost": XGBClassifier(
        n_estimators=300, scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", random_state=42,
    ),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_resultados = {}
for nome, modelo in modelos.items():
    scores = cross_validate(modelo, X_train_proc, y_train, cv=skf,
                             scoring=["roc_auc", "recall", "f1"])
    cv_resultados[nome] = {
        "roc_auc": f'{scores["test_roc_auc"].mean():.4f} ± {scores["test_roc_auc"].std():.4f}',
        "recall": f'{scores["test_recall"].mean():.4f} ± {scores["test_recall"].std():.4f}',
        "f1": f'{scores["test_f1"].mean():.4f} ± {scores["test_f1"].std():.4f}',
    }

print("\nValidação cruzada (5-fold, média ± desvio-padrão):")
print(pd.DataFrame(cv_resultados).T)

# %% [markdown]
# ## 3. Treinar no treino completo e avaliar no teste (holdout)
# Mesmo conjunto de teste do Passo 3 — dá pra comparar direto com o baseline.

# %%
resultados_teste = {}
resultados_treino = {}
probas_teste = {}

for nome, modelo in modelos.items():
    modelo.fit(X_train_proc, y_train)
    proba_treino = modelo.predict_proba(X_train_proc)[:, 1]
    proba_teste = modelo.predict_proba(X_test_proc)[:, 1]
    probas_teste[nome] = proba_teste
    resultados_treino[nome] = roc_auc_score(y_train, proba_treino)
    resultados_teste[nome] = roc_auc_score(y_test, proba_teste)

overfit_df = pd.DataFrame({
    "roc_auc_treino": resultados_treino,
    "roc_auc_teste": resultados_teste,
}).round(4)
overfit_df["gap"] = (overfit_df["roc_auc_treino"] - overfit_df["roc_auc_teste"]).round(4)
print("\nROC-AUC treino vs. teste (gap grande = sinal de overfitting):")
print(overfit_df)

# %% [markdown]
# ## 4. Curvas ROC sobrepostas (teste)

# %%
fig, ax = plt.subplots(figsize=(6, 5))
cores = {"logistic": "#264653", "random_forest": "#2a9d8f", "xgboost": "#e76f51"}
for nome, proba in probas_teste.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, label=f"{nome} (AUC={auc:.3f})", color=cores[nome])
ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax.set_xlabel("Falso positivo (1 - especificidade)")
ax.set_ylabel("Verdadeiro positivo (recall)")
ax.set_title("Curva ROC — comparação dos 3 modelos")
ax.legend()
plt.tight_layout()
plt.savefig("../data/modelos_roc_comparacao.png", dpi=100)
plt.show()

# %% [markdown]
# ## 5. Importância de features (prévia do Passo 6 — SHAP)
# Importância nativa do XGBoost: quanto cada feature contribuiu para reduzir o
# erro nas árvores. É uma visão global e rápida — o SHAP no Passo 6 vai além,
# explicando previsões individuais.

# %%
importancias = pd.Series(modelos["xgboost"].feature_importances_, index=feature_names)
top10 = importancias.sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(7, 5))
top10.sort_values().plot(kind="barh", ax=ax, color="#e76f51")
ax.set_title("Top 10 features — importância no XGBoost")
ax.set_xlabel("Importância")
plt.tight_layout()
plt.savefig("../data/modelos_feature_importance.png", dpi=100)
plt.show()

print("\nTop 10 features (XGBoost):")
print(top10.sort_values(ascending=False).round(4))

# %% [markdown]
# ## 6. Salvar os modelos

# %%
joblib.dump(modelos["random_forest"], "../models/random_forest.pkl")
joblib.dump(modelos["xgboost"], "../models/xgboost.pkl")
print("\nModelos salvos em models/random_forest.pkl e models/xgboost.pkl")

# %% [markdown]
# ## 7. Conclusões do Passo 4
# - A validação cruzada confirma que os ensembles (RF, XGBoost) batem a
#   Regressão Logística em ROC-AUC — as árvores capturam interações entre
#   variáveis (ex.: PAY_0 combinado com LIMIT_BAL) que um modelo linear não vê.
# - O gap treino-teste do Random Forest/XGBoost tende a ser maior que o da
#   Regressão Logística — sinal de que esses modelos overfitam mais fácil,
#   o que motiva o tuning de hiperparâmetros no próximo passo.
# - PAY_0 continua no topo da importância — consistente com a EDA (Passo 1) e
#   a correlação mais forte com o alvo.
#
# → Próximo: **Passo 5 — Tuning de hiperparâmetros + ajuste de threshold.**
