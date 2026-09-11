# %% [markdown]
# # Passo 3 — Baseline (Logistic Regression) + métricas certas
# Previsão de inadimplência de cartão de crédito.
#
# Objetivo: treinar o primeiro modelo e entender por que **accuracy sozinha
# engana** num problema desbalanceado (~22% de inadimplentes). Vamos comparar
# um baseline ingênuo com um baseline que trata o desbalanceamento.

# %%
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)

# %% [markdown]
# ## 1. Carregar dados processados no Passo 2
# Reaproveitamos o split e o pipeline já ajustados — nada de limpeza ou fit
# de scaler aqui, para não repetir trabalho nem arriscar vazamento.

# %%
train_df = pd.read_csv("../data/processed/train.csv")
test_df = pd.read_csv("../data/processed/test.csv")
pipeline = joblib.load("../models/preprocessor.pkl")

target = "default"
X_train, y_train = train_df.drop(columns=[target]), train_df[target]
X_test, y_test = test_df.drop(columns=[target]), test_df[target]

X_train_proc = pipeline.transform(X_train)
X_test_proc = pipeline.transform(X_test)

print("Treino:", X_train_proc.shape, "| Teste:", X_test_proc.shape)

# %% [markdown]
# ## 2. Baseline ingênuo (sem tratar desbalanceamento)

# %%
modelo_ingenuo = LogisticRegression(max_iter=1000, random_state=42)
modelo_ingenuo.fit(X_train_proc, y_train)
pred_ingenuo = modelo_ingenuo.predict(X_test_proc)

print("Accuracy:", round(accuracy_score(y_test, pred_ingenuo), 4))
print("Recall (classe 1 = inadimplente):", round(recall_score(y_test, pred_ingenuo), 4))
print("\nMatriz de confusão:\n", confusion_matrix(y_test, pred_ingenuo))

# %% [markdown]
# ## 3. Por que accuracy engana aqui
# Repare: a accuracy fica próxima da proporção de adimplentes (~78%) porque o
# modelo ingênuo acerta a maioria só "chutando" a classe majoritária — e quase
# não identifica quem realmente vai inadimplir (recall baixo da classe 1).
# Num caso de crédito, isso é o pior erro possível: deixar passar o risco.

# %% [markdown]
# ## 4. Baseline balanceado (`class_weight="balanced"`)
# Compensa o desbalanceamento penalizando mais os erros na classe minoritária.

# %%
modelo_balanceado = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
modelo_balanceado.fit(X_train_proc, y_train)
pred_balanceado = modelo_balanceado.predict(X_test_proc)
proba_balanceado = modelo_balanceado.predict_proba(X_test_proc)[:, 1]

resultados = pd.DataFrame({
    "ingênuo": {
        "accuracy": accuracy_score(y_test, pred_ingenuo),
        "precision": precision_score(y_test, pred_ingenuo),
        "recall": recall_score(y_test, pred_ingenuo),
        "f1": f1_score(y_test, pred_ingenuo),
        "roc_auc": roc_auc_score(y_test, modelo_ingenuo.predict_proba(X_test_proc)[:, 1]),
    },
    "balanceado": {
        "accuracy": accuracy_score(y_test, pred_balanceado),
        "precision": precision_score(y_test, pred_balanceado),
        "recall": recall_score(y_test, pred_balanceado),
        "f1": f1_score(y_test, pred_balanceado),
        "roc_auc": roc_auc_score(y_test, proba_balanceado),
    },
}).round(4)

print(resultados)
print("\nRelatório completo (balanceado):\n", classification_report(y_test, pred_balanceado))

# %% [markdown]
# ## 5. Matriz de confusão e curva ROC (modelo balanceado)

# %%
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, pred_balanceado), annot=True, fmt="d",
            cmap="Blues", ax=ax, xticklabels=["Pred. 0", "Pred. 1"],
            yticklabels=["Real 0", "Real 1"])
ax.set_title("Matriz de confusão — baseline balanceado")
plt.tight_layout()
plt.savefig("../data/baseline_confusion.png", dpi=100)
plt.show()

# %%
fpr, tpr, _ = roc_curve(y_test, proba_balanceado)
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(fpr, tpr, color="#e76f51", label=f"AUC = {roc_auc_score(y_test, proba_balanceado):.3f}")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax.set_xlabel("Falso positivo (1 - especificidade)")
ax.set_ylabel("Verdadeiro positivo (recall)")
ax.set_title("Curva ROC — baseline balanceado")
ax.legend()
plt.tight_layout()
plt.savefig("../data/baseline_roc.png", dpi=100)
plt.show()

# %% [markdown]
# ## 6. Salvar o modelo baseline

# %%
joblib.dump(modelo_balanceado, "../models/baseline_logreg.pkl")
print("Modelo salvo em models/baseline_logreg.pkl")

# %% [markdown]
# ## 7. Conclusões do Passo 3
# - Accuracy do ingênuo parece boa, mas o recall da classe 1 é péssimo — ele
#   erra justamente quem inadimple.
# - O balanceado troca um pouco de accuracy/precision por muito mais recall:
#   é a troca certa aqui, porque deixar passar um inadimplente custa mais caro
#   que investigar um adimplente por engano.
# - ROC-AUC dá uma visão de qualidade do modelo independente do threshold —
#   útil para comparar modelos nos próximos passos.
#
# → Próximo: **Passo 4 — Modelos (Random Forest, XGBoost) + validação cruzada.**
