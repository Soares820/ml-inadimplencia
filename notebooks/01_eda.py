# %% [markdown]
# # Passo 1 — Análise Exploratória (EDA)
# Previsão de inadimplência de cartão de crédito.
#
# Objetivo deste passo: **entender os dados antes de modelar**. Nenhum modelo ainda.
# Vamos responder: como é o alvo? há valores estranhos? quais variáveis parecem
# prever inadimplência? há risco de data leakage?

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
sns.set_theme(style="whitegrid")

# %% [markdown]
# ## 1. Carga dos dados
# Rode ESTE bloco na sua máquina — ele baixa o dataset real da UCI.
# (No ambiente do Claude a UCI está bloqueada, então lá usamos o sintético.)

# %%
try:
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=350)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    df = df.rename(columns={"Y": "default", "default payment next month": "default"})
    # A API da ucimlrepo devolve as features como X1..X23 em vez dos nomes
    # originais do dataset. Renomeamos para bater com o CSV sintético.
    df = df.rename(columns={
        "X1": "LIMIT_BAL", "X2": "SEX", "X3": "EDUCATION", "X4": "MARRIAGE", "X5": "AGE",
        "X6": "PAY_0", "X7": "PAY_2", "X8": "PAY_3", "X9": "PAY_4", "X10": "PAY_5", "X11": "PAY_6",
        "X12": "BILL_AMT1", "X13": "BILL_AMT2", "X14": "BILL_AMT3",
        "X15": "BILL_AMT4", "X16": "BILL_AMT5", "X17": "BILL_AMT6",
        "X18": "PAY_AMT1", "X19": "PAY_AMT2", "X20": "PAY_AMT3",
        "X21": "PAY_AMT4", "X22": "PAY_AMT5", "X23": "PAY_AMT6",
    })
    print("Dataset REAL da UCI carregado.")
except Exception as e:
    print(f"UCI indisponível ({e}). Usando sintético.")
    df = pd.read_csv("../data/credit_default_SINTETICO.csv")

print("Shape:", df.shape)
df.head()

# %% [markdown]
# ## 2. Sanidade: tipos, nulos, duplicatas

# %%
print(df.info())
print("\nNulos por coluna:\n", df.isnull().sum()[df.isnull().sum() > 0])
print("\nLinhas duplicadas:", df.duplicated().sum())

# %% [markdown]
# ## 3. O alvo (target)
# Sempre comece pelo alvo. Aqui descobrimos o **desbalanceamento** — fato que
# vai ditar quais métricas usar mais adiante (accuracy vai enganar).

# %%
target = "default"
print(df[target].value_counts())
print("Proporção de inadimplentes:", round(df[target].mean(), 4))

fig, ax = plt.subplots(figsize=(5, 4))
df[target].value_counts().plot(kind="bar", ax=ax, color=["#2a9d8f", "#e76f51"])
ax.set_title("Distribuição do alvo (0=adimplente, 1=inadimplente)")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig("../data/eda_target.png", dpi=100)
plt.show()

# %% [markdown]
# ## 4. Estatísticas das numéricas
# Procuramos: escalas muito diferentes (vão exigir escalonamento no Passo 2),
# valores impossíveis, outliers.

# %%
df.describe().T.round(1)

# %% [markdown]
# ## 5. Taxa de inadimplência por variável categórica
# Aqui começamos a ver **quais features têm sinal**. Se a taxa de default muda
# muito entre categorias, a variável provavelmente é preditiva.

# %%
for col in ["SEX", "EDUCATION", "MARRIAGE", "PAY_0"]:
    if col in df.columns:
        taxa = df.groupby(col)[target].mean().round(3)
        print(f"\nTaxa de default por {col}:")
        print(taxa)

# %% [markdown]
# `PAY_0` (status do último pagamento) deve mostrar a taxa subindo conforme o
# atraso aumenta — forte candidato a variável mais importante do modelo.

# %%
if "PAY_0" in df.columns:
    fig, ax = plt.subplots(figsize=(7, 4))
    df.groupby("PAY_0")[target].mean().plot(kind="bar", ax=ax, color="#e76f51")
    ax.set_title("Taxa de inadimplência por status de pagamento (PAY_0)")
    ax.set_ylabel("Taxa de default")
    plt.tight_layout()
    plt.savefig("../data/eda_pay0.png", dpi=100)
    plt.show()

# %% [markdown]
# ## 6. Correlação entre numéricas
# Duas coisas a observar:
# 1. Features muito correlacionadas entre si (ex.: BILL_AMT1..6) → redundância.
# 2. Correlação de cada feature com o alvo → sinal bruto.

# %%
num_cols = df.select_dtypes(include=[np.number]).columns
corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, square=True,
            cbar_kws={"shrink": 0.6})
ax.set_title("Matriz de correlação")
plt.tight_layout()
plt.savefig("../data/eda_corr.png", dpi=100)
plt.show()

print("\nCorrelação de cada feature com o alvo (ordenada):")
print(corr[target].drop(target).sort_values(ascending=False).round(3))

# %% [markdown]
# ## 7. Conclusões da EDA (preencha ao rodar)
# - Alvo desbalanceado (~__%): usaremos precision/recall/ROC-AUC, não accuracy.
# - Variáveis PAY_* parecem as mais preditivas.
# - BILL_AMT1..6 são muito correlacionadas entre si → possível redução no Passo 2.
# - Nenhum nulo relevante / X duplicatas a tratar.
#
# **Data leakage check:** todas as features existem ANTES do mês que queremos
# prever? Sim — são histórico. Nenhuma coluna "vaza" a resposta.
#
# → Próximo: **Passo 2 — Feature engineering + pré-processamento.**
