# %% [markdown]
# # Passo 2 — Feature Engineering + Pré-processamento
# Previsão de inadimplência de cartão de crédito.
#
# Objetivo: preparar os dados para modelagem, sem vazar informação do teste
# para o treino. Usamos os achados do Passo 1 (EDA):
# - 35 linhas duplicadas → remover.
# - `EDUCATION` tem categorias fora do dicionário oficial (0, 5, 6) → agrupar.
# - `MARRIAGE` tem categoria fora do dicionário oficial (0) → agrupar.
# - Alvo desbalanceado (~22%) → split precisa ser estratificado.

# %%
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# %% [markdown]
# ## 1. Carga dos dados (mesma lógica do Passo 1)

# %%
try:
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=350)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    df = df.rename(columns={"Y": "default", "default payment next month": "default"})
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

print("Shape bruto:", df.shape)

# %% [markdown]
# ## 2. Limpeza das anomalias encontradas na EDA

# %%
target = "default"

antes = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Duplicatas removidas: {antes - len(df)}")

# Dicionário oficial: EDUCATION 1=graduate,2=university,3=high school,4=others.
# 0, 5, 6 não são documentados -> agrupamos em 4 (others).
if "EDUCATION" in df.columns:
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

# Dicionário oficial: MARRIAGE 1=married,2=single,3=others. 0 não é documentado.
if "MARRIAGE" in df.columns:
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

print("Shape após limpeza:", df.shape)

# %% [markdown]
# ## 3. Split treino/teste (estratificado, ANTES de qualquer scaling/encoding)
# Ajustar scaler/encoder no dataset inteiro vazaria estatísticas do teste para
# o treino. Por isso o split vem primeiro.

# %%
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Treino:", X_train.shape, "| Teste:", X_test.shape)
print("Taxa de default (treino):", round(y_train.mean(), 4))
print("Taxa de default (teste):", round(y_test.mean(), 4))

# %% [markdown]
# ## 4. Pipeline de pré-processamento
# - Numéricas (incluindo `PAY_0..PAY_6`, que são ordinais): `StandardScaler`.
# - Categóricas nominais (`SEX`, `EDUCATION`, `MARRIAGE`): `OneHotEncoder`.
# O pipeline é ajustado (`fit`) só no treino e aplicado (`transform`) nos dois.

# %%
cat_cols = [c for c in ["SEX", "EDUCATION", "MARRIAGE"] if c in X.columns]
num_cols = [c for c in X.columns if c not in cat_cols]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

X_train_proc = pipeline.fit_transform(X_train)
X_test_proc = pipeline.transform(X_test)

print("Shape treino pós-processamento:", X_train_proc.shape)
print("Shape teste pós-processamento:", X_test_proc.shape)

# %% [markdown]
# ## 5. Salvar artefatos para o Passo 3
# Guardamos os splits limpos (antes do encoding, legíveis) e o pipeline
# ajustado — assim o Passo 3 não repete a limpeza nem re-ajusta o scaler.

# %%
X_train.assign(**{target: y_train}).to_csv("../data/processed/train.csv", index=False)
X_test.assign(**{target: y_test}).to_csv("../data/processed/test.csv", index=False)
joblib.dump(pipeline, "../models/preprocessor.pkl")

print("Salvos: data/processed/train.csv, data/processed/test.csv, models/preprocessor.pkl")

# %% [markdown]
# ## 6. Conclusões do Passo 2
# - 35 duplicatas removidas; `EDUCATION`/`MARRIAGE` com categorias fora do
#   dicionário oficial agrupadas em "outros".
# - Split estratificado preserva a taxa de ~22% de inadimplência em treino e
#   teste — sem vazamento, porque o scaler/encoder foi ajustado só no treino.
# - Pipeline salvo em `models/preprocessor.pkl` para reuso nos próximos passos.
#
# → Próximo: **Passo 3 — Baseline (Logistic Regression) + métricas certas.**
