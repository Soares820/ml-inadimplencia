# Previsão de Inadimplência — Cartão de Crédito

Pipeline completo de Machine Learning para prever se um cliente vai ficar inadimplente
no próximo mês: da análise exploratória ao endpoint de previsão em produção, com
explicabilidade via SHAP.

## Resultados

| | |
|---|---|
| Dataset | UCI Default of Credit Card Clients (id=350) — 30.000 clientes reais, 23 features |
| Taxa de inadimplência | 22,12% (desbalanceado) |
| Modelo final | XGBoost, tunado via `RandomizedSearchCV` + validação cruzada 5-fold |
| ROC-AUC (teste) | 0,776 |
| Recall (threshold ajustado por custo) | 70% dos inadimplentes reais identificados |
| Tamanho do modelo final | 0,34 MB (vs. 182 MB de um Random Forest não regularizado) |
| Explicabilidade | SHAP (importância global + explicação por cliente) |
| Deploy | API REST em FastAPI, endpoint `/predict` |

O ganho mais relevante não veio de trocar de algoritmo, e sim de tratar o
desbalanceamento e ajustar o ponto de corte de decisão: um baseline "ingênuo"
tinha 81% de accuracy mas identificava só 25% dos inadimplentes reais — inútil
para o negócio. O pipeline final troca accuracy por recall de forma deliberada,
com o trade-off justificado por uma suposição de custo (deixar passar um
inadimplente custa ~5x mais que investigar um bom pagador à toa).

## Pipeline

```
Dados brutos (UCI)
      │
      ▼
EDA ──────────────► distribuição do alvo, correlações, checagem de leakage
      │
      ▼
Pré-processamento ─► split estratificado → scaling/encoding ajustados só no treino
      │
      ▼
Modelagem ─────────► Logistic Regression → Random Forest → XGBoost (CV 5-fold)
      │
      ▼
Tuning ────────────► RandomizedSearchCV + ajuste de threshold por custo de erro
      │
      ▼
Interpretação ─────► SHAP (importância global e por cliente)
      │
      ▼
Deploy ────────────► FastAPI /predict
```

## Como rodar

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install ucimlrepo scikit-learn xgboost pandas matplotlib seaborn shap fastapi uvicorn

# Rodar o pipeline completo, na ordem
cd notebooks
python 01_eda.py
python 02_preprocessing.py
python 03_baseline.py
python 04_modelos.py
python 05_tuning.py
python 06_interpretacao.py

# Subir a API
cd ../src
uvicorn api:app --reload
```

## Usando a API

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 3, "MARRIAGE": 1, "AGE": 43,
    "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 39177, "BILL_AMT2": 39607, "BILL_AMT3": 17070,
    "BILL_AMT4": 13038, "BILL_AMT5": 8904, "BILL_AMT6": 4740,
    "PAY_AMT1": 2000, "PAY_AMT2": 1500, "PAY_AMT3": 3500,
    "PAY_AMT4": 600, "PAY_AMT5": 500, "PAY_AMT6": 4000
  }'
# {"probabilidade_default":0.3488,"classificacao":0,"threshold_usado":0.42}
```

## Estrutura

```
ml-inadimplencia/
├── data/            # dataset sintético de fallback, dados processados, gráficos gerados
├── notebooks/        # 01_eda → 06_interpretacao, um arquivo por etapa do pipeline
├── src/api.py        # API FastAPI que serve o modelo final
├── models/           # pipeline de pré-processamento + modelos treinados (.pkl)
└── README.md
```

## O que cada etapa cobre

| Notebook | Conceito de ML |
|---|---|
| `01_eda.py` | Distribuições, correlação, checagem de data leakage, desbalanceamento |
| `02_preprocessing.py` | Encoding, escalonamento, split treino/teste sem vazamento |
| `03_baseline.py` | Por que accuracy engana; precision, recall, ROC-AUC, matriz de confusão |
| `04_modelos.py` | Ensembles (bagging vs. boosting), overfitting, cross-validation |
| `05_tuning.py` | Random search, ajuste de threshold por custo de erro assimétrico |
| `06_interpretacao.py` | Explicabilidade via SHAP — requisito real em crédito/risco |
| `src/api.py` | Servir o modelo treinado em produção via API |

## Stack

Python · pandas · scikit-learn · XGBoost · SHAP · FastAPI · matplotlib/seaborn

## Sobre o desenvolvimento

Este projeto foi construído com assistência de IA (Claude Code) sob minha direção:
defini o escopo, as decisões de negócio (ex.: custo de falso negativo vs. falso
positivo) e revisei cada etapa; a IA escreveu o código e executou o pipeline.
Os commits refletem essa autoria de forma transparente.
