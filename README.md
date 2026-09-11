# Previsão de Inadimplência — Projeto de ML

Classificação binária para prever se um cliente vai ficar inadimplente no próximo mês.
Primeiro projeto da trilha de transição para **Engenheiro de Machine Learning**.

## Dataset
**Default of Credit Card Clients** (UCI, id=350) — 30.000 clientes, 23 features, target binário.
- Target: `default` (1 = inadimplente no próximo mês, 0 = adimplente)
- Desbalanceado: ~22% de inadimplentes (por isso accuracy não serve como métrica principal)

## Estrutura
```
ml-inadimplencia/
├── data/            # datasets (não versionar dados sensíveis)
├── notebooks/       # análise exploratória e experimentos
├── src/             # código reutilizável (pipeline, treino, API)
├── models/          # modelos treinados serializados
└── README.md
```

## Roteiro (o que vamos construir)
- [x] **Passo 0** — Setup do repo + carga de dados
- [x] **Passo 1** — EDA (análise exploratória)
- [x] **Passo 2** — Feature engineering + pré-processamento
- [x] **Passo 3** — Baseline (Logistic Regression) + métricas certas
- [x] **Passo 4** — Modelos (Random Forest, XGBoost) + validação cruzada
- [x] **Passo 5** — Tuning de hiperparâmetros + threshold
- [x] **Passo 6** — Interpretação (feature importance, SHAP)
- [x] **Passo 7** — Deploy (FastAPI + endpoint de previsão)

## Conceitos-chave por passo
| Passo | Conceito de ML que você domina ao terminar |
|-------|--------------------------------------------|
| 1 | Distribuições, correlação, data leakage, desbalanceamento |
| 2 | Encoding, escalonamento, split treino/teste sem vazamento |
| 3 | Por que accuracy engana; precision, recall, ROC-AUC, matriz de confusão |
| 4 | Ensembles, overfitting, cross-validation |
| 5 | Grid/random search, ajuste de threshold por custo de erro |
| 6 | Explicabilidade — requisito real em crédito/risco |
| 7 | Servir modelo em produção via API |

## Setup
```bash
pip install ucimlrepo scikit-learn xgboost pandas matplotlib seaborn shap fastapi uvicorn
```
