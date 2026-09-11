"""
Passo 7 — Deploy (FastAPI + endpoint de previsão)

Serve o pipeline de pré-processamento (Passo 2) + modelo tuned (Passo 5) atrás
de um endpoint HTTP. O cliente manda os dados brutos; a API aplica exatamente
as mesmas transformações do treino e devolve a probabilidade de default.

Rodar localmente:
    uvicorn api:app --reload --app-dir src
"""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
THRESHOLD = 0.42  # escolhido no Passo 5 (custo FN = 5x custo FP), fora da amostra de teste

app = FastAPI(
    title="API de Previsão de Inadimplência",
    description="Recebe os dados de um cliente e devolve a probabilidade de inadimplência no próximo mês.",
    version="1.0.0",
)

pipeline = joblib.load(BASE_DIR / "models" / "preprocessor.pkl")
modelo = joblib.load(BASE_DIR / "models" / "melhor_modelo_tuned.pkl")


class Cliente(BaseModel):
    LIMIT_BAL: float = Field(..., description="Limite de crédito concedido")
    SEX: int = Field(..., description="1 = masculino, 2 = feminino")
    EDUCATION: int = Field(..., description="1 = pós-graduação, 2 = graduação, 3 = ensino médio, 4 = outros")
    MARRIAGE: int = Field(..., description="1 = casado, 2 = solteiro, 3 = outros")
    AGE: int
    PAY_0: int = Field(..., description="Status do pagamento no mês mais recente (-2..8)")
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float

    class Config:
        json_schema_extra = {
            "example": {
                "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 3, "MARRIAGE": 1, "AGE": 43,
                "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
                "BILL_AMT1": 39177, "BILL_AMT2": 39607, "BILL_AMT3": 17070,
                "BILL_AMT4": 13038, "BILL_AMT5": 8904, "BILL_AMT6": 4740,
                "PAY_AMT1": 2000, "PAY_AMT2": 1500, "PAY_AMT3": 3500,
                "PAY_AMT4": 600, "PAY_AMT5": 500, "PAY_AMT6": 4000,
            }
        }


class Previsao(BaseModel):
    probabilidade_default: float
    classificacao: int
    threshold_usado: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Previsao)
def predict(cliente: Cliente):
    df = pd.DataFrame([cliente.model_dump()])
    X_proc = pipeline.transform(df)
    proba = float(modelo.predict_proba(X_proc)[0, 1])
    return Previsao(
        probabilidade_default=round(proba, 4),
        classificacao=int(proba >= THRESHOLD),
        threshold_usado=THRESHOLD,
    )
