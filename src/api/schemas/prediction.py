"""Modelos e contratos de dados para predição molecular (Pydantic).

Define os esquemas formais de entrada (PredictionRequest) e saída (PredictionResponse)
para a futura rota de inferência multiobjetivo (A2A e hERG), aplicando validação
estrita de tipos, restrições estruturais de fronteira e documentação OpenAPI.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Contrato de dados de entrada para solicitação de predição molecular.

    Recebe a representação linear estrutural de uma única molécula via SMILES.
    Nesta etapa, realiza validação estrita de contrato (tipo string, preenchimento,
    limites de tamanho e higienização de espaços). A validação quimioinformática
    (RDKit) ocorre na camada de serviço/inferência.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        strict=True,
    )

    smiles: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description=(
            "Representação linear SMILES da molécula a ser analisada. "
            "Deve ser uma string não vazia. O limite de 1000 caracteres acomoda com folga "
            "estruturas de moléculas pequenas e fármacos comerciais enquanto protege o "
            "servidor contra payloads excessivos e vulnerabilidades de ReDoS."
        ),
        examples=["Cc1nc(C)c(C(=O)NC2CCN(Cc3ccccc3)CC2)s1"],
    )


class PredictionResponse(BaseModel):
    """Contrato de dados de saída com as predições multiobjetivo do MultiTarget BioPredict.

    Retorna as estimativas computacionais geradas pelos modelos serializados:
    1. Potência no receptor purinérgico A2A (pIC50, regressão contínua).
    2. Probabilidade de bloqueio do canal de potássio cardíaco hERG (classificação binária).
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )

    smiles: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description=(
            "Representação SMILES da molécula avaliada, retornada para permitir "
            "rastreabilidade e correlação inequívoca entre a solicitação e a resposta."
        ),
        examples=["Cc1nc(C)c(C(=O)NC2CCN(Cc3ccccc3)CC2)s1"],
    )

    p_ic50: float = Field(
        ...,
        description=(
            "Estimativa da potência de afinidade da molécula pelo receptor purinérgico A2A, "
            "expressa em escala logarítmica pIC50 (-log10 IC50 em Molar). "
            "Calculada pelo modelo de regressão XGBoost homologado."
        ),
        examples=[6.75],
    )

    p_herg_positive: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Probabilidade estimada pelo modelo de classificação Random Forest de que o composto "
            "seja classificado como hERG positivo (bloqueador do canal hERG). "
            "Nota: representa estritamente a probabilidade estatística calculada pelo algoritmo de "
            "Machine Learning a partir de fingerprints e descritores físico-químicos, e NÃO uma "
            "probabilidade clínica direta de arritmia ou cardiotoxicidade em humanos."
        ),
        examples=[0.25],
    )

