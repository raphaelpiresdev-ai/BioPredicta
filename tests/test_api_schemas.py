"""Suíte de testes para a Fase 4 — Modelagem e Validação Estrita de Dados com Pydantic.

Verifica os 10 requisitos obrigatórios da Fase 4:
1. Instanciação correta do esquema de entrada (PredictionRequest).
2. Instanciação correta do esquema de saída (PredictionResponse).
3. Rejeição de strings vazias ou compostas exclusivamente por espaços.
4. Rejeição de tipos incompatíveis (sem coerções indevidas em strict mode).
5. Rejeição de campos extras não previstos no contrato (extra='forbid').
6. Rejeição de valores fora dos limites de probabilidade hERG (0.0 <= p <= 1.0).
7. Rejeição de valores numéricos não finitos (NaN e infinito).
8. Rejeição de campos obrigatórios ausentes.
9. Funcionamento da serialização e desserialização dos modelos (dump/json).
10. Geração correta do JSON Schema dos contratos (OpenAPI compliance).
"""

from __future__ import annotations

import math
import pytest
from pydantic import ValidationError

from src.api.schemas import PredictionRequest, PredictionResponse


class TestPredictionRequestSchema:
    """Testes unitários para o contrato de entrada PredictionRequest."""

    def test_valid_request_instantiation(self):
        """Requisito 1: Instanciação bem-sucedida com SMILES válido e sanitização de espaços."""
        req = PredictionRequest(smiles="  CN1C=NC2=C1C(=O)N(C(=O)N2C)C  ")
        assert req.smiles == "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"

    def test_rejection_of_empty_string(self):
        """Requisito 3: Rejeição de string vazia."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(smiles="")
        assert "String should have at least 1 character" in str(exc_info.value)

    def test_rejection_of_whitespace_only_string(self):
        """Requisito 3: Rejeição de string contendo exclusivamente espaços."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(smiles="    ")
        assert "String should have at least 1 character" in str(exc_info.value)

    def test_rejection_of_incompatible_types(self):
        """Requisito 4: Rejeição de tipos incompatíveis (números, listas, objetos, nulos)."""
        # Número inteiro
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(smiles=12345)
        assert "Input should be a valid string" in str(exc_info.value)

        # Número float
        with pytest.raises(ValidationError):
            PredictionRequest(smiles=98.6)

        # Lista
        with pytest.raises(ValidationError):
            PredictionRequest(smiles=["CCO"])

        # Dicionário
        with pytest.raises(ValidationError):
            PredictionRequest(smiles={"smiles": "CCO"})

        # Valor nulo (None)
        with pytest.raises(ValidationError):
            PredictionRequest(smiles=None)

    def test_rejection_of_extra_fields(self):
        """Requisito 5: Rejeição de campos adicionais não previstos no contrato."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(smiles="CCO", extra_param="invalido", outro_campo=123)
        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_rejection_of_exceeding_max_length(self):
        """Requisito 1/6: Rejeição de string que ultrapassa o limite máximo de 1000 caracteres."""
        smiles_longo = "C" * 1001
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(smiles=smiles_longo)
        assert "String should have at most 1000 characters" in str(exc_info.value)

        # Limite exato de 1000 caracteres deve ser aceito
        smiles_limite = "C" * 1000
        req = PredictionRequest(smiles=smiles_limite)
        assert len(req.smiles) == 1000

    def test_missing_required_smiles_field(self):
        """Requisito 8: Rejeição quando o campo obrigatório smiles está ausente."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest()
        assert "Field required" in str(exc_info.value)

    def test_request_serialization_and_deserialization(self):
        """Requisito 9: Serialização em dicionário, JSON e desserialização a partir de dict/JSON."""
        req = PredictionRequest(smiles="c1ccccc1")
        dados_dict = req.model_dump()
        assert dados_dict == {"smiles": "c1ccccc1"}

        dados_json = req.model_dump_json()
        assert '{"smiles":"c1ccccc1"}' in dados_json

        # Desserialização a partir de JSON
        req_recuperado = PredictionRequest.model_validate_json('{"smiles": "c1ccccc1"}')
        assert req_recuperado.smiles == "c1ccccc1"

    def test_request_json_schema(self):
        """Requisito 10: Geração correta do JSON Schema para documentação OpenAPI."""
        schema = PredictionRequest.model_json_schema()
        assert schema["type"] == "object"
        assert "smiles" in schema["properties"]
        prop_smiles = schema["properties"]["smiles"]
        assert prop_smiles["type"] == "string"
        assert prop_smiles["minLength"] == 1
        assert prop_smiles["maxLength"] == 1000
        assert "smiles" in schema["required"]


class TestPredictionResponseSchema:
    """Testes unitários para o contrato de saída PredictionResponse."""

    def test_valid_response_instantiation(self):
        """Requisito 2: Instanciação bem-sucedida do esquema de resposta."""
        resp = PredictionResponse(
            smiles="Cc1nc(C)c(C(=O)NC2CCN(Cc3ccccc3)CC2)s1",
            p_ic50=7.42,
            p_herg_positive=0.18,
        )
        assert resp.smiles == "Cc1nc(C)c(C(=O)NC2CCN(Cc3ccccc3)CC2)s1"
        assert resp.p_ic50 == 7.42
        assert resp.p_herg_positive == 0.18

    def test_valid_herg_boundaries(self):
        """Requisito 6: Aceitação rigorosa dos limites de probabilidade hERG (0.0 e 1.0)."""
        resp_min = PredictionResponse(smiles="CCO", p_ic50=5.0, p_herg_positive=0.0)
        assert resp_min.p_herg_positive == 0.0

        resp_max = PredictionResponse(smiles="CCO", p_ic50=5.0, p_herg_positive=1.0)
        assert resp_max.p_herg_positive == 1.0

    def test_rejection_of_herg_out_of_bounds(self):
        """Requisito 6: Rejeição de probabilidades hERG inferiores a zero ou superiores a um."""
        with pytest.raises(ValidationError) as exc_neg:
            PredictionResponse(smiles="CCO", p_ic50=6.0, p_herg_positive=-0.01)
        assert "greater than or equal to 0" in str(exc_neg.value)

        with pytest.raises(ValidationError) as exc_pos:
            PredictionResponse(smiles="CCO", p_ic50=6.0, p_herg_positive=1.001)
        assert "less than or equal to 1" in str(exc_pos.value)

    def test_rejection_of_non_finite_numeric_values(self):
        """Requisito 7: Rejeição de valores NaN e infinitos em campos numéricos."""
        # p_ic50 como NaN
        with pytest.raises(ValidationError):
            PredictionResponse(smiles="CCO", p_ic50=float("nan"), p_herg_positive=0.5)

        # p_ic50 como +Inf
        with pytest.raises(ValidationError):
            PredictionResponse(smiles="CCO", p_ic50=float("inf"), p_herg_positive=0.5)

        # p_ic50 como -Inf
        with pytest.raises(ValidationError):
            PredictionResponse(smiles="CCO", p_ic50=float("-inf"), p_herg_positive=0.5)

        # p_herg_positive como NaN
        with pytest.raises(ValidationError):
            PredictionResponse(smiles="CCO", p_ic50=6.5, p_herg_positive=float("nan"))

    def test_rejection_of_incompatible_types_in_response(self):
        """Requisito 4: Rejeição de strings ou tipos incompatíveis em campos numéricos sob strict mode."""
        with pytest.raises(ValidationError):
            PredictionResponse(smiles="CCO", p_ic50="sete_ponto_dois", p_herg_positive=0.5)

        with pytest.raises(ValidationError):
            PredictionResponse(smiles="CCO", p_ic50=6.5, p_herg_positive="alto")

    def test_rejection_of_extra_fields_in_response(self):
        """Requisito 5: Rejeição de campos extras não previstos na resposta."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(
                smiles="CCO",
                p_ic50=6.5,
                p_herg_positive=0.3,
                classificacao_texto="Moderado",  # Campo proibido
            )
        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_rejection_of_missing_required_fields_in_response(self):
        """Requisito 8: Rejeição de resposta com qualquer campo obrigatório ausente."""
        # Ausência de smiles
        with pytest.raises(ValidationError):
            PredictionResponse(p_ic50=6.5, p_herg_positive=0.3)

        # Ausência de p_ic50
        with pytest.raises(ValidationError):
            PredictionResponse(smiles="CCO", p_herg_positive=0.3)

        # Ausência de p_herg_positive
        with pytest.raises(ValidationError):
            PredictionResponse(smiles="CCO", p_ic50=6.5)

    def test_response_serialization_and_deserialization(self):
        """Requisito 9: Serialização em dicionário, JSON e validação."""
        resp = PredictionResponse(smiles="CCO", p_ic50=6.12, p_herg_positive=0.45)
        dump = resp.model_dump()
        assert dump == {"smiles": "CCO", "p_ic50": 6.12, "p_herg_positive": 0.45}

        json_str = resp.model_dump_json()
        recuperado = PredictionResponse.model_validate_json(json_str)
        assert recuperado.smiles == resp.smiles
        assert recuperado.p_ic50 == resp.p_ic50
        assert recuperado.p_herg_positive == resp.p_herg_positive

    def test_response_json_schema(self):
        """Requisito 10: Geração correta do JSON Schema da resposta."""
        schema = PredictionResponse.model_json_schema()
        assert schema["type"] == "object"
        assert set(schema["required"]) == {"smiles", "p_ic50", "p_herg_positive"}

        props = schema["properties"]
        assert props["smiles"]["type"] == "string"
        assert props["p_ic50"]["type"] == "number"
        assert props["p_herg_positive"]["type"] == "number"
        assert props["p_herg_positive"]["minimum"] == 0.0
        assert props["p_herg_positive"]["maximum"] == 1.0

