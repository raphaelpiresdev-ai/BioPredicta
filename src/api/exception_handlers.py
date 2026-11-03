"""Módulo centralizado de tratamento e tradução de exceções da API FastAPI.

Padroniza as respostas de erro de validação (RequestValidationError e JSONDecodeError)
para o idioma português brasileiro, preservando os códigos HTTP originais (HTTP 422),
os identificadores técnicos de tipo de erro ('type') e os caminhos de localização ('loc').
"""

from __future__ import annotations

import logging
from json.decoder import JSONDecodeError
from typing import Any, Dict, List

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def translate_validation_error(err: Dict[str, Any]) -> Dict[str, Any]:
    """Traduz e higieniza um item individual de erro de validação do Pydantic para português brasileiro.

    Preserva 'loc' e 'type', traduz 'msg' e omite 'input' para proteger dados sensíveis.
    """
    err_type = str(err.get("type", ""))
    loc = list(err.get("loc", []))

    # Identifica o nome do campo a partir da hierarquia loc (geralmente o último elemento string)
    field_name = None
    for item in reversed(loc):
        if isinstance(item, str) and item != "body":
            field_name = item
            break

    # 1. JSON malformado / sintaxe incorreta
    if err_type in ("json_invalid", "json_type"):
        msg = "JSON inválido. Verifique a sintaxe da requisição e utilize aspas duplas nos valores de texto."
        clean_loc = ["body"]

    # 2. Campo obrigatório ausente
    elif err_type == "missing":
        if field_name:
            msg = f"O campo '{field_name}' é obrigatório."
        else:
            msg = "Campo obrigatório ausente."
        clean_loc = loc

    # 3. Tipo incompatível: string esperada
    elif err_type == "string_type":
        if field_name:
            msg = f"O campo '{field_name}' deve ser uma string."
        else:
            msg = "O valor deve ser uma string."
        clean_loc = loc

    # 4. Tipo incompatível: número esperado (int ou float)
    elif err_type in ("int_type", "float_type", "number_type"):
        if field_name:
            msg = f"O campo '{field_name}' deve ser um número."
        else:
            msg = "O valor deve ser um número."
        clean_loc = loc

    # 5. Campo adicional não previsto no contrato (extra='forbid')
    elif err_type == "extra_forbidden":
        if field_name:
            msg = f"O campo adicional '{field_name}' não é permitido."
        else:
            msg = "Campo adicional não permitido."
        clean_loc = loc

    # 6. String vazia ou contendo somente espaços
    elif err_type == "string_too_short":
        raw_input = err.get("input")
        if isinstance(raw_input, str) and (not raw_input or raw_input.strip() == ""):
            if field_name:
                msg = f"O campo '{field_name}' não pode ser vazio ou conter apenas espaços."
            else:
                msg = "O valor não pode ser vazio ou conter apenas espaços."
        else:
            min_len = err.get("ctx", {}).get("min_length", 1)
            if field_name:
                msg = f"O campo '{field_name}' deve ter no mínimo {min_len} caractere(s)."
            else:
                msg = f"O valor deve ter no mínimo {min_len} caractere(s)."
        clean_loc = loc

    # 7. String excedendo o limite máximo de caracteres
    elif err_type == "string_too_long":
        max_len = err.get("ctx", {}).get("max_length", 1000)
        if field_name:
            msg = f"O campo '{field_name}' excede o comprimento máximo permitido de {max_len} caracteres."
        else:
            msg = f"O valor excede o comprimento máximo permitido de {max_len} caracteres."
        clean_loc = loc

    # 8. Limites numéricos (maior ou igual, menor ou igual)
    elif err_type in ("greater_than_equal", "greater_than"):
        ge_val = err.get("ctx", {}).get("ge")
        if field_name:
            msg = f"O campo '{field_name}' deve ser maior ou igual a {ge_val}."
        else:
            msg = f"O valor deve ser maior ou igual a {ge_val}."
        clean_loc = loc
    elif err_type in ("less_than_equal", "less_than"):
        le_val = err.get("ctx", {}).get("le")
        if field_name:
            msg = f"O campo '{field_name}' deve ser menor ou igual a {le_val}."
        else:
            msg = f"O valor deve ser menor ou igual a {le_val}."
        clean_loc = loc

    # 9. Número não finito (NaN ou infinito)
    elif err_type == "finite_number":
        if field_name:
            msg = f"O campo '{field_name}' deve ser um número finito (não são aceitos NaN ou infinito)."
        else:
            msg = "O valor deve ser um número finito."
        clean_loc = loc

    # 10. Fallback seguro para qualquer outro tipo de erro
    else:
        if field_name:
            msg = f"Valor inválido fornecido para o campo '{field_name}'."
        else:
            msg = "Valor inválido fornecido na requisição."
        clean_loc = loc

    return {
        "loc": clean_loc,
        "msg": msg,
        "type": err_type,
    }


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Manipulador centralizado para erros de validação de requisição (RequestValidationError).

    Converte a lista de erros técnicos do Pydantic em mensagens amigáveis em português brasileiro,
    higienizando os logs internos para omitir entradas moleculares brutas e mantendo HTTP 422.
    """
    translated = [translate_validation_error(err) for err in exc.errors()]
    diagnostic_summary = [{"loc": item["loc"], "type": item["type"]} for item in translated]
    logger.warning(
        "Falha de validação da requisição em %s %s: %s",
        request.method,
        request.url.path,
        diagnostic_summary,
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": translated},
    )


async def json_decode_exception_handler(
    request: Request,
    exc: JSONDecodeError,
) -> JSONResponse:
    """Manipulador para erros de decodificação sintática direta de JSON (JSONDecodeError).

    Garante que payloads JSON malformados recebam mensagem clara em português com HTTP 422,
    sem registrar o conteúdo bruto da requisição nos logs.
    """
    logger.warning(
        "JSON malformado recebido em %s %s (linha=%d, coluna=%d): %s",
        request.method,
        request.url.path,
        exc.lineno,
        exc.colno,
        exc.msg,
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={
            "detail": [
                {
                    "loc": ["body"],
                    "msg": "JSON inválido. Verifique a sintaxe da requisição e utilize aspas duplas nos valores de texto.",
                    "type": "json_invalid",
                }
            ]
        },
    )

