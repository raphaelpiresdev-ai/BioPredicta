"""
MultiTarget BioPredict — Plataforma de Triagem Virtual Multiobjetivo (A2A × hERG)
Aplicação Web Interativa em Streamlit com Explicabilidade Química SHAP (RDKit Morgan bitInfo)
e Editor Molecular Bidimensional Ketcher.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D

from src.chem_utils import (
    DESCRIPTOR_NAMES,
    FEATURE_NAMES,
    FINGERPRINT_NAMES,
    canonicalize_smiles,
    compute_2d_descriptors,
    featurize_dataframe,
)
from src.ad_utils import (
    build_ad_section_html,
    compute_ad_diagnostics,
    get_ad_reference,
)

# -----------------------------------------------------------------------------
# 1. Configuração da Página
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MultiTarget BioPredict | A2A × hERG",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_html(html_str: str) -> None:
    """Renderiza HTML customizado sem indentação de 4 espaços para evitar blocos <pre><code> no Markdown."""
    clean_html = "\n".join(line.strip() for line in html_str.strip().splitlines() if line.strip())
    st.markdown(clean_html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Estilização CSS Customizada (Dark Mode: #0E1117 / #1E222D / #00D2BE / #FFFFFF)
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg-primary: #0E1117;
    --bg-card: #1E222D;
    --bg-card-elevated: #252A37;
    --accent-teal: #00D2BE;
    --accent-teal-glow: rgba(0, 210, 190, 0.16);
    --text-main: #FFFFFF;
    --text-muted: #CBD5E1;
    --text-subtle: #94A3B8;
    --danger-red: #FF4B4B;
    --warning-amber: #F59E0B;
    --border-subtle: rgba(255, 255, 255, 0.10);
}

html, body, .stApp {
    background-color: var(--bg-primary);
    color: var(--text-main);
    font-family: 'Inter', sans-serif;
}

section[data-testid="stSidebar"] {
    background-color: var(--bg-card) !important;
    border-right: 1px solid var(--border-subtle);
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif;
    color: var(--text-main) !important;
    letter-spacing: -0.02em;
}

/* Cabeçalho Principal */
.hero-banner {
    background: linear-gradient(135deg, rgba(30, 34, 45, 0.95) 0%, rgba(14, 17, 23, 0.98) 100%);
    border: 1px solid rgba(0, 210, 190, 0.28);
    border-left: 5px solid #00D2BE;
    border-radius: 14px;
    padding: 1.4rem 1.75rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.45);
}

.main-title {
    font-family: 'Outfit', sans-serif;
    font-size: 2.35rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00D2BE 0%, #38BDF8 55%, #FFFFFF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
    line-height: 1.15;
}

.sub-title {
    font-size: 1.02rem;
    color: #E2E8F0;
    margin-bottom: 0;
    font-weight: 400;
    line-height: 1.5;
}

/* Cartões Principais (metric-card) */
.metric-card {
    background-color: #1E222D;
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 14px;
    padding: 1.35rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35);
    transition: border-color 0.2s ease, transform 0.2s ease;
}

.metric-card:hover {
    border-color: rgba(0, 210, 190, 0.40);
}

.card-header-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.08rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 0.55rem;
}

.kpi-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    margin: 0.9rem 0;
}

.kpi-box {
    background-color: #141820;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 0.75rem 0.9rem;
}

.kpi-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94A3B8;
    font-weight: 600;
    margin-bottom: 0.2rem;
}

.kpi-value {
    font-family: 'Outfit', sans-serif;
    font-size: 1.55rem;
    font-weight: 700;
    color: #FFFFFF;
}

.kpi-value-accent {
    color: #00D2BE;
}

/* Semáforos Independentes */
.semaphore-green {
    background: rgba(0, 210, 190, 0.14);
    border: 1.5px solid #00D2BE;
    color: #00F5D4;
    padding: 0.48rem 1rem;
    border-radius: 9999px;
    font-weight: 700;
    display: inline-block;
    font-size: 0.98rem;
    letter-spacing: 0.01em;
}

.semaphore-yellow {
    background: rgba(245, 158, 11, 0.15);
    border: 1.5px solid #F59E0B;
    color: #FBBF24;
    padding: 0.48rem 1rem;
    border-radius: 9999px;
    font-weight: 700;
    display: inline-block;
    font-size: 0.98rem;
    letter-spacing: 0.01em;
}

.semaphore-red {
    background: rgba(255, 75, 75, 0.15);
    border: 1.5px solid #FF4B4B;
    color: #FF7676;
    padding: 0.48rem 1rem;
    border-radius: 9999px;
    font-weight: 700;
    display: inline-block;
    font-size: 0.98rem;
    letter-spacing: 0.01em;
}

/* Caixa de Aviso de Amostragem SHAP (Top 10) */
.xai-audit-banner {
    background: linear-gradient(90deg, rgba(0, 210, 190, 0.12) 0%, rgba(30, 34, 45, 0.95) 100%);
    border: 1px solid rgba(0, 210, 190, 0.45);
    border-left: 5px solid #00D2BE;
    border-radius: 12px;
    padding: 1rem 1.35rem;
    margin: 0.75rem 0 1.25rem 0;
    color: #FFFFFF;
    font-size: 0.95rem;
    line-height: 1.55;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
}

/* Cartões de Farmacóforos e Descritores (XAI Grid) */
.pharmacophore-card {
    background-color: #1E222D;
    border: 1px solid rgba(255, 255, 255, 0.11);
    border-radius: 12px;
    padding: 1.15rem 1.25rem;
    margin-bottom: 1rem;
    min-height: 220px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.30);
}

.pharmacophore-card-pos {
    border-top: 3px solid #00D2BE;
}

.pharmacophore-card-neg {
    border-top: 3px solid #FF4B4B;
}

.pharm-card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 0.5rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 0.65rem;
    margin-bottom: 0.8rem;
}

.pharm-feature-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.84rem;
    font-weight: 600;
    color: #00D2BE;
    background: rgba(0, 210, 190, 0.12);
    padding: 0.16rem 0.55rem;
    border-radius: 6px;
    border: 1px solid rgba(0, 210, 190, 0.30);
    display: inline-block;
    margin-bottom: 0.3rem;
}

.pharm-feature-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.02rem;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1.25;
}

.shap-tag-pos {
    font-family: 'JetBrains Mono', monospace;
    background: rgba(0, 210, 190, 0.16);
    border: 1px solid #00D2BE;
    color: #00F5D4;
    padding: 0.28rem 0.65rem;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 700;
    white-space: nowrap;
}

.shap-tag-neg {
    font-family: 'JetBrains Mono', monospace;
    background: rgba(255, 75, 75, 0.16);
    border: 1px solid #FF4B4B;
    color: #FF8080;
    padding: 0.28rem 0.65rem;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 700;
    white-space: nowrap;
}

.pharm-body-present {
    display: flex;
    gap: 0.95rem;
    align-items: center;
}

/* Container Dark/Transparente com Tamanho Controlado para o Fragmento RDKit */
.bit-img-wrapper {
    background: rgba(14, 17, 23, 0.78);
    border: 1.5px solid rgba(0, 210, 190, 0.50);
    border-radius: 10px;
    padding: 6px;
    width: 140px;
    min-width: 140px;
    max-width: 140px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    box-shadow: inset 0 0 12px rgba(0, 0, 0, 0.45);
}

.bit-img-wrapper img {
    width: 126px;
    height: 92px;
    object-fit: contain;
    border-radius: 6px;
    display: block;
}

.bit-smarts-caption {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #00D2BE;
    background: rgba(30, 34, 45, 0.95);
    border: 1px solid rgba(255, 255, 255, 0.08);
    width: 100%;
    text-align: center;
    border-radius: 4px;
    padding: 2px 4px;
    margin-top: 4px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.absent-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: rgba(255, 75, 75, 0.14);
    border: 1px solid rgba(255, 75, 75, 0.45);
    color: #FF9494;
    font-weight: 700;
    font-size: 0.84rem;
    padding: 0.32rem 0.75rem;
    border-radius: 8px;
    margin-bottom: 0.55rem;
}

.present-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: rgba(0, 210, 190, 0.14);
    border: 1px solid rgba(0, 210, 190, 0.45);
    color: #00F5D4;
    font-weight: 700;
    font-size: 0.80rem;
    padding: 0.22rem 0.6rem;
    border-radius: 6px;
    margin-bottom: 0.45rem;
}

.descriptor-val-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: #141820;
    border: 1px solid rgba(0, 210, 190, 0.40);
    color: #FFFFFF;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 0.95rem;
    padding: 0.4rem 0.8rem;
    border-radius: 8px;
    margin-bottom: 0.6rem;
}

.pharm-desc-text {
    color: #E2E8F0;
    font-size: 0.88rem;
    line-height: 1.5;
    margin: 0;
}

.smiles-box {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.80rem;
    background: #141820;
    color: #00D2BE;
    border: 1px solid rgba(0, 210, 190, 0.25);
    border-radius: 8px;
    padding: 0.55rem 0.75rem;
    word-break: break-all;
    margin-top: 0.55rem;
}

.mol-canvas-frame {
    background: rgba(14, 17, 23, 0.82);
    border: 1.5px solid rgba(0, 210, 190, 0.40);
    border-radius: 12px;
    padding: 0.5rem;
    text-align: center;
}

.insample-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: rgba(245, 158, 11, 0.16);
    border: 1.5px solid #F59E0B;
    color: #FBBF24;
    font-weight: 700;
    font-size: 0.80rem;
    padding: 0.24rem 0.68rem;
    border-radius: 9999px;
    letter-spacing: 0.01em;
}

.outsample-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: rgba(56, 189, 248, 0.14);
    border: 1.5px solid #38BDF8;
    color: #7DD3FC;
    font-weight: 700;
    font-size: 0.80rem;
    padding: 0.24rem 0.68rem;
    border-radius: 9999px;
    letter-spacing: 0.01em;
}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 3. Mapeamento Farmacofórico de Bits ECFP4 e Descritores Físico-Químicas 2D
# -----------------------------------------------------------------------------
PHARMACOPHORE_BIT_CATALOG: Dict[int, Dict[str, str]] = {
    831: {
        "label": "Aceptor de H em Anel Heteroaromático (N sp²)",
        "scaffold": "Subestrutura piridínica / pirimidínica endocíclica [n(c)c]",
        "role_a2a": (
            "Atua como aceptor direcional de ligação de hidrogênio na fenda ortostérica do receptor A2A "
            "(interação crítica com Asn253 e pontes de água estruturais)."
        ),
        "role_herg": (
            "Modula a densidade eletrônica aromática; anéis heteroaromáticos com nitrogênio endocíclico "
            "tendem a reduzir o LogP efetivo frente a fenilas lipofílicas puras."
        ),
    },
    964: {
        "label": "Núcleo Amino-Heterocíclico / Purínico (Doador-Aceptor)",
        "scaffold": "Tríade 2-aminopirimidínica / aminotriazínica [c(N)(n)n]",
        "role_a2a": (
            "Farmacóforo mestre de antagonistas A2A de alta potência (como ZM241385 e tiazolo-pirimidinas). "
            "Realiza ancoragem bidentada de hidrogênio com Asn253 e Glu169."
        ),
        "role_herg": (
            "Aumenta a superfície polar (TPSA) do núcleo central, desfavorecendo o aprisionamento "
            "hidrofóbico na cavidade interna do canal hERG."
        ),
    },
    904: {
        "label": "Substituinte Fluoroarílico (Halogênio Eletronegativo)",
        "scaffold": "Átomo de flúor terminal ligado a anel aromático [-F]",
        "role_a2a": (
            "Bloqueia sítios de metabolismo oxidativo (CYP450) e preenche sub-bolsões hidrofóbicos "
            "profundos do receptor A2A com mínimo custo entrópico."
        ),
        "role_herg": (
            "Eleva a lipofilicidade aromática e polarizabilidade em anéis periféricos, podendo "
            "intensificar interações π-empilhamento com Phe656 no canal hERG."
        ),
    },
    234: {
        "label": "Núcleo Imídico / Xantínico N-Metilado",
        "scaffold": "Sistema diona N-substituído de xantina/uracila [n(C)(c(=O)c)c(n)=O]",
        "role_a2a": (
            "Elemento estrutural clássico das metilxantinas (Cafeína/Teofilina) que mimetiza o anel "
            "purínico da adenosina no bolso ortostérico A2A."
        ),
        "role_herg": (
            "Padroniza um esqueleto planar polar de baixo volume molecular e caráter neutro, "
            "resultando em perfil de segurança cardíaca favorável."
        ),
    },
    564: {
        "label": "Nitrogênio Purínico / Imidazólico N-Alquilado",
        "scaffold": "Nitrogênio terciário endocíclico N-alquilado [n(c)(c)C]",
        "role_a2a": (
            "Ancora o sistema bicíclico condensado por empilhamento π-π com o resíduo conservado Phe168 "
            "na alça extracelular do receptor A2A."
        ),
        "role_herg": (
            "Ausência de basicidade protonável em pH fisiológico evita a interação cátion-π "
            "típica de bloqueadores clássicos do poro de potássio hERG."
        ),
    },
    575: {
        "label": "Sistema Xantínico N-Etilado / Heterociclo Condensado",
        "scaffold": "Subunidade N-alquil-imidazopirimidinodiona [n(CC)(c(=O)c)c(n)=O]",
        "role_a2a": (
            "Substituinte N-propil/N-etil xantínico (presente na Istradefilina) que projeta grupos "
            "alquila no sub-bolso hidrofóbico, elevando seletividade A2A vs A1."
        ),
        "role_herg": (
            "Mantém neutralidade eletrostática no núcleo xantínico, evitando formação de cátions "
            "amônio quaternários cardiotóxicos."
        ),
    },
    517: {
        "label": "Aceptor Carbonílico em Anel Condensado Purínico",
        "scaffold": "Carbonila conjugada em junção de anel xantínico [c(n(C)c)(=O)c(c)n]",
        "role_a2a": (
            "Oxigênio carbonílico exocíclico que atua como aceptor de ligação de hidrogênio com Asn253, "
            "orientando o plano purínico na cavidade transmembrana."
        ),
        "role_herg": (
            "Aumenta o momento dipolar do heterociclo central, reduzindo a afinidade pelo lúmen "
            "hidrofóbico de Kv11.1."
        ),
    },
    34: {
        "label": "Substituinte N-Etil / Alquil-Heteroaromático",
        "scaffold": "Cadeia alifática curta ligada a nitrogênio heterocíclico [C(C)n]",
        "role_a2a": (
            "Ajusta o encaixe estérico e lipofílico na entrada hidrofóbica do sítio ortostérico A2A."
        ),
        "role_herg": (
            "Contribuinte moderado para a lipofilicidade global sem introduzir centro básico protonável."
        ),
    },
    465: {
        "label": "Junção N-Alquil-Purinona / Sulfonamida Arílica",
        "scaffold": "Ambiente circular de raio 2 em anel heteroaromático funcionalizado",
        "role_a2a": (
            "Estabiliza a conformação bioativa do ligante através de contatos de van der Waals "
            "com resíduos hidrofóbicos da hélice TM6 (Leu249/Ile274)."
        ),
        "role_herg": (
            "Modula o volume molecular e a distribuição de carga superficial no reconhecimento do canal."
        ),
    },
    850: {
        "label": "Ponte Alquil-Heteroaromática [C(n(c)c)C]",
        "scaffold": "Ramificação alifática adjacente a anel purínico/azólico",
        "role_a2a": (
            "Favorece a complementaridade hidrofóbica na fenda de ligação de antagonistas xantínicos."
        ),
        "role_herg": (
            "Altera o balanço lipofílico local da estrutura sem criar alerta estrutural de bloqueio."
        ),
    },
    128: {
        "label": "Amina Alifática / Centro Nitrogenado sp³",
        "scaffold": "Átomo de nitrogênio alifático não-aromático [N]",
        "role_a2a": (
            "Pode atuar como doador/aceptor de H em cadeias laterais flexíveis voltadas ao vestíbulo extracelular."
        ),
        "role_herg": (
            "Aminas alifáticas protonáveis em pH 7,4 constituem um alerta farmacofórico clássico "
            "de interação cátion-π com Tyr652 no canal hERG."
        ),
    },
    147: {
        "label": "Grupo Amino Primário / Secundário Polar",
        "scaffold": "Centro nitrogenado doador de ligação de hidrogênio [N]",
        "role_a2a": (
            "Participa de pontes de hidrogênio diretas com o resíduo Glu169 na entrada do bolso A2A."
        ),
        "role_herg": (
            "Dependendo do pKa, centros nitrogenados básicos elevam a propensão de oclusão eletrostática do poro."
        ),
    },
    672: {
        "label": "Junção de Anel Condensado Purínico / Azólico",
        "scaffold": "Carbono de fusão bicíclica heteroaromática [c(c)(n)n]",
        "role_a2a": (
            "Define a rigidez planar do esqueleto bicíclico (purina/xantina/triazolopirimidina) "
            "indispensável para o empilhamento π-π com Phe168."
        ),
        "role_herg": (
            "Confere planaridade aromática controlada com heteroátomos embutidos."
        ),
    },
    433: {
        "label": "Amina Exocíclica Conjugada a Anel Aromático (-NH₂)",
        "scaffold": "Grupo amino arílico primário [Nc]",
        "role_a2a": (
            "Doador de ligação de hidrogênio de alta energia para Asn253 e Glu169; sua presença "
            "é determinante para afinidade nanomolar em antagonistas não-xantínicos."
        ),
        "role_herg": (
            "Reduz o caráter lipofílico do anel aromático e não sofre protonação forte em pH fisiológico."
        ),
    },
    323: {
        "label": "Eixo Biarílico / Heteroaril-Aril Conjugado",
        "scaffold": "Ligação direta entre anel heteroaromático e fenila/furano [c(n)(-c)c]",
        "role_a2a": (
            "Estende o sistema conjugado planar para alcançar o sub-bolsão aromático profundo do receptor A2A."
        ),
        "role_herg": (
            "Sistemas biarílicos extensos podem aumentar interações hidrofóbicas no vestíbulo do canal hERG."
        ),
    },
    312: {
        "label": "Anel Furano / Oxazol Conjugado (Aceptor de H)",
        "scaffold": "Heterociclo oxigenado de 5 membros conjugado [o(c(-c)c)c]",
        "role_a2a": (
            "Subunidade furânica clássica que se insere profundamente no bolso hidrofóbico interno "
            "do receptor A2A próxima a His250 e Trp246."
        ),
        "role_herg": (
            "Contribui moderadamente para a área apolar sem adicionar basicidade."
        ),
    },
    829: {
        "label": "Anel Fenila Para-Substituído Lipofílico",
        "scaffold": "Subestrutura aromática benzenoide substituída [c(cc)c(c)C]",
        "role_a2a": (
            "Interage por forças hidrofóbicas e dispersão de London nas regiões periféricas do receptor."
        ),
        "role_herg": (
            "Anéis benzênicos lipofílicos periféricos ancoram o ligante nos resíduos aromáticos "
            "Phe656 do canal hERG, aumentando o risco cardiotóxico."
        ),
    },
    338: {
        "label": "Metileno Benzílico / Carbinol Arílico [C(c)(O)C]",
        "scaffold": "Carbono sp³ adjacente a anel aromático e grupo hidroxila/heteroátomo",
        "role_a2a": (
            "Atua como espaçador flexível em cadeias laterais aril-alquílicas."
        ),
        "role_herg": (
            "Comum em bloqueadores hERG tipo difenilmetanol (ex.: Terfenadina), posicionando anéis "
            "fenila para bloqueio estérico do poro."
        ),
    },
    887: {
        "label": "Cadeia Alifática Espaçadora (-CH₂-CH₂-)",
        "scaffold": "Segmento metilênico linear flexível [C(C)C]",
        "role_a2a": (
            "Confere liberdade conformacional à cadeia lateral, podendo aumentar a penalidade entrópica na ligação A2A."
        ),
        "role_herg": (
            "Espaçadores alifáticos de 3 a 4 carbonos conectando uma amina terciária a grupos aromáticos "
            "formam o farmacóforo clássico de bloqueio do canal hERG."
        ),
    },
    767: {
        "label": "Amina Terciária / Centro Alifático Ramificado",
        "scaffold": "Nitrogênio terciário protonável [N(C)(C)C] ou carbono substituído",
        "role_a2a": (
            "Baixa complementaridade com o bolso ortostérico purinérgico A2A."
        ),
        "role_herg": (
            "Alerta crítico de cardiotoxicidade: aminas terciárias alifáticas (como na Piperidina da Terfenadina) "
            "protonam em pH fisiológico e bloqueiam o fluxo de K⁺ por interação cátion-π."
        ),
    },
    807: {
        "label": "Grupo Hidroxila / Oxigênio Polar [-OH / -O-]",
        "scaffold": "Átomo de oxigênio sp³ doador/aceptor de H [O]",
        "role_a2a": (
            "Pode formar ligações de hidrogênio auxiliares no vestíbulo extracelular."
        ),
        "role_herg": (
            "Modula a polaridade local; em álcoois benzidrílicos volumosos pode coexistir com alta lipofilicidade."
        ),
    },
    893: {
        "label": "Oxigênio Carbonílico / Heterociclo Nitrogenado",
        "scaffold": "Grupo carbonila polar [O=C] ou anel pirazólico substituído",
        "role_a2a": (
            "Atua como ponto de ancoragem polar em amidas e xantinas."
        ),
        "role_herg": (
            "Grupos carbonila alifáticos/amídicos elevam a TPSA e reduzem a permeação no poro hidrofóbico hERG."
        ),
    },
    726: {
        "label": "Arcabouço Aromático Benzenoide (CH sp²)",
        "scaffold": "Segmento de anel aromático carbocíclico [c(c)c]",
        "role_a2a": (
            "Participa de interações hidrofóbicas e empilhamento aromático no sítio receptor."
        ),
        "role_herg": (
            "Múltiplos anéis benzênicos aumentam o LogP e favorecem o empilhamento π-π com Tyr652/Phe656 no hERG."
        ),
    },
    650: {
        "label": "Centro Oxigenado Polar (Carbonila / Éter / Hidroxila)",
        "scaffold": "Átomo de oxigênio funcional [O]",
        "role_a2a": (
            "Contribui para ligações de hidrogênio e solvatação controlada no bolsão A2A."
        ),
        "role_herg": (
            "A presença de átomos de oxigênio polares atenua a lipofilicidade global da molécula."
        ),
    },
    653: {
        "label": "Cadeia Alquil-Amina Terciária [C(CC)N(C)C]",
        "scaffold": "Espaçador propil/piperidinil ligado a nitrogênio básico",
        "role_a2a": (
            "Estrutura pouco complementar ao reconhecimento purinérgico clássico do receptor A2A."
        ),
        "role_herg": (
            "Motivo estrutural fortemente associado a bloqueadores de canais iônicos cardíacos (hERG)."
        ),
    },
    121: {
        "label": "Ligação Carbono-Nitrogênio (N-Alquil / Heteroaril)",
        "scaffold": "Unidade metil/alquil ligada a nitrogênio [Cn / C(C)N]",
        "role_a2a": (
            "Presente em substituintes N-metil/N-etil de xantinas que preenchem sub-bolsões apolares."
        ),
        "role_herg": (
            "Em aminas alifáticas terciárias, contribui para a basicidade do centro catiônico."
        ),
    },
    549: {
        "label": "Núcleo Bicíclico Xantínico / Aromático Condensado",
        "scaffold": "Fusão imidazo-pirimidinodiona [c(c(n)=O)(c(n)n)n(c)C]",
        "role_a2a": (
            "Esqueleto central de xantinas substituídas nas posições C8, essencial para antagonismo A2A."
        ),
        "role_herg": (
            "Núcleo heterocíclico polar que confere baixo risco intrínseco de bloqueio de potássio."
        ),
    },
    523: {
        "label": "Carbono C8-Substituído em Anel Purínico / Éter Metílico",
        "scaffold": "Junção de substituinte na posição 8 da xantina [c(n)(n)C] ou grupo metóxi",
        "role_a2a": (
            "A substituição na posição 8 do anel xantínico (como o grupo estiril da Istradefilina) "
            "é o principal vetor de ganho de potência nanomolar e seletividade para A2A."
        ),
        "role_herg": (
            "Introduz grupos polares metoxilados periféricos que amortecem o risco cardiotóxico."
        ),
    },
    219: {
        "label": "Ponte Etil-Arílica / Benzílica [C(C)c]",
        "scaffold": "Carbono alifático ligado a anel aromático",
        "role_a2a": (
            "Conecta grupos aromáticos periféricos ao núcleo central."
        ),
        "role_herg": (
            "Aumenta a flexibilidade hidrofóbica, facilitando a adaptação conformacional no canal hERG."
        ),
    },
    80: {
        "label": "Carbono Alifático Saturado (sp³)",
        "scaffold": "Centro metilênico / alquílico saturado [C]",
        "role_a2a": (
            "Contribui para o preenchimento de cavidades hidrofóbicas rasas."
        ),
        "role_herg": (
            "Eleva incrementalmente o caráter lipofílico da cadeia carbonada."
        ),
    },
    1: {
        "label": "Centro Carbonado / Heteroaromático Substituído",
        "scaffold": "Microambiente atômico de carbono funcionalizado",
        "role_a2a": (
            "Participa da complementaridade estérica no bolsão de ligação."
        ),
        "role_herg": (
            "Influencia o balanço estérico e lipofílico da estrutura."
        ),
    },
}

DESCRIPTOR_INFO_CATALOG: Dict[str, Dict[str, str]] = {
    "MolLogP": {
        "label": "Coeficiente de Partição Octanol/Água (Lipofilicidade)",
        "unit": "log P",
        "role_a2a": (
            "Controla o efeito hidrofóbico de desolvatação ao entrar na fenda transmembrana do receptor A2A. "
            "Valores moderados (1,5 a 3,8) otimizam a afinidade sem comprometer a solubilidade aquosa."
        ),
        "role_herg": (
            "Principal determinante físico-químico da cardiotoxicidade hERG: compostos altamente lipofílicos "
            "(MolLogP > 4,5) atravessam facilmente a membrana e se alojam na cavidade hidrofóbica de Phe656/Tyr652."
        ),
    },
    "TPSA": {
        "label": "Área de Superfície Polar Topológica (TPSA)",
        "unit": "Å²",
        "role_a2a": (
            "Reflete a capacidade global de formar ligações de hidrogênio no bolso ortostérico A2A. "
            "Faixas entre 55 e 95 Å² equilibram alta afinidade purinérgica e permeabilidade no SNC."
        ),
        "role_herg": (
            "Superfícies polares elevadas (TPSA > 75 Å²) impõem barreira energética à desolvatação, "
            "protegendo contra a oclusão hidrofóbica do poro do canal hERG."
        ),
    },
    "MolWt": {
        "label": "Massa Molecular Média (Volume Estérico)",
        "unit": "Da",
        "role_a2a": (
            "Define o volume molecular disponível para ocupar simultaneamente o bolso ortostérico e o "
            "vestíbulo secundário do receptor A2A (ótimo entre 300 e 460 Da)."
        ),
        "role_herg": (
            "Moléculas volumosas (> 400 Da) com múltiplos anéis aromáticos apresentam maior superfície "
            "de contato van der Waals no lúmen interno do canal Kv11.1."
        ),
    },
    "NumHDonors": {
        "label": "Doadores de Ligação de Hidrogênio (HBD)",
        "unit": "grupos (-OH / -NH)",
        "role_a2a": (
            "Grupos doadores de H (especialmente -NH₂ exocíclico) são cruciais para ancoragem com Asn253 e Glu169."
        ),
        "role_herg": (
            "Modula a solubilidade aquosa e a carga formal em pH fisiológico."
        ),
    },
    "NumHAcceptors": {
        "label": "Aceptores de Ligação de Hidrogênio (HBA)",
        "unit": "átomos (N / O)",
        "role_a2a": (
            "Nitrogênios endocíclicos e carbonilas estabelecem a rede de reconhecimento molecular no receptor A2A."
        ),
        "role_herg": (
            "Heteroátomos aceptores aumentam a polaridade e reduzem a afinidade inespecífica por canais lipofílicos."
        ),
    },
    "NumRotatableBonds": {
        "label": "Ligações Rotacionáveis (Flexibilidade Conformacional)",
        "unit": "ligações",
        "role_a2a": (
            "Controla a penalidade entrópica no acoplamento ao receptor A2A; esqueletos rígidos pré-organizados "
            "costumam exibir maior potência nanomolar."
        ),
        "role_herg": (
            "Alta flexibilidade conformacional permite que a molécula se dobre para interagir simultaneamente "
            "com múltiplos resíduos aromáticos no interior do poro hERG."
        ),
    },
}


# -----------------------------------------------------------------------------
# 4. Funções Auxiliares de Quimioinformática e Renderização RDKit (Morgan bitInfo)
# -----------------------------------------------------------------------------
def extract_morgan_bit_info(mol: Chem.Mol, radius: int = 2, n_bits: int = 1024) -> Dict[int, Any]:
    """Gera o Morgan Fingerprint (ECFP4) rastreando o dicionário bitInfo do RDKit."""
    bit_info: Dict[int, Any] = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits, bitInfo=bit_info)
    return bit_info


def get_bit_fragment_smiles(mol: Chem.Mol, bit_id: int, bit_info: Dict[int, Any]) -> Tuple[str, str, int]:
    """Extrai o SMILES do fragmento atômico exato, símbolo do átomo central e raio do bit ativo."""
    if bit_id not in bit_info or not bit_info[bit_id]:
        return "", "", 0

    atom_idx, radius = bit_info[bit_id][0]
    center_atom = mol.GetAtomWithIdx(atom_idx)
    center_symbol = center_atom.GetSymbol()

    if radius == 0:
        return center_symbol, center_symbol, 0

    env_bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
    if not env_bonds:
        return center_symbol, center_symbol, radius

    atom_map: Dict[int, int] = {}
    submol = Chem.PathToSubmol(mol, env_bonds, atomMap=atom_map)
    if submol is None or submol.GetNumAtoms() == 0:
        return center_symbol, center_symbol, radius

    rooted_idx = atom_map.get(atom_idx, 0)
    frag_smiles = Chem.MolToSmiles(submol, rootedAtAtom=rooted_idx, canonical=False)
    return frag_smiles, center_symbol, radius


def render_molecule_dark_base64(mol: Chem.Mol, size: Tuple[int, int] = (420, 260)) -> Optional[str]:
    """Renderiza a molécula 2D completa com fundo transparente e paleta Dark Mode via MolDraw2DCairo."""
    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        opts = drawer.drawOptions()
        rdMolDraw2D.SetDarkMode(opts)
        opts.clearBackground = False
        opts.bondLineWidth = 2.2
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return base64.b64encode(drawer.GetDrawingText()).decode("utf-8")
    except Exception:
        return None


def render_morgan_bit_base64(
    mol: Chem.Mol,
    bit_id: int,
    bit_info: Dict[int, Any],
    size: Tuple[int, int] = (200, 145),
) -> Optional[str]:
    """Desenha o microambiente atômico do bit ativo com Draw.DrawMorganBit em Dark Mode transparente."""
    if bit_id not in bit_info:
        return None
    try:
        opts = rdMolDraw2D.MolDrawOptions()
        rdMolDraw2D.SetDarkMode(opts)
        opts.clearBackground = False
        opts.bondLineWidth = 2.0

        drawn = Draw.DrawMorganBit(
            mol,
            bit_id,
            bit_info,
            molSize=size,
            useSVG=False,
            drawOptions=opts,
            centerColor=(0.0, 0.82, 0.75, 0.55),
            aromaticColor=(0.96, 0.62, 0.04, 0.45),
            ringColor=(0.38, 0.45, 0.55, 0.42),
            extraColor=(0.45, 0.52, 0.62, 0.35),
        )
        if isinstance(drawn, bytes):
            png_bytes = drawn
        else:
            buf = io.BytesIO()
            drawn.save(buf, format="PNG")
            png_bytes = buf.getvalue()
        return base64.b64encode(png_bytes).decode("utf-8")
    except Exception:
        try:
            atom_idx, radius = bit_info[bit_id][0]
            highlight_atoms = [atom_idx]
            if radius > 0:
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
                for b_idx in env:
                    bond = mol.GetBondWithIdx(b_idx)
                    highlight_atoms.extend([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
            opts = drawer.drawOptions()
            rdMolDraw2D.SetDarkMode(opts)
            opts.clearBackground = False
            drawer.DrawMolecule(mol, highlightAtoms=list(set(highlight_atoms)))
            drawer.FinishDrawing()
            return base64.b64encode(drawer.GetDrawingText()).decode("utf-8")
        except Exception:
            return None


def resolve_pharmacophore_details(
    feature_name: str,
    mol: Chem.Mol,
    bit_info: Dict[int, Any],
    target: str = "A2A",
) -> Dict[str, str]:
    """Retorna rótulo farmacofórico, subestrutura e interpretação biológica para qualquer atributo."""
    if feature_name in DESCRIPTOR_INFO_CATALOG:
        info = DESCRIPTOR_INFO_CATALOG[feature_name]
        return {
            "label": info["label"],
            "unit": info["unit"],
            "scaffold": "Propriedade Físico-Química Global 2D",
            "role": info["role_a2a"] if target == "A2A" else info["role_herg"],
        }

    bit_id = int(feature_name.split("_")[1])
    frag_smi, center_sym, radius = get_bit_fragment_smiles(mol, bit_id, bit_info)

    if bit_id in PHARMACOPHORE_BIT_CATALOG:
        entry = PHARMACOPHORE_BIT_CATALOG[bit_id]
        scaffold_str = f"Fragmento na molécula: {frag_smi} (r={radius})" if frag_smi else entry["scaffold"]
        return {
            "label": entry["label"],
            "unit": "",
            "scaffold": scaffold_str,
            "frag_smi": frag_smi or entry["scaffold"],
            "role": entry["role_a2a"] if target == "A2A" else entry["role_herg"],
        }

    if frag_smi:
        has_arom = any(ch in frag_smi for ch in ("c", "n", "o", "s"))
        has_hetero = any(ch in frag_smi.lower() for ch in ("n", "o", "s", "f", "cl"))
        if has_arom and "n" in frag_smi.lower():
            dyn_label = f"Subestrutura Heteroaromática Nitrogenada (Raio {radius})"
        elif has_arom:
            dyn_label = f"Microambiente Aromático Conjugado (Raio {radius})"
        elif has_hetero:
            dyn_label = f"Grupo Funcional Polar Centra-Atômico ({center_sym}, Raio {radius})"
        else:
            dyn_label = f"Subestrutura Alifática / Alquílica ({center_sym}, Raio {radius})"

        role_txt = (
            f"Microambiente topológico centrado em {center_sym} ({frag_smi}) que modula o encaixe "
            f"estérico e eletrônico no sítio ortostérico do receptor A2A."
            if target == "A2A"
            else f"Microambiente topológico centrado em {center_sym} ({frag_smi}) que influencia a "
            f"interação hidrofóbica/eletrostática com o canal de potássio hERG."
        )
        return {
            "label": dyn_label,
            "unit": "",
            "scaffold": f"Fragmento ativo: {frag_smi} (r={radius})",
            "frag_smi": frag_smi,
            "role": role_txt,
        }

    return {
        "label": f"Subestrutura Topológica de Referência ECFP4 (Bit #{bit_id})",
        "unit": "",
        "scaffold": "Motivo farmacofórico de referência ausente neste composto",
        "frag_smi": f"Bit ECFP4 #{bit_id}",
        "role": (
            "Subestrutura recorrente em ligantes ativos de adenosina A2A cuja presença ou ausência "
            "desloca a potência teórica predita pelo ensemble de árvores."
            if target == "A2A"
            else "Motivo estrutural monitorado pelo classificador Random Forest para estimativa de risco "
            "de bloqueio do canal hERG."
        ),
    }


def build_pharmacophore_card_html(
    feature_name: str,
    mol_value: float,
    shap_impact: float,
    mol: Chem.Mol,
    bit_info: Dict[int, Any],
    target: str = "A2A",
) -> str:
    """Gera o HTML compactado (sem indentação Markdown) de um cartão de farmacóforo/descritor."""
    details = resolve_pharmacophore_details(feature_name, mol, bit_info, target=target)
    is_descriptor = feature_name in DESCRIPTOR_NAMES

    if target == "A2A":
        is_favorable = shap_impact >= 0
        effect_word = "Aumenta Afinidade" if shap_impact >= 0 else "Atenua Afinidade"
        unit_suffix = " pIC50"
    else:
        is_favorable = shap_impact < 0
        effect_word = "Reduz Toxicidade" if shap_impact < 0 else "Aumenta Risco hERG"
        unit_suffix = " P(hERG)"

    card_border_cls = "pharmacophore-card-pos" if is_favorable else "pharmacophore-card-neg"
    tag_cls = "shap-tag-pos" if is_favorable else "shap-tag-neg"
    shap_str = f"SHAP: {shap_impact:+.3f}{unit_suffix} ({effect_word})"

    if is_descriptor:
        val_formatted = (
            f"{int(round(mol_value))}"
            if feature_name in ("NumHDonors", "NumHAcceptors", "NumRotatableBonds")
            else f"{mol_value:.2f}"
        )
        body_html = (
            f'<div>'
            f'<div class="descriptor-val-badge">'
            f'<span>📐 Valor Calculado:</span>'
            f'<span style="color:#00D2BE;">{val_formatted} {details["unit"]}</span>'
            f'</div>'
            f'<p class="pharm-desc-text">{details["role"]}</p>'
            f'</div>'
        )
    else:
        bit_id = int(feature_name.split("_")[1])
        is_present = int(round(mol_value)) == 1

        if is_present:
            b64_img = render_morgan_bit_base64(mol, bit_id, bit_info)
            frag_smi = details.get("frag_smi", f"fp_{bit_id}")
            img_block = (
                f'<div class="bit-img-wrapper">'
                f'<img src="data:image/png;base64,{b64_img}" alt="{feature_name}"/>'
                f'<div class="bit-smarts-caption" title="{frag_smi}">{frag_smi}</div>'
                f'</div>'
                if b64_img
                else ""
            )
            body_html = (
                f'<div class="pharm-body-present">'
                f'{img_block}'
                f'<div style="flex:1;">'
                f'<div class="present-badge">✅ Presente na molécula (Bit Ativo = 1)</div>'
                f'<p class="pharm-desc-text">{details["role"]}</p>'
                f'</div>'
                f'</div>'
            )
        else:
            absence_reason = (
                "A ausência desta subestrutura farmacofórica na molécula submetida reduziu a complementaridade "
                "teórica de ligação no bolsão ortostérico do receptor A2A."
                if (target == "A2A" and shap_impact < 0)
                else (
                    "A ausência deste grupo desfavorável contribuiu positivamente para a potência estimada no receptor A2A."
                    if target == "A2A"
                    else (
                        "A ausência deste alerta estrutural cardiotóxico protegeu a molécula contra o bloqueio do canal hERG."
                        if shap_impact < 0
                        else "A falta deste grupo polar de proteção elevou a propensão relativa ao bloqueio hERG."
                    )
                )
            )
            body_html = (
                f'<div>'
                f'<div class="absent-badge">❌ Ausente na molécula (Valor = 0)</div>'
                f'<p class="pharm-desc-text" style="margin-bottom:0.35rem;">'
                f'<strong>Subestrutura de Referência:</strong> {details["scaffold"]}'
                f'</p>'
                f'<p class="pharm-desc-text">{absence_reason} {details["role"]}</p>'
                f'</div>'
            )

    card_html = (
        f'<div class="pharmacophore-card {card_border_cls}">'
        f'<div class="pharm-card-header">'
        f'<div>'
        f'<span class="pharm-feature-id">{feature_name}</span>'
        f'<div class="pharm-feature-title">{details["label"]}</div>'
        f'</div>'
        f'<span class="{tag_cls}">{shap_str}</span>'
        f'</div>'
        f'{body_html}'
        f'</div>'
    )
    return card_html


# -----------------------------------------------------------------------------
# 5. Carregamento dos Recursos em Cache Local (8 GB RAM Friendly)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models_and_explainers():
    """Carrega os modelos treinados (.joblib) e instancia os explicadores SHAP em cache."""
    root = Path(__file__).resolve().parent
    model_a2a = joblib.load(root / "models" / "a2a_model.joblib")
    model_herg = joblib.load(root / "models" / "herg_model.joblib")

    with open(root / "models" / "a2a_model_metadata.json", "r", encoding="utf-8") as f:
        meta_a2a = json.load(f)

    with open(root / "models" / "herg_model_metadata.json", "r", encoding="utf-8") as f:
        meta_herg = json.load(f)

    explainer_a2a = shap.TreeExplainer(model_a2a)
    explainer_herg = shap.TreeExplainer(model_herg)

    return model_a2a, model_herg, meta_a2a, meta_herg, explainer_a2a, explainer_herg


@st.cache_resource(show_spinner=False)
def load_ad_reference_cached(endpoint: str):
    """Carrega e mantém em cache os dados de referência do Domínio de Aplicabilidade."""
    return get_ad_reference(endpoint)


try:
    model_a2a, model_herg, meta_a2a, meta_herg, explainer_a2a, explainer_herg = load_models_and_explainers()
    models_loaded = True
except Exception as e:
    models_loaded = False
    load_error_msg = str(e)


# -----------------------------------------------------------------------------
# 6. Dados dos Controles Farmacológicos e Proveniência Amostral
# -----------------------------------------------------------------------------
REFERENCE_CONTROLS: Dict[str, Dict[str, Any]] = {
    "☕ Cafeína (CHEMBL113)": {
        "name": "Cafeína",
        "chembl_id": "CHEMBL113",
        "smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "canonical_smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "desc": (
            "Antagonista não-seletivo clássico de receptores de adenosina (A1/A2A). "
            "Presente no conjunto de treinamento A2A com potência micromolar medida em ensaios de ligação."
        ),
        "a2a_in_sample": True,
        "a2a_sample_status": "Amostra de Treinamento (In-Sample)",
        "a2a_exp_pic50": 4.8533,
        "a2a_exp_assay": (
            "Medição experimental ChEMBL (CHEMBL113): mediana pIC50 = 4,85 oriunda de 2 ensaios de ligação "
            "radioligante (assay_type = 'B', IC50 = 9.600 nM e 20.472 nM) em data/processed/a2a_curated.csv."
        ),
        "herg_in_sample": False,
        "herg_sample_status": "Fora da Amostra de Treino (Out-of-Sample)",
        "external_ref": (
            "Referência externa (Fredholm et al., Pharmacol. Rev.): antagonista purinérgico micromolar "
            "(IC50 / Ki ~ 10–24 µM em A2A humano) sem bloqueio clinicamente significativo de hERG em faixas fisiológicas."
        ),
    },
    "💊 Istradefilina (CHEMBL431770)": {
        "name": "Istradefilina",
        "chembl_id": "CHEMBL431770",
        "smiles": "CCN1C(=O)C2=C(N=C(N2C)/C=C/C3=CC(=C(C=C3)OC)OC)N(C1=O)CC",
        "canonical_smiles": "CCn1c(=O)c2c(nc(/C=C/c3ccc(OC)c(OC)c3)n2C)n(CC)c1=O",
        "desc": (
            "Antagonista seletivo do receptor A2A aprovado (FDA/PMDA) para Doença de Parkinson. "
            "Registrado na base local de treinamento A2A sob CHEMBL431770 a partir de ensaio funcional celular (cAMP)."
        ),
        "a2a_in_sample": True,
        "a2a_sample_status": "Amostra de Treinamento (In-Sample)",
        "a2a_exp_pic50": 5.2799,
        "a2a_exp_assay": (
            "Medição experimental ChEMBL (CHEMBL431770): mediana pIC50 = 5,28 oriunda de 2 registros de ensaio "
            "funcional celular de inibição de cAMP induzido por NECA em células CHO (assay_type = 'F', "
            "CHEMBL2384072, IC50 = 5.250 nM e 5.248 nM). Endpoint funcional celular distinto de ensaios de ligação (pKi)."
        ),
        "herg_in_sample": False,
        "herg_sample_status": "Fora da Amostra de Treino (Out-of-Sample)",
        "external_ref": (
            "Referência externa (Literatura / Bula FDA Nourianz® / KW-6002): apresenta afinidade nanomolar em ensaios "
            "de ligação radioligante (pKi ~ 7,5–8,1), porém o dataset curado local contém exclusivamente o IC50 "
            "funcional celular (pIC50 = 5,28), o que explica objetivamente a predição micromolar do modelo."
        ),
    },
    "⚠️ Terfenadina (CHEMBL22)": {
        "name": "Terfenadina",
        "chembl_id": "CHEMBL22",
        "smiles": "CC(C)(C)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4",
        "canonical_smiles": "CC(C)(C)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1",
        "desc": (
            "Anti-histamínico de 2ª geração retirado do mercado por cardiotoxicidade severa associada ao bloqueio "
            "do canal hERG. Controle positivo de referência externa para o escore de risco hERG."
        ),
        "a2a_in_sample": False,
        "a2a_sample_status": "Fora da Amostra de Treino (Out-of-Sample)",
        "a2a_exp_pic50": None,
        "a2a_exp_assay": "Sem medição experimental em data/processed/a2a_curated.csv (composto ausente do treino A2A).",
        "herg_in_sample": False,
        "herg_sample_status": "Fora da Amostra de Treino (Out-of-Sample)",
        "external_ref": (
            "Referência externa (Roy et al., Circulation 1996): bloqueador potente do canal hERG (IC50 ~ 56 nM em "
            "patch-clamp) associado a prolongamento QT e Torsades de Pointes; sem atividade alvo reportada em A2A."
        ),
    },
    "🌿 Taurina (Endógeno)": {
        "name": "Taurina",
        "chembl_id": "Endógeno",
        "smiles": "NCCS(=O)(=O)O",
        "canonical_smiles": "NCCS(=O)(=O)O",
        "desc": (
            "Ácido aminossulfônico endógeno altamente hidrofílico (MolWt = 125,2 Da), utilizado como sonda "
            "de controle negativo estrutural."
        ),
        "a2a_in_sample": False,
        "a2a_sample_status": "Fora da Amostra de Treino (Out-of-Sample)",
        "a2a_exp_pic50": None,
        "a2a_exp_assay": (
            "Sem medição experimental em data/processed/a2a_curated.csv (composto ausente do treino A2A; "
            "MolWt = 125,2 Da abaixo da faixa de massa molecular do treino A2A [180,2–757,5 Da])."
        ),
        "herg_in_sample": False,
        "herg_sample_status": "Fora da Amostra de Treino (Out-of-Sample)",
        "external_ref": (
            "Referência externa (Literatura bioquímica): composto osmorregulador endógeno sem atividade ligante "
            "ortostérica documentada no receptor A2A nem bloqueio do canal hERG."
        ),
    },
    "🍵 L-Teanina (Nutracêutico)": {
        "name": "L-Teanina",
        "chembl_id": "Nutracêutico",
        "smiles": "CCNC(=O)CCC(N)C(=O)O",
        "canonical_smiles": "CCNC(=O)CCC(N)C(=O)O",
        "desc": (
            "Aminoácido não-proteico de Camellia sinensis (MolWt = 174,2 Da), utilizado como sonda "
            "nutracêutica de controle negativo."
        ),
        "a2a_in_sample": False,
        "a2a_sample_status": "Fora da Amostra de Treino (Out-of-Sample)",
        "a2a_exp_pic50": None,
        "a2a_exp_assay": (
            "Sem medição experimental em data/processed/a2a_curated.csv (composto ausente do treino A2A; "
            "MolWt = 174,2 Da abaixo da faixa de massa molecular do treino A2A [180,2–757,5 Da])."
        ),
        "herg_in_sample": False,
        "herg_sample_status": "Fora da Amostra de Treino (Out-of-Sample)",
        "external_ref": (
            "Referência externa (Literatura nutricional/neuroquímica): aminoácido polar sem atividade direta "
            "documentada de antagonismo ortostérico em A2A ou bloqueio do canal hERG."
        ),
    },
}


@st.cache_data
def load_training_lookup() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """Carrega índices de consulta rápida (canonical_smiles) das bases curadas de treinamento A2A e hERG."""
    root = Path(__file__).resolve().parent
    a2a_lookup: Dict[str, Dict[str, Any]] = {}
    herg_lookup: Dict[str, int] = {}

    a2a_path = root / "data" / "processed" / "a2a_curated.csv"
    if a2a_path.exists():
        df_a2a = pd.read_csv(a2a_path, usecols=["molecule_chembl_id", "canonical_smiles", "pIC50"])
        for row in df_a2a.itertuples(index=False):
            a2a_lookup[str(row.canonical_smiles)] = {
                "molecule_chembl_id": str(row.molecule_chembl_id),
                "pIC50": float(row.pIC50),
            }

    herg_path = root / "data" / "processed" / "herg_curated.csv"
    if herg_path.exists():
        df_herg = pd.read_csv(herg_path, usecols=["canonical_smiles", "hERG_label"])
        for row in df_herg.itertuples(index=False):
            herg_lookup[str(row.canonical_smiles)] = int(row.hERG_label)

    return a2a_lookup, herg_lookup


def resolve_compound_provenance(canonical_smi: str) -> Dict[str, Any]:
    """
    Resolve a identidade verificada, o status amostral (In-Sample vs. Out-of-Sample),
    as medições experimentais presentes no dataset local e as referências externas
    para qualquer estrutura SMILES canônica submetida.
    """
    a2a_lookup, herg_lookup = load_training_lookup()
    a2a_hit = a2a_lookup.get(canonical_smi)
    herg_hit = herg_lookup.get(canonical_smi)

    for ctrl_key, ctrl_meta in REFERENCE_CONTROLS.items():
        ctrl_canon = ctrl_meta.get("canonical_smiles") or canonicalize_smiles(ctrl_meta["smiles"])
        if ctrl_canon == canonical_smi:
            a2a_in_sample = bool(a2a_hit is not None or ctrl_meta["a2a_in_sample"])
            herg_in_sample = bool(herg_hit is not None or ctrl_meta["herg_in_sample"])
            exp_pic50 = float(a2a_hit["pIC50"]) if a2a_hit is not None else ctrl_meta["a2a_exp_pic50"]
            return {
                "is_reference_control": True,
                "control_key": ctrl_key,
                "name": ctrl_meta["name"],
                "chembl_id": ctrl_meta["chembl_id"],
                "canonical_smiles": canonical_smi,
                "desc": ctrl_meta["desc"],
                "a2a_in_sample": a2a_in_sample,
                "a2a_sample_status": (
                    "Amostra de Treinamento (In-Sample)"
                    if a2a_in_sample
                    else "Fora da Amostra de Treino (Out-of-Sample)"
                ),
                "a2a_exp_pic50": exp_pic50,
                "a2a_exp_assay": ctrl_meta["a2a_exp_assay"],
                "herg_in_sample": herg_in_sample,
                "herg_sample_status": (
                    "Amostra de Treinamento (In-Sample)"
                    if herg_in_sample
                    else "Fora da Amostra de Treino (Out-of-Sample)"
                ),
                "herg_exp_label": herg_hit,
                "external_ref": ctrl_meta["external_ref"],
            }

    a2a_in_sample = a2a_hit is not None
    herg_in_sample = herg_hit is not None
    chembl_id = a2a_hit["molecule_chembl_id"] if a2a_hit is not None else "Não catalogado localmente"
    exp_pic50 = float(a2a_hit["pIC50"]) if a2a_hit is not None else None
    exp_assay = (
        f"Medição experimental curada do ChEMBL ({chembl_id}): mediana pIC50 = {exp_pic50:.2f} "
        "em data/processed/a2a_curated.csv."
        if exp_pic50 is not None
        else "Sem medição experimental em data/processed/a2a_curated.csv (Fora da Amostra / Out-of-Sample)."
    )

    return {
        "is_reference_control": False,
        "control_key": None,
        "name": f"Composto Catalogado ({chembl_id})" if a2a_in_sample else "Molécula Personalizada",
        "chembl_id": chembl_id,
        "canonical_smiles": canonical_smi,
        "desc": (
            f"Estrutura molecular identificada no conjunto de treinamento A2A ({chembl_id})."
            if a2a_in_sample
            else "Estrutura molecular submetida para triagem preditiva multiobjetivo e decomposição farmacofórica SHAP."
        ),
        "a2a_in_sample": a2a_in_sample,
        "a2a_sample_status": (
            "Amostra de Treinamento (In-Sample)"
            if a2a_in_sample
            else "Fora da Amostra de Treino (Out-of-Sample)"
        ),
        "a2a_exp_pic50": exp_pic50,
        "a2a_exp_assay": exp_assay,
        "herg_in_sample": herg_in_sample,
        "herg_sample_status": (
            "Amostra de Treinamento (In-Sample)"
            if herg_in_sample
            else "Fora da Amostra de Treino (Out-of-Sample)"
        ),
        "herg_exp_label": herg_hit,
        "external_ref": None,
    }


def build_control_provenance_html(
    provenance: Dict[str, Any],
    pred_pIC50: float,
    prob_herg: float,
) -> str:
    """
    Constrói o cartão HTML de auditoria amostral e separação explícita entre:
    (1) predição do modelo A2A, (2) medição experimental no dataset utilizado,
    (3) escore de probabilidade estimado para hERG e índice complementar, e
    (4) referência bibliográfica externa.
    """
    a2a_badge_cls = "insample-badge" if provenance["a2a_in_sample"] else "outsample-badge"
    a2a_badge_icon = "📌" if provenance["a2a_in_sample"] else "🔬"
    herg_badge_cls = "insample-badge" if provenance["herg_in_sample"] else "outsample-badge"
    herg_badge_icon = "📌" if provenance["herg_in_sample"] else "🔬"

    if provenance["a2a_exp_pic50"] is not None:
        exp_val_str = f"pIC50 exp. = {provenance['a2a_exp_pic50']:.2f}"
        delta_val = pred_pIC50 - float(provenance["a2a_exp_pic50"])
        exp_sub_str = f"Resíduo in-sample (Predito − Experimental): {delta_val:+.2f} log"
    else:
        exp_val_str = "Não disponível no dataset local"
        exp_sub_str = "Composto fora da amostra de treinamento A2A (Out-of-Sample)"

    ext_ref_block = ""
    if provenance.get("external_ref"):
        ext_ref_block = (
            f'<div style="margin-top:0.65rem; padding-top:0.55rem; border-top:1px solid rgba(255,255,255,0.08); '
            f'font-size:0.83rem; color:#CBD5E1;">'
            f'<strong style="color:#00D2BE;">📚 Referência Bibliográfica Externa vs. Endpoint do Dataset:</strong> '
            f'{provenance["external_ref"]}'
            f'</div>'
        )

    insample_warning = ""
    if provenance["a2a_in_sample"] or provenance["herg_in_sample"]:
        insample_warning = (
            '<div style="margin-top:0.55rem; font-size:0.81rem; color:#FBBF24;">'
            '⚠️ <strong>Aviso de Validação (Modelo Reajustado em 100% dos Dados):</strong> '
            'Este composto integra o conjunto de treinamento do modelo serializado '
            '(<code>Amostra de Treinamento (In-Sample)</code>). Sua predição reflete ajuste interno '
            'pós-refit e <strong>não constitui validação cega externa (out-of-sample)</strong>.'
            '</div>'
        )

    comp_index = 1.0 - prob_herg

    html = f"""
<div class="metric-card" style="padding:1.05rem 1.3rem; border-left:4px solid #00D2BE; margin-bottom:1.1rem;">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem; margin-bottom:0.65rem;">
        <div>
            <span style="color:#00D2BE; font-weight:700; font-size:1.02rem;">
                🧾 Identidade Verificada &amp; Proveniência — {provenance["name"]} ({provenance["chembl_id"]})
            </span>
        </div>
        <div style="display:flex; gap:0.45rem; flex-wrap:wrap;">
            <span class="{a2a_badge_cls}">{a2a_badge_icon} A2A: {provenance["a2a_sample_status"]}</span>
            <span class="{herg_badge_cls}">{herg_badge_icon} hERG: {provenance["herg_sample_status"]}</span>
        </div>
    </div>
    <p class="pharm-desc-text" style="font-size:0.86rem; margin-bottom:0.75rem;">{provenance["desc"]}</p>
    <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:0.75rem;">
        <div class="kpi-box">
            <div class="kpi-label">1. Predição do Modelo (A2A)</div>
            <div style="font-size:1.05rem; font-weight:700; color:#00D2BE;">pIC50 predito = {pred_pIC50:.2f}</div>
            <div style="font-size:0.78rem; color:#94A3B8; margin-top:0.15rem;">Estimativa do XGBoost Regressor</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-label">2. Medição Experimental no Dataset (A2A)</div>
            <div style="font-size:1.05rem; font-weight:700; color:#FFFFFF;">{exp_val_str}</div>
            <div style="font-size:0.78rem; color:#94A3B8; margin-top:0.15rem;">{exp_sub_str}</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-label">3. Escore Predito pelo Modelo (hERG)</div>
            <div style="font-size:1.05rem; font-weight:700; color:#FFFFFF;">
                P(hERG) = {prob_herg:.1%} <span style="font-size:0.82rem; color:#00D2BE;">(1 - P = {comp_index:.1%})</span>
            </div>
            <div style="font-size:0.78rem; color:#94A3B8; margin-top:0.15rem;">
                Escore RF (predict_proba não calibrado) e índice complementar
            </div>
        </div>
    </div>
    <div style="margin-top:0.65rem; font-size:0.82rem; color:#CBD5E1;">
        <strong>🧪 Origem Experimental no Dataset Local (A2A):</strong> {provenance["a2a_exp_assay"]}
    </div>
    {ext_ref_block}
    {insample_warning}
</div>
"""
    return "\n".join(line.strip() for line in html.strip().splitlines() if line.strip())


# -----------------------------------------------------------------------------
# 7. Barra Lateral (Sidebar)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧬 MultiTarget BioPredict")
    st.caption("Triagem Virtual Multiobjetivo $A_{2A}$ × hERG")

    st.markdown("---")
    st.markdown("#### ⚙️ Modo de Entrada")
    input_mode = st.radio(
        "Selecione o método de submissão:",
        ["🧪 Controles de Referência (1-Clique)", "✏️ Inserir SMILES (Texto)", "🎨 Desenhar Molécula (Ketcher)"],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### 📊 Modelos Preditivos Locais")
    render_html(
        """
        <div class="metric-card" style="padding:0.9rem 1rem; margin-bottom:0.6rem;">
            <div style="color:#00D2BE; font-weight:700; font-size:0.9rem;">🎯 Eficácia Terapêutica (A2A)</div>
            <div style="font-size:0.82rem; color:#E2E8F0; margin-top:0.2rem;">
                Algoritmo: <code>XGBoost Regressor</code><br>
                Holdout 80/20 prévio (n=273): <b>R² = 0,723</b> | <b>RMSE = 0,578</b><br>
                <span style="color:#94A3B8; font-size:0.76rem;">Artefato final reajustado em 100% dos dados (1.364 compostos).</span>
            </div>
        </div>
        <div class="metric-card" style="padding:0.9rem 1rem; margin-bottom:0.4rem;">
            <div style="color:#00D2BE; font-weight:700; font-size:0.9rem;">❤️ Triagem hERG (Escore do Modelo)</div>
            <div style="font-size:0.82rem; color:#E2E8F0; margin-top:0.2rem;">
                Algoritmo: <code>Random Forest Classifier</code><br>
                Holdout 80/20 prévio (n=2.628): <b>ROC-AUC = 0,930</b> | <b>Brier = 0,112</b><br>
                <span style="color:#94A3B8; font-size:0.76rem;">Artefato final reajustado em 100% (13.136 compostos); escore via <code>predict_proba</code> sem calibração pós-hoc.</span>
            </div>
        </div>
        """
    )

    st.markdown("---")
    st.markdown("#### 🚦 Classes Operacionais dos Modelos")
    st.markdown(
        r"""
        **Eficácia $A_{2A}$ ($pIC_{50}$ predito):**
        - 🟢 **Alta Potência:** $pIC_{50} \ge 6{,}5$
        - 🟡 **Moderada:** $5{,}0 \le pIC_{50} < 6{,}5$
        - 🔴 **Baixa/Inativo:** $pIC_{50} < 5{,}0$

        **Escore Estimado hERG ($P_{\text{hERG}}$):**
        - 🟢 **Baixo (Operacional):** $P < 0{,}30$ ($30\%$)
        - 🟡 **Intermediário/Moderado:** $0{,}30 \le P < 0{,}70$ ($30\% \le P < 70\%$)
        - 🔴 **Alto (Operacional):** $P \ge 0{,}70$ ($70\%$)
        """
    )
    st.caption(
        "Nota de Governança: As métricas de holdout referem-se à divisão 80/20 anterior ao ajuste final (refit 100%). "
        "Predições sobre compostos presentes no treino (In-Sample) não constituem avaliação independente."
    )
    st.caption("100% Local · AMD Ryzen 7 CPU · 8 GB RAM")


# -----------------------------------------------------------------------------
# 8. Cabeçalho Principal
# -----------------------------------------------------------------------------
render_html(
    """
    <div class="hero-banner">
        <div class="main-title">MultiTarget BioPredict</div>
        <div class="sub-title">
            Plataforma Computacional de Triagem Virtual Multiobjetivo e Interpretabilidade Química (RDKit + SHAP):
            Afinidade Predita ao Receptor de Adenosina <b>A<sub>2A</sub></b> vs. Escore de Probabilidade Estimado para Bloqueio do Canal <b>hERG (K<sub>v</sub>11.1)</b>
        </div>
    </div>
    """
)

if not models_loaded:
    st.error(f"Erro ao carregar os modelos preditivos em `models/`: {load_error_msg}")
    st.stop()


# -----------------------------------------------------------------------------
# 9. Resolução da Molécula Submetida (3 Modos Preservados)
# -----------------------------------------------------------------------------
active_smiles = ""
active_name = "Molécula Personalizada"
active_desc = "Estrutura molecular submetida para auditoria preditiva multiobjetivo e decomposição farmacofórica SHAP."

if input_mode == "🧪 Controles de Referência (1-Clique)":
    st.markdown("#### 🧪 Selecione um Composto de Controle Farmacológico")
    btn_cols = st.columns(len(REFERENCE_CONTROLS))
    if (
        "selected_control_key" not in st.session_state
        or st.session_state["selected_control_key"] not in REFERENCE_CONTROLS
    ):
        st.session_state["selected_control_key"] = list(REFERENCE_CONTROLS.keys())[0]

    for idx, (ctrl_key, ctrl_val) in enumerate(REFERENCE_CONTROLS.items()):
        is_active = st.session_state["selected_control_key"] == ctrl_key
        if btn_cols[idx].button(
            ctrl_key,
            key=f"ctrl_btn_{idx}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state["selected_control_key"] = ctrl_key
            st.rerun()

    selected_ctrl = st.session_state["selected_control_key"]
    ctrl_data = REFERENCE_CONTROLS[selected_ctrl]
    active_smiles = ctrl_data["smiles"]
    active_name = ctrl_data["name"]
    active_desc = ctrl_data["desc"]

elif input_mode == "✏️ Inserir SMILES (Texto)":
    col_input, col_preset = st.columns([3, 1])
    with col_preset:
        preset_choice = st.selectbox(
            "Atalhos rápidos:",
            ["(Personalizado)", "Cafeína", "Istradefilina", "Terfenadina", "Taurina", "L-Teanina"],
            index=0,
        )
    default_smi = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
    if preset_choice == "Cafeína":
        default_smi = REFERENCE_CONTROLS["☕ Cafeína (CHEMBL113)"]["smiles"]
    elif preset_choice == "Istradefilina":
        default_smi = REFERENCE_CONTROLS["💊 Istradefilina (CHEMBL431770)"]["smiles"]
    elif preset_choice == "Terfenadina":
        default_smi = REFERENCE_CONTROLS["⚠️ Terfenadina (CHEMBL22)"]["smiles"]
    elif preset_choice == "Taurina":
        default_smi = REFERENCE_CONTROLS["🌿 Taurina (Endógeno)"]["smiles"]
    elif preset_choice == "L-Teanina":
        default_smi = REFERENCE_CONTROLS["🍵 L-Teanina (Nutracêutico)"]["smiles"]

    with col_input:
        user_smiles = st.text_input(
            "Insira o código SMILES da molécula:",
            value=default_smi,
            placeholder="Ex: CCN1C(=O)C2=C(N=C(N2C)/C=C/C3=CC(=C(C=C3)OC)OC)N(C1=O)CC",
        )
    active_smiles = user_smiles.strip()

elif input_mode == "🎨 Desenhar Molécula (Ketcher)":
    try:
        from streamlit_ketcher import st_ketcher

        st.markdown("Desenhe ou edite a estrutura química no editor bidimensional abaixo e clique em **Apply**:")
        ketcher_smiles = st_ketcher("Cn1c(=O)c2c(ncn2C)n(C)c1=O", height=450)
        active_smiles = ketcher_smiles.strip() if ketcher_smiles else ""
        active_name = "Composto Desenhado (Ketcher)"
    except Exception as e:
        st.warning(f"Editor Ketcher indisponível ({e}). Por favor, utilize a entrada por código SMILES.")
        active_smiles = st.text_input("SMILES:", value="Cn1c(=O)c2c(ncn2C)n(C)c1=O")


# -----------------------------------------------------------------------------
# 10. Padronização, Featurização e Rastreamento de Bits (bitInfo RDKit)
# -----------------------------------------------------------------------------
if not active_smiles:
    st.warning("Por favor, selecione, insira ou desenhe uma molécula para iniciar a triagem.")
    st.stop()

canonical_smi = canonicalize_smiles(active_smiles)

if canonical_smi is None:
    st.error(
        f"❌ **Estrutura Química Inválida:** O RDKit não pôde interpretar o código submetido (`{active_smiles}`). "
        "Verifique a valência e sintaxe do código SMILES."
    )
    st.stop()

provenance = resolve_compound_provenance(canonical_smi)
active_name = provenance["name"]
active_desc = provenance["desc"]

mol_obj = Chem.MolFromSmiles(canonical_smi)
bit_info_map = extract_morgan_bit_info(mol_obj, radius=2, n_bits=1024)

df_input = pd.DataFrame([{"canonical_smiles": canonical_smi}])
df_feat = featurize_dataframe(df_input, "canonical_smiles")
X_sample = df_feat[list(FEATURE_NAMES)]


# -----------------------------------------------------------------------------
# 11. Inferência Multiobjetivo Concorrente
# -----------------------------------------------------------------------------
pred_pIC50 = float(model_a2a.predict(X_sample)[0])
pred_IC50_nM = float(10 ** (9.0 - pred_pIC50))

prob_herg = float(model_herg.predict_proba(X_sample)[0, 1])
safety_index = 1.0 - prob_herg

if pred_pIC50 >= 6.5:
    a2a_color_class = "semaphore-green"
    a2a_label = "🟢 Alta Potência Operacional (pIC50 ≥ 6,5)"
    a2a_desc = (
        "Predição do modelo na faixa operacional de alta potência (pIC50 ≥ 6,5). "
        "Verifique a proveniência experimental e o domínio de aplicabilidade antes de inferir atividade biológica."
    )
elif pred_pIC50 >= 5.0:
    a2a_color_class = "semaphore-yellow"
    a2a_label = "🟡 Potência Moderada Operacional (5,0 ≤ pIC50 < 6,5)"
    a2a_desc = (
        "Predição do modelo na faixa operacional micromolar moderada (5,0 ≤ pIC50 < 6,5). "
        "Compare com a medição experimental no dataset quando disponível."
    )
else:
    a2a_color_class = "semaphore-red"
    a2a_label = "🔴 Baixa Potência / Inativo Operacional (pIC50 < 5,0)"
    a2a_desc = "Predição do modelo abaixo do limiar operacional micromolar (pIC50 < 5,0)."

def classify_herg_operational_risk(prob_herg: float) -> Dict[str, str]:
    """
    Classifica o escore estimado P_hERG segundo a convenção canônica:
    - Baixo: p < 0.30
    - Intermediário: 0.30 <= p < 0.70
    - Alto: p >= 0.70
    """
    if prob_herg < 0.30:
        return {
            "risk_class": "baixo",
            "color_class": "semaphore-green",
            "bar_color": "#00D2BE",
            "label": "🟢 Classe Operacional: Baixo Risco (P_hERG < 0,30)",
            "desc": (
                "Escore de probabilidade estimado pelo modelo abaixo de 0,30 (classe operacional baixa). "
                "Trata-se de estimativa de classificador Random Forest não calibrado, não de probabilidade clínica direta."
            ),
        }
    elif prob_herg < 0.70:
        return {
            "risk_class": "intermediario",
            "color_class": "semaphore-yellow",
            "bar_color": "#F59E0B",
            "label": "🟡 Classe Operacional: Intermediário/Moderado (0,30 ≤ P_hERG < 0,70)",
            "desc": (
                "Escore de probabilidade estimado pelo modelo na faixa operacional intermediária (0,30 ≤ P_hERG < 0,70). "
                "Requer verificação experimental in vitro (ex.: patch-clamp) antes de conclusão farmacológica."
            ),
        }
    else:
        return {
            "risk_class": "alto",
            "color_class": "semaphore-red",
            "bar_color": "#FF4B4B",
            "label": "🔴 Classe Operacional: Alto Risco (P_hERG ≥ 0,70)",
            "desc": (
                "Escore de probabilidade estimado pelo modelo igual ou superior a 0,70 (P_hERG ≥ 0,70; classe operacional alta para bloqueio hERG). "
                "Alerta preditivo computacional; não equivale diretamente à probabilidade clínica de arritmia."
            ),
        }


herg_operational = classify_herg_operational_risk(prob_herg)
herg_color_class = herg_operational["color_class"]
herg_label = herg_operational["label"]
herg_desc = herg_operational["desc"]

render_html(build_control_provenance_html(provenance, pred_pIC50, prob_herg))


# -----------------------------------------------------------------------------
# 12. Painel de Decisão Multiobjetivo (Cartões Estilizados metric-card)
# -----------------------------------------------------------------------------
st.subheader("🎯 Painel de Decisão Multiobjetivo")

col_mol, col_sem1, col_sem2 = st.columns([1.15, 1.42, 1.42])

with col_mol:
    mol_b64 = render_molecule_dark_base64(mol_obj, size=(420, 250))
    if mol_b64:
        mol_img_html = (
            f'<div class="mol-canvas-frame">'
            f'<img src="data:image/png;base64,{mol_b64}" style="max-width:100%; height:180px; object-fit:contain; border-radius:8px;"/>'
            f'</div>'
        )
    else:
        mol_img_html = '<div style="color:#94A3B8;">Imagem não disponível</div>'

    render_html(
        f"""
        <div class="metric-card">
            <div class="card-header-title">
                <span>🧪 Estrutura 2D Padronizada (RDKit)</span>
                <span style="font-size:0.78rem; color:#00D2BE; font-weight:600;">{len(bit_info_map)} bits ativos</span>
            </div>
            {mol_img_html}
            <div class="smiles-box" title="SMILES Canônico Isomérico">{canonical_smi}</div>
        </div>
        """
    )

with col_sem1:
    ic50_str = f"{pred_IC50_nM:.1f} nM" if pred_IC50_nM < 1000 else f"{pred_IC50_nM / 1000:.2f} µM"
    gauge_val_pct = int(np.clip((pred_pIC50 - 4.0) / (10.0 - 4.0), 0.0, 1.0) * 100)
    a2a_sample_badge_cls = "insample-badge" if provenance["a2a_in_sample"] else "outsample-badge"
    a2a_sample_icon = "📌" if provenance["a2a_in_sample"] else "🔬"
    exp_kpi_str = (
        f"{provenance['a2a_exp_pic50']:.2f} (ChEMBL)"
        if provenance["a2a_exp_pic50"] is not None
        else "Ausente (Out-of-Sample)"
    )

    render_html(
        f"""
        <div class="metric-card">
            <div class="card-header-title">
                <span>1. Potência no Receptor A<sub>2A</sub> (Regressão)</span>
                <span style="font-size:0.78rem; color:#94A3B8;">XGBoost</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.4rem; margin-bottom:0.75rem;">
                <span class="{a2a_color_class}">{a2a_label}</span>
                <span class="{a2a_sample_badge_cls}">{a2a_sample_icon} {provenance["a2a_sample_status"]}</span>
            </div>
            <div class="kpi-grid">
                <div class="kpi-box">
                    <div class="kpi-label">Potência Predita (pIC50)</div>
                    <div class="kpi-value kpi-value-accent">{pred_pIC50:.2f}</div>
                    <div style="font-size:0.75rem; color:#94A3B8;">IC50 predito: {ic50_str}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Medição no Dataset (pIC50 exp.)</div>
                    <div class="kpi-value" style="font-size:1.22rem;">{exp_kpi_str}</div>
                    <div style="font-size:0.75rem; color:#94A3B8;">ID: {provenance["chembl_id"]}</div>
                </div>
            </div>
            <div style="background:#141820; border-radius:999px; height:10px; width:100%; overflow:hidden; margin:0.65rem 0;">
                <div style="background:linear-gradient(90deg, #00D2BE, #38BDF8); width:{gauge_val_pct}%; height:100%; border-radius:999px;"></div>
            </div>
            <p class="pharm-desc-text" style="font-size:0.84rem; color:#CBD5E1; margin-top:0.45rem;">
                <strong>Interpretação Operacional:</strong> {a2a_desc}
            </p>
        </div>
        """
    )

with col_sem2:
    herg_bar_pct = int(np.clip(prob_herg * 100, 0, 100))
    herg_bar_color = "#00D2BE" if prob_herg < 0.30 else ("#F59E0B" if prob_herg < 0.70 else "#FF4B4B")
    herg_sample_badge_cls = "insample-badge" if provenance["herg_in_sample"] else "outsample-badge"
    herg_sample_icon = "📌" if provenance["herg_in_sample"] else "🔬"

    render_html(
        f"""
        <div class="metric-card">
            <div class="card-header-title">
                <span>2. Triagem hERG (Canal K<sub>v</sub>11.1)</span>
                <span style="font-size:0.78rem; color:#94A3B8;">Random Forest (predict_proba)</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.4rem; margin-bottom:0.75rem;">
                <span class="{herg_color_class}">{herg_label}</span>
                <span class="{herg_sample_badge_cls}">{herg_sample_icon} {provenance["herg_sample_status"]}</span>
            </div>
            <div class="kpi-grid">
                <div class="kpi-box">
                    <div class="kpi-label">Escore Estimado P(hERG)</div>
                    <div class="kpi-value" style="color:{herg_bar_color};">{prob_herg:.1%}</div>
                    <div style="font-size:0.75rem; color:#94A3B8;">Escore do modelo (não calibrado)</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Índice Complementar (1 - P_hERG)</div>
                    <div class="kpi-value kpi-value-accent">{safety_index:.1%}</div>
                    <div style="font-size:0.75rem; color:#94A3B8;">Derivado matematicamente do escore</div>
                </div>
            </div>
            <div style="background:#141820; border-radius:999px; height:10px; width:100%; overflow:hidden; margin:0.65rem 0;">
                <div style="background:{herg_bar_color}; width:{herg_bar_pct}%; height:100%; border-radius:999px;"></div>
            </div>
            <p class="pharm-desc-text" style="font-size:0.84rem; color:#CBD5E1; margin-top:0.45rem;">
                <strong>Interpretação Operacional:</strong> {herg_desc}
            </p>
        </div>
        """
    )


# -----------------------------------------------------------------------------
# 13. Propriedades Físico-Químicas 2D & Regra de Lipinski (Cartões Estilizados)
# -----------------------------------------------------------------------------
desc_dict = compute_2d_descriptors(mol_obj)
mol_wt = desc_dict["MolWt"]
mol_logp = desc_dict["MolLogP"]
tpsa = desc_dict["TPSA"]
hbd = desc_dict["NumHDonors"]
hba = desc_dict["NumHAcceptors"]
rot_bonds = desc_dict["NumRotatableBonds"]

lipinski_violations = sum([
    mol_wt > 500,
    mol_logp > 5.0,
    hbd > 5,
    hba > 10,
])

with st.expander("📋 Propriedades Físico-Químicas 2D e Auditoria de Drogabilidade (Regra de Lipinski)", expanded=False):
    lip_badge = (
        '<span class="semaphore-green" style="font-size:0.88rem; padding:0.3rem 0.85rem;">✅ 0 Violações — Alta Drogabilidade Oral</span>'
        if lipinski_violations == 0
        else f'<span class="semaphore-yellow" style="font-size:0.88rem; padding:0.3rem 0.85rem;">⚠️ {lipinski_violations} Violação(ões) da Regra dos Cinco de Lipinski</span>'
    )
    render_html(
        f"""
        <div class="metric-card" style="margin-bottom:0.4rem;">
            <div class="card-header-title">
                <span>Parâmetros Físico-Químicos Calculados (RDKit 2D)</span>
                {lip_badge}
            </div>
            <div style="display:grid; grid-template-columns:repeat(6, 1fr); gap:0.75rem; margin-top:0.5rem;">
                <div class="kpi-box">
                    <div class="kpi-label">Massa (MolWt)</div>
                    <div class="kpi-value" style="font-size:1.25rem;">{mol_wt:.1f} <span style="font-size:0.8rem; color:#94A3B8;">Da</span></div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Lipofilicidade (LogP)</div>
                    <div class="kpi-value" style="font-size:1.25rem; color:#00D2BE;">{mol_logp:.2f}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Área Polar (TPSA)</div>
                    <div class="kpi-value" style="font-size:1.25rem;">{tpsa:.1f} <span style="font-size:0.8rem; color:#94A3B8;">Å²</span></div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Doadores H (HBD)</div>
                    <div class="kpi-value" style="font-size:1.25rem;">{hbd}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Aceptores H (HBA)</div>
                    <div class="kpi-value" style="font-size:1.25rem;">{hba}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Lig. Rotacionáveis</div>
                    <div class="kpi-value" style="font-size:1.25rem;">{rot_bonds}</div>
                </div>
            </div>
        </div>
        """
    )


# -----------------------------------------------------------------------------
# 14. Matriz Multiobjetivo 2D de Decisão Farmacológica (Alto Contraste Dark Mode)
# -----------------------------------------------------------------------------
def build_multiobjective_figure(
    x_cur: float,
    y_cur: float,
    active_name: str,
) -> Tuple[plt.Figure, plt.Axes]:
    """Constrói o gráfico multiobjetivo (pIC50 vs. 1 - P_hERG) em coordenadas estritas de dados."""
    fig_mat, ax_mat = plt.subplots(figsize=(8.2, 5.2), facecolor="#0E1117")
    ax_mat.set_facecolor("#1E222D")

    # Quatro linhas explícitas de referência delimitando as faixas operacionais
    ax_mat.axvline(
        5.0,
        color="#94A3B8",
        linestyle=":",
        linewidth=1.1,
        alpha=0.80,
        label="Limiar Potência Moderada A2A (pIC50 = 5,0)",
    )
    ax_mat.axvline(
        6.5,
        color="#00D2BE",
        linestyle="--",
        linewidth=1.3,
        alpha=0.85,
        label="Limiar Alta Potência A2A (pIC50 = 6,5)",
    )
    ax_mat.axhline(
        0.70,
        color="#00D2BE",
        linestyle=":",
        linewidth=1.3,
        alpha=0.85,
        label="Limiar Operacional Baixo Risco hERG (1 - P_hERG = 0,70 | P_hERG = 0,30)",
    )
    ax_mat.axhline(
        0.30,
        color="#FF4B4B",
        linestyle=":",
        linewidth=1.1,
        alpha=0.80,
        label="Limiar Operacional Alto Risco hERG (1 - P_hERG = 0,30 | P_hERG = 0,70)",
    )

    # Sombreamento em coordenadas cartesianas absolutas de dados (X: pIC50, Y: 1 - P_hERG)
    ax_mat.fill_between(
        [6.5, 11.5],
        0.70,
        1.00,
        color="#00D2BE",
        alpha=0.14,
        label="Zona Ideal Operacional (pIC50 ≥ 6,5 e P_hERG < 0,30)",
    )
    ax_mat.fill_between(
        [6.5, 11.5],
        0.00,
        0.30,
        color="#FF4B4B",
        alpha=0.14,
        label="Potente porém Alto Risco Operacional hERG (pIC50 ≥ 6,5 e P_hERG ≥ 0,70)",
    )
    ax_mat.fill_between(
        [3.5, 5.0],
        0.00,
        0.30,
        color="#FF4B4B",
        alpha=0.08,
        label="Inativo & Alto Risco Operacional hERG (pIC50 < 5,0 e P_hERG ≥ 0,70)",
    )

    ax_mat.text(
        9.0,
        0.95,
        "QUADRANTE IDEAL 🏆\nPotente & Baixo Escore hERG",
        ha="center",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color="#00F5D4",
    )
    ax_mat.text(
        9.0,
        0.05,
        "DESCARTE CRÍTICO ⚠️\nPotente porém Alto Escore hERG",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#FF7676",
    )
    ax_mat.text(
        4.25,
        0.95,
        "INATIVO & BAIXO ESCORE hERG\nBaixo Interesse Terapêutico",
        ha="center",
        va="top",
        fontsize=8.8,
        fontweight="bold",
        color="#CBD5E1",
    )
    ax_mat.text(
        4.25,
        0.05,
        "INATIVO & ALTO ESCORE hERG\nReprovado",
        ha="center",
        va="bottom",
        fontsize=8.8,
        fontweight="bold",
        color="#94A3B8",
    )

    ax_mat.scatter(x_cur, y_cur, s=300, color="#00D2BE", edgecolors="#FFFFFF", linewidths=2.5, zorder=6)

    offset_x = -2.1 if x_cur > 9.2 else 0.22
    offset_y = -0.14 if y_cur > 0.88 else 0.04
    ax_mat.annotate(
        f"{active_name}\n(pIC50={x_cur:.2f} | 1-P={y_cur:.1%})",
        (x_cur + offset_x, y_cur + offset_y),
        fontsize=9.5,
        fontweight="bold",
        color="#FFFFFF",
        bbox=dict(boxstyle="round,pad=0.35", fc="#0E1117", ec="#00D2BE", lw=1.5, alpha=0.95),
    )

    ax_mat.set_xlabel("Potência Predita A2A: pIC50 (Maior = Maior Potência Estimada)", fontsize=10, fontweight="bold", color="#FFFFFF")
    ax_mat.set_ylabel("Índice Complementar Derivado: 1 - P(hERG) [Escore do Modelo]", fontsize=10, fontweight="bold", color="#FFFFFF")
    ax_mat.set_title("Posicionamento Cartesiano Multiobjetivo (Classes Operacionais)", fontsize=11.5, fontweight="bold", color="#FFFFFF")
    ax_mat.set_xlim(3.5, 11.5)
    ax_mat.set_ylim(-0.05, 1.05)
    ax_mat.tick_params(colors="#E2E8F0", labelsize=9)
    for spine in ax_mat.spines.values():
        spine.set_color("#475569")
    ax_mat.grid(True, linestyle=":", alpha=0.35, color="#64748B")

    plt.tight_layout()
    return fig_mat, ax_mat


def classify_herg_safety_axis(y_cur: float, atol: float = 1e-12) -> str:
    """
    Classifica o eixo complementar y = 1 - P_hERG segundo a convenção canônica:
    - Baixo risco: y > 0.70
    - Intermediário: 0.30 < y <= 0.70
    - Alto risco: y <= 0.30
    Utiliza tolerância de ponto flutuante (np.isclose) nas fronteiras 0.30 e 0.70
    para evitar desvios de representação binária IEEE 754 em y = 1.0 - p.
    """
    if y_cur <= 0.30 or bool(np.isclose(y_cur, 0.30, rtol=0.0, atol=atol)):
        return "alto"
    elif y_cur > 0.70 and not bool(np.isclose(y_cur, 0.70, rtol=0.0, atol=atol)):
        return "baixo"
    else:
        return "intermediario"


def classify_multiobjective_quadrant(
    x_cur: float,
    y_cur: float,
    atol: float = 1e-12,
) -> Dict[str, str]:
    """
    Classifica a posição cartesiana (x_cur = pIC50, y_cur = 1 - P_hERG) nos quadrantes
    operacionais, garantindo que y = 0.30 (P_hERG = 0.70) pertença ao risco alto (y <= 0.30)
    e que o intervalo intermediário seja estritamente 0.30 < y <= 0.70 (0.30 <= P_hERG < 0.70).
    """
    herg_y_class = classify_herg_safety_axis(y_cur, atol=atol)

    if x_cur >= 6.5 and herg_y_class == "baixo":
        return {
            "herg_class": herg_y_class,
            "quad_badge_cls": "semaphore-green",
            "quad_title": "🏆 Quadrante 1 — Zona Ideal Operacional",
            "quad_text": (
                "Potência predita nanomolar no receptor A2A (pIC50 ≥ 6,5) combinada a baixo escore estimado de "
                "bloqueio hERG (P_hERG < 0,30; 1 - P_hERG > 0,70). Candidato prioritário para validação experimental."
            ),
        }
    elif x_cur >= 6.5 and herg_y_class == "alto":
        return {
            "herg_class": herg_y_class,
            "quad_badge_cls": "semaphore-red",
            "quad_title": "⚠️ Quadrante 4 — Potente porém Alto Escore hERG",
            "quad_text": (
                "O composto exibe alta potência predita em A2A, mas apresenta escore elevado de probabilidade "
                "estimada para bloqueio hERG (P_hERG ≥ 0,70; 1 - P_hERG ≤ 0,30). Recomenda-se redução de MolLogP ou atenuação de aminas básicas."
            ),
        }
    elif x_cur < 5.0 and herg_y_class == "baixo":
        return {
            "herg_class": herg_y_class,
            "quad_badge_cls": "semaphore-yellow",
            "quad_title": "🌿 Quadrante 2 — Baixa Potência Predita & Baixo Escore hERG",
            "quad_text": (
                "Baixo escore estimado de bloqueio hERG (P_hERG < 0,30; 1 - P_hERG > 0,70), porém potência predita em A2A abaixo do "
                "limiar operacional micromolar (pIC50 < 5,0)."
            ),
        }
    elif x_cur < 5.0 and herg_y_class == "alto":
        return {
            "herg_class": herg_y_class,
            "quad_badge_cls": "semaphore-red",
            "quad_title": "❌ Quadrante 3 — Baixa Potência Predita & Alto Escore hERG",
            "quad_text": "Baixa potência predita em A2A aliada a alto escore operacional de bloqueio hERG (P_hERG ≥ 0,70; 1 - P_hERG ≤ 0,30).",
        }
    else:
        return {
            "herg_class": herg_y_class,
            "quad_badge_cls": "semaphore-yellow",
            "quad_title": "🟡 Zona de Transição Operacional / Otimização Hit-to-Lead",
            "quad_text": (
                "Composto situado na faixa operacional intermediária de potência predita A2A (5,0 ≤ pIC50 < 6,5) "
                "e/ou de escore estimado hERG (0,30 ≤ P_hERG < 0,70; 0,30 < 1 - P_hERG ≤ 0,70). Utilize a decomposição SHAP abaixo para análise estrutural."
            ),
        }


st.subheader("🗺️ Matriz de Decisão Multiobjetivo (Potência Predita A2A × Índice Complementar 1 - P_hERG)")

col_quad, col_quad_desc = st.columns([2.0, 1.25])

x_cur = pred_pIC50
y_cur = safety_index

with col_quad:
    fig_mat, ax_mat = build_multiobjective_figure(x_cur, y_cur, active_name)
    st.pyplot(fig_mat)
    plt.close(fig_mat)

with col_quad_desc:
    quad_info = classify_multiobjective_quadrant(x_cur, y_cur)
    quad_badge_cls = quad_info["quad_badge_cls"]
    quad_title = quad_info["quad_title"]
    quad_text = quad_info["quad_text"]

    render_html(
        f"""
        <div class="metric-card" style="height:100%;">
            <div class="card-header-title">
                <span>🧭 Diagnóstico Operacional de Posicionamento</span>
            </div>
            <div style="margin-bottom:0.85rem;">
                <span class="{quad_badge_cls}" style="font-size:0.9rem;">{quad_title}</span>
            </div>
            <p class="pharm-desc-text" style="margin-bottom:1rem;">{quad_text}</p>
            <div class="kpi-box" style="margin-bottom:0.6rem;">
                <div class="kpi-label">Eixo Horizontal (X) — Potência Predita A2A</div>
                <div style="color:#FFFFFF; font-size:0.9rem;">
                    <b>pIC50 predito = {x_cur:.2f}</b> (Classes operacionais: &lt; 5,0 | 5,0–6,5 | ≥ 6,5)
                </div>
            </div>
            <div class="kpi-box">
                <div class="kpi-label">Eixo Vertical (Y) — Índice Complementar Derivado</div>
                <div style="color:#FFFFFF; font-size:0.9rem;">
                    <b>1 - P(hERG) = {y_cur:.1%}</b> (Escore P_hERG = {prob_herg:.1%} | Baixo risco operacional: P &lt; 0,30)
                </div>
            </div>
        </div>
        """
    )


# -----------------------------------------------------------------------------
# 15. Aviso de Amostragem do SHAP (Top 10) & Seção de Explicabilidade Química
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔍 Interpretabilidade Química e Farmacofórica Local (XAI — SHAP & RDKit)")

render_html(
    """
    <div class="xai-audit-banner">
        <strong>Auditoria Local de Atributos (XAI):</strong> Exibindo exclusivamente os
        <strong>10 principais atributos (Top 10)</strong> com maior peso absoluto de decisão nesta predição.
        As demais 1.020 dimensões de Morgan Fingerprint (ECFP4) e descritores secundários possuem
        contribuição residual para este composto específico.
    </div>
    """
)

tab_shap_a2a, tab_shap_herg = st.tabs([
    "🎯 Farmacóforos e Descritores da Potência A2A",
    "❤️ Alertas Estruturais e Descritores do Risco hERG",
])


# -----------------------------------------------------------------------------
# Aba 1: Explicabilidade Química A2A
# -----------------------------------------------------------------------------
with tab_shap_a2a:
    with st.spinner("Mapeando contribuições farmacofóricas locais (SHAP + RDKit bitInfo) para A2A..."):
        sv_a2a = explainer_a2a(X_sample)
        vals_a2a = sv_a2a.values[0]
        base_a2a = float(sv_a2a.base_values[0])

        top_idx_a2a = np.argsort(np.abs(vals_a2a))[::-1][:10]
        top_names_a2a = [FEATURE_NAMES[i] for i in top_idx_a2a]
        top_impacts_a2a = [float(vals_a2a[i]) for i in top_idx_a2a]
        top_mol_vals_a2a = [float(X_sample.iloc[0, i]) for i in top_idx_a2a]

        labels_a2a = []
        for n, v in zip(top_names_a2a, top_mol_vals_a2a):
            det = resolve_pharmacophore_details(n, mol_obj, bit_info_map, target="A2A")
            short_lbl = det["label"][:38] + ("…" if len(det["label"]) > 38 else "")
            if n in DESCRIPTOR_NAMES:
                labels_a2a.append(f"{n} ({short_lbl}) = {v:.2f}")
            else:
                state_str = "Presente" if int(round(v)) == 1 else "Ausente"
                labels_a2a.append(f"{n} · {short_lbl} [{state_str}]")

        fig_sa, ax_sa = plt.subplots(figsize=(10.2, 4.6), facecolor="#0E1117")
        ax_sa.set_facecolor("#1E222D")
        bar_colors_a2a = ["#00D2BE" if imp >= 0 else "#FF4B4B" for imp in top_impacts_a2a]

        y_pos = np.arange(len(top_names_a2a))
        ax_sa.barh(y_pos, top_impacts_a2a, color=bar_colors_a2a, edgecolor="#FFFFFF", linewidth=0.6, alpha=0.92)
        ax_sa.set_yticks(y_pos)
        ax_sa.set_yticklabels(labels_a2a, fontsize=9, fontweight="bold", color="#FFFFFF")
        ax_sa.invert_yaxis()
        ax_sa.axvline(0, color="#CBD5E1", linestyle="-", linewidth=1.1)
        ax_sa.set_xlabel(
            "Impacto SHAP no pIC50 (Ciano/Verde = Aumenta Afinidade A2A | Vermelho = Atenua Afinidade)",
            fontsize=9.5,
            fontweight="bold",
            color="#FFFFFF",
        )
        ax_sa.set_title(
            f"Decomposição Aditiva SHAP — Top 10 Atributos para A2A (Linha de Base = {base_a2a:.2f} → Predito = {pred_pIC50:.2f})",
            fontsize=11,
            fontweight="bold",
            color="#FFFFFF",
        )
        ax_sa.tick_params(colors="#E2E8F0", labelsize=9)
        for spine in ax_sa.spines.values():
            spine.set_color("#475569")
        ax_sa.grid(axis="x", linestyle=":", alpha=0.35, color="#64748B")
        plt.tight_layout()
        st.pyplot(fig_sa)
        plt.close(fig_sa)

        st.markdown("#### 🧩 Grade Visual de Farmacóforos e Descritores Decisivos (Receptor $A_{2A}$)")
        st.caption(
            "Os fragmentos moleculares abaixo são extraídos diretamente da molécula em análise via `bitInfo` do RDKit "
            "com fundo transparente Dark Mode (destaque ciano = átomo central do bit; âmbar = átomos aromáticos do ambiente circular ECFP4)."
        )

        for row_start in range(0, len(top_names_a2a), 2):
            grid_cols = st.columns(2)
            for col_offset in range(2):
                item_idx = row_start + col_offset
                if item_idx < len(top_names_a2a):
                    card_html = build_pharmacophore_card_html(
                        feature_name=top_names_a2a[item_idx],
                        mol_value=top_mol_vals_a2a[item_idx],
                        shap_impact=top_impacts_a2a[item_idx],
                        mol=mol_obj,
                        bit_info=bit_info_map,
                        target="A2A",
                    )
                    with grid_cols[col_offset]:
                        st.markdown(card_html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Aba 2: Explicabilidade Química hERG
# -----------------------------------------------------------------------------
with tab_shap_herg:
    with st.spinner("Mapeando alertas estruturais e descritores locais (SHAP + RDKit bitInfo) para hERG..."):
        sv_herg = explainer_herg(X_sample)
        vals_herg = sv_herg.values[0, :, 1]
        base_herg = float(sv_herg.base_values[0, 1])

        top_idx_herg = np.argsort(np.abs(vals_herg))[::-1][:10]
        top_names_herg = [FEATURE_NAMES[i] for i in top_idx_herg]
        top_impacts_herg = [float(vals_herg[i]) for i in top_idx_herg]
        top_mol_vals_herg = [float(X_sample.iloc[0, i]) for i in top_idx_herg]

        labels_herg = []
        for n, v in zip(top_names_herg, top_mol_vals_herg):
            det = resolve_pharmacophore_details(n, mol_obj, bit_info_map, target="hERG")
            short_lbl = det["label"][:38] + ("…" if len(det["label"]) > 38 else "")
            if n in DESCRIPTOR_NAMES:
                labels_herg.append(f"{n} ({short_lbl}) = {v:.2f}")
            else:
                state_str = "Presente" if int(round(v)) == 1 else "Ausente"
                labels_herg.append(f"{n} · {short_lbl} [{state_str}]")

        fig_sh, ax_sh = plt.subplots(figsize=(10.2, 4.6), facecolor="#0E1117")
        ax_sh.set_facecolor("#1E222D")
        bar_colors_herg = ["#FF4B4B" if imp >= 0 else "#00D2BE" for imp in top_impacts_herg]

        y_pos = np.arange(len(top_names_herg))
        ax_sh.barh(y_pos, top_impacts_herg, color=bar_colors_herg, edgecolor="#FFFFFF", linewidth=0.6, alpha=0.92)
        ax_sh.set_yticks(y_pos)
        ax_sh.set_yticklabels(labels_herg, fontsize=9, fontweight="bold", color="#FFFFFF")
        ax_sh.invert_yaxis()
        ax_sh.axvline(0, color="#CBD5E1", linestyle="-", linewidth=1.1)
        ax_sh.set_xlabel(
            "Impacto SHAP em P(hERG) (Vermelho = Aumenta Risco Cardiotóxico | Ciano/Verde = Efeito Protetor)",
            fontsize=9.5,
            fontweight="bold",
            color="#FFFFFF",
        )
        ax_sh.set_title(
            f"Decomposição Aditiva SHAP — Top 10 Atributos para Risco hERG (Base = {base_herg:.1%} → Predito = {prob_herg:.1%})",
            fontsize=11,
            fontweight="bold",
            color="#FFFFFF",
        )
        ax_sh.tick_params(colors="#E2E8F0", labelsize=9)
        for spine in ax_sh.spines.values():
            spine.set_color("#475569")
        ax_sh.grid(axis="x", linestyle=":", alpha=0.35, color="#64748B")
        plt.tight_layout()
        st.pyplot(fig_sh)
        plt.close(fig_sh)

        st.markdown("#### 🧩 Grade Visual de Alertas Estruturais e Descritores Decisivos (Canal hERG)")
        st.caption(
            "Avaliação mecanística de subestruturas lipofílicas, centros básicos protonáveis e descritores de "
            "superfície polar sobre a probabilidade de oclusão do canal de potássio Kv11.1."
        )

        for row_start in range(0, len(top_names_herg), 2):
            grid_cols = st.columns(2)
            for col_offset in range(2):
                item_idx = row_start + col_offset
                if item_idx < len(top_names_herg):
                    card_html = build_pharmacophore_card_html(
                        feature_name=top_names_herg[item_idx],
                        mol_value=top_mol_vals_herg[item_idx],
                        shap_impact=top_impacts_herg[item_idx],
                        mol=mol_obj,
                        bit_info=bit_info_map,
                        target="hERG",
                    )
                    with grid_cols[col_offset]:
                        st.markdown(card_html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 16. Diagnóstico de Cobertura Estrutural (Domínio de Aplicabilidade)
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("🧭 Diagnóstico de Cobertura Estrutural (Domínio de Aplicabilidade)")

try:
    ref_a2a = load_ad_reference_cached("A2A")
    diag_a2a = compute_ad_diagnostics(mol_obj, "A2A", reference=ref_a2a)
except Exception as exc:
    diag_a2a = {
        "status": "technical_error",
        "endpoint": "A2A",
        "error_message": f"Não foi possível calcular o diagnóstico: {exc}",
    }

try:
    ref_herg = load_ad_reference_cached("hERG")
    diag_herg = compute_ad_diagnostics(mol_obj, "hERG", reference=ref_herg)
except Exception as exc:
    diag_herg = {
        "status": "technical_error",
        "endpoint": "hERG",
        "error_message": f"Não foi possível calcular o diagnóstico: {exc}",
    }

render_html(build_ad_section_html(diag_a2a, diag_herg))


# -----------------------------------------------------------------------------
# 17. Rodapé e Governança Técnica
# -----------------------------------------------------------------------------
st.markdown("---")
footer_c1, footer_c2 = st.columns([2, 1])
footer_c1.caption(
    "**MultiTarget BioPredict** — Plataforma Acadêmica de Triagem Virtual Multiobjetivo. "
    "Desenvolvido por Raphael Pires. Licença MIT. RDKit · scikit-learn · XGBoost · SHAP · Streamlit."
)
footer_c2.caption(
    "Ambiente 100% Local · Restrições de 8 GB RAM respeitadas · Sem dependência de nuvem."
)
