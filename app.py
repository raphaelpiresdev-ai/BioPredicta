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
from src.pubchem_utils import (
    build_pubchem_card_html,
    identify_compound_pubchem,
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
# -----------------------------------------------------------------------------
# 2. Design System Centralizado (src.ui.theme)
# -----------------------------------------------------------------------------
from src.ui.theme import apply_theme
apply_theme()


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
    926: {
        "label": "Carbono Alifático Não Funcionalizado (Raio 0)",
        "scaffold": "Átomo de carbono alifático isolado [C;r0]",
        "role_a2a": (
            "Hipótese da literatura: Centro carbonado alifático simples; contribui para a hidrofobicidade geral e empacotamento apolar."
        ),
        "role_herg": (
            "Hipótese da literatura: Cadeias alquílicas modulam incrementalmente a lipofilicidade global da molécula."
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

BIT_TRANSLATION_MAP: Dict[str, str] = {
    # Bits de Morgan Fingerprint (ECFP4) recorrentes e decisivos
    "fp_831": "Aceptor de H em Anel Heteroaromático (N sp²)",
    "fp_964": "Núcleo Purínico / Doador-Aceptor de H",
    "fp_904": "Substituinte Fluoroarílico",
    "fp_234": "Núcleo Imídico / Xantínico N-Metilado",
    "fp_564": "Nitrogênio Purínico N-Alquilado",
    "fp_575": "Sistema Xantínico N-Etilado",
    "fp_517": "Aceptor Carbonílico Purínico",
    "fp_34": "Substituinte N-Etil / Alquil-Heterociclo",
    "fp_465": "Junção N-Alquil-Purinona",
    "fp_850": "Ponte Alquil-Heteroaromática",
    "fp_128": "Amina Alifática / Centro Básico",
    "fp_147": "Grupo Amino Primário / Secundário Polar",
    "fp_672": "Junção de Anel Purínico / Heteroaromático",
    "fp_433": "Amina Exocíclica Conjugada (-NH2)",
    "fp_323": "Eixo Biarílico / Heteroaril-Aril Conjugado",
    "fp_312": "Anel Furano / Oxazol Heteroaromático",
    "fp_829": "Anel Fenila Para-Substituído",
    "fp_338": "Metileno Benzílico / Carbinol Arílico",
    "fp_887": "Cadeia Espaçadora Alifática (-CH2-CH2-)",
    "fp_767": "Amina Terciária / Centro Ramificado",
    "fp_807": "Grupo Hidroxila / Éter Polar (-OH / -O-)",
    "fp_893": "Oxigênio Carbonílico Heterocíclico",
    "fp_726": "Arcabouço Aromático Benzenoide",
    "fp_650": "Centro Oxigenado Polar",
    "fp_653": "Cadeia Alquil-Amina Terciária",
    "fp_121": "Ligação Carbono-Nitrogênio",
    "fp_549": "Núcleo Bicíclico Xantínico",
    "fp_523": "Substituinte C8-Purínico / Metóxi",
    "fp_219": "Ponte Etil-Arílica / Benzílica",
    "fp_80": "Carbono Alifático Saturado (sp³)",
    "fp_1": "Centro Carbonado Substituído",
    "fp_484": "Subestrutura Heteroaromática Nitrogenada",
    "fp_825": "Espaçador Alifático Adjacente a Heteroátomo",
    "fp_926": "Carbono Alifático (Raio 0)",
    # Descritores Físico-Químicos 2D Globais
    "MolWt": "Massa Molecular Média (MolWt)",
    "MolLogP": "Lipofilicidade Calculada (MolLogP)",
    "TPSA": "Área de Superfície Polar Topológica (TPSA)",
    "NumHDonors": "Doadores de Ligação de Hidrogênio (HBD)",
    "NumHAcceptors": "Aceptores de Ligação de Hidrogênio (HBA)",
    "NumRotatableBonds": "Ligações Rotacionáveis (Flexibilidade)",
}


def resolve_translated_feature_name(
    feature_name: str,
    mol: Optional[Chem.Mol] = None,
    bit_info: Optional[Dict[int, Any]] = None,
) -> str:
    """Retorna a denominação química traduzida e didática para qualquer atributo (bit ou descritor)."""
    if feature_name in BIT_TRANSLATION_MAP:
        return BIT_TRANSLATION_MAP[feature_name]
    if feature_name in DESCRIPTOR_INFO_CATALOG:
        return DESCRIPTOR_INFO_CATALOG[feature_name]["label"]
    if feature_name.startswith("fp_"):
        try:
            bit_id = int(feature_name.split("_")[1])
            if bit_id in PHARMACOPHORE_BIT_CATALOG:
                return PHARMACOPHORE_BIT_CATALOG[bit_id]["label"]
            if mol is not None and bit_info is not None:
                det = resolve_pharmacophore_details(feature_name, mol, bit_info)
                return det.get("label", f"Subestrutura ECFP4 (#{bit_id})")
        except Exception:
            pass
    return feature_name


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
    translated_name = resolve_translated_feature_name(feature_name, mol, bit_info)
    details = resolve_pharmacophore_details(feature_name, mol, bit_info, target=target)
    is_descriptor = feature_name in DESCRIPTOR_NAMES

    if target == "A2A":
        is_favorable = shap_impact >= 0
        effect_word = "Aumenta Afinidade" if shap_impact >= 0 else "Atenua Afinidade"
        unit_suffix = " pIC50"
    else:
        is_favorable = shap_impact < 0
        effect_word = "Reduz Risco" if shap_impact < 0 else "Aumenta Risco hERG"
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
            f'<div class="pharm-body-present">'
            f'<div class="bit-img-wrapper" style="border: 1.5px solid rgba(0, 210, 190, 0.45); background: rgba(14, 17, 23, 0.85); display: flex; flex-direction: column; align-items: center; justify-content: center; height: 120px;">'
            f'<div style="font-size: 2rem; line-height: 1;">📐</div>'
            f'<div style="font-size: 0.78rem; color: #00D2BE; font-weight: 700; margin-top: 4px;">{val_formatted} {details["unit"]}</div>'
            f'<div style="font-size: 0.68rem; color: #94A3B8; margin-top: 2px;">Propriedade 2D</div>'
            f'</div>'
            f'<div style="flex:1;">'
            f'<div class="descriptor-val-badge">📐 Descritor Físico-Químico: <b style="color:#00D2BE;">{val_formatted} {details["unit"]}</b></div>'
            f'<p class="pharm-desc-text" style="margin-top:0.35rem;">{details["role"]}</p>'
            f'</div>'
            f'</div>'
        )
    else:
        bit_id = int(feature_name.split("_")[1])
        is_present = int(round(mol_value)) == 1

        if is_present:
            b64_img = render_morgan_bit_base64(mol, bit_id, bit_info)
            frag_smi = details.get("frag_smi", f"fp_{bit_id}")
            if b64_img:
                img_block = (
                    f'<div class="bit-img-wrapper">'
                    f'<img src="data:image/png;base64,{b64_img}" alt="{feature_name}"/>'
                    f'<div class="bit-smarts-caption" title="{frag_smi}">{frag_smi}</div>'
                    f'</div>'
                )
            else:
                img_block = (
                    f'<div class="bit-img-wrapper" style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:120px; background:rgba(14,17,23,0.85);">'
                    f'<span style="font-size:1.8rem;">🔬</span>'
                    f'<span style="font-size:0.70rem; color:#00D2BE; font-weight:600; margin-top:4px;">Microambiente Ativo</span>'
                    f'</div>'
                )
            body_html = (
                f'<div class="pharm-body-present">'
                f'{img_block}'
                f'<div style="flex:1;">'
                f'<div class="present-badge">✅ Subestrutura Presente (Bit Ativo = 1)</div>'
                f'<p class="pharm-desc-text" style="margin-top:0.35rem;">{details["role"]}</p>'
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
            absent_icon_block = (
                f'<div class="bit-img-wrapper" style="border: 1.5px dashed rgba(255, 75, 75, 0.50); background: rgba(14, 17, 23, 0.85); display: flex; flex-direction: column; align-items: center; justify-content: center; height: 120px;">'
                f'<div style="font-size: 2.2rem; line-height: 1;">❌</div>'
                f'<div style="font-size: 0.72rem; color: #FF4B4B; font-weight: 700; margin-top: 6px;">Bit Ausente</div>'
                f'</div>'
            )
            body_html = (
                f'<div class="pharm-body-present">'
                f'{absent_icon_block}'
                f'<div style="flex:1;">'
                f'<div class="absent-badge">❌ Subestrutura Ausente (Bit = 0)</div>'
                f'<p class="pharm-desc-text" style="margin-bottom:0.35rem; margin-top:0.35rem;">'
                f'<strong>Referência:</strong> {details["scaffold"]}'
                f'</p>'
                f'<p class="pharm-desc-text">{absence_reason} {details["role"]}</p>'
                f'</div>'
                f'</div>'
            )

    card_html = (
        f'<div class="pharmacophore-card {card_border_cls}">'
        f'<div class="pharm-card-header">'
        f'<div>'
        f'<span class="pharm-feature-id">{feature_name}</span>'
        f'<div class="pharm-feature-title">{translated_name}</div>'
        f'</div>'
        f'<span class="{tag_cls}">{shap_str}</span>'
        f'</div>'
        f'{body_html}'
        f'</div>'
    )
    clean_lines = [line.strip() for line in card_html.splitlines() if line.strip()]
    return "\n".join(clean_lines)


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
    <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:0.75rem;">
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
# 6.1 Funções Canônicas de Classificação e Gráficos (Fachada Pública de Testes)
# -----------------------------------------------------------------------------
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

def build_multiobjective_figure(
    x_cur: float,
    y_cur: float,
    active_name: str,
) -> Tuple[plt.Figure, plt.Axes]:
    """Constrói o gráfico multiobjetivo (pIC50 vs. 1 - P_hERG) em coordenadas estritas de dados."""
    fig_mat, ax_mat = plt.subplots(figsize=(10.5, 5.2), facecolor="#0E1117")
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
        "QUADRANTE IDEAL\nPotente & Baixo Escore hERG",
        ha="center",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color="#00F5D4",
    )
    ax_mat.text(
        9.0,
        0.05,
        "DESCARTE CRÍTICO\nPotente porém Alto Escore hERG",
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


# -----------------------------------------------------------------------------
# 7. Execução Legada em Modo de Transição
# -----------------------------------------------------------------------------
def render_legacy_analysis_view() -> None:
    """Executa a visualização e inferência funcional do modo legado."""
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
            <div class="metric-card" style="padding:0.85rem 1rem; margin-bottom:0.5rem;">
                <div style="color:#00D2BE; font-weight:700; font-size:0.88rem;">🎯 Eficácia A<sub>2A</sub></div>
                <div style="font-size:0.80rem; color:#CBD5E1; margin-top:0.25rem;">
                    Predição contínua de afinidade (pIC<sub>50</sub> / IC<sub>50</sub>) via ensemble XGBoost.
                </div>
            </div>
            <div class="metric-card" style="padding:0.85rem 1rem; margin-bottom:0.5rem;">
                <div style="color:#FF4B4B; font-weight:700; font-size:0.88rem;">❤️ Risco Cardíaco hERG</div>
                <div style="font-size:0.80rem; color:#CBD5E1; margin-top:0.25rem;">
                    Triagem antitóxica do canal K<sub>v</sub>11.1 com classificação semafórica de risco.
                </div>
            </div>
            """
        )
        with st.expander("ℹ️ Metadados e Governança dos Modelos", expanded=False):
            st.markdown(
                """
                * **A2A (XGBoost Regressor):**
                  Holdout 80/20 prévio (n=273): $R^2 = 0{,}723$ | $\\text{RMSE} = 0{,}578$. Artefato final reajustado em 100% dos dados (1.364 compostos).
                * **hERG (Random Forest Classifier):**
                  Holdout 80/20 prévio (n=2.628): $\\text{ROC-AUC} = 0{,}930$ | $\\text{Brier} = 0{,}112$. Artefato final reajustado em 100% (13.136 compostos); escore via `predict_proba` sem calibração pós-hoc.
                * *Nota:* As métricas de validação referem-se à divisão holdout independente anterior ao reajuste final.
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
        st.caption("100% Local · AMD Ryzen 7 CPU · 8 GB RAM")


    # -----------------------------------------------------------------------------
    # 8. Cabeçalho Principal e Guia de Onboarding Didático
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

    with st.expander("📖 Guia Rápido: Como interpretar os resultados", expanded=False):
        col_guide_bio, col_guide_xai = st.columns(2)
        with col_guide_bio:
            st.markdown(
                r"""
                #### 🧬 Alvos Biológicos & Decisão Farmacológica
                * **🎯 Eficácia Terapêutica (Receptor $A_{2A}$):**
                  Avalia a capacidade da molécula de inibir o receptor de adenosina $A_{2A}$,
                  alvo farmacológico de relevância para doenças neurodegenerativas (ex: Parkinson),
                  imuno-oncologia e modulação inflamatória.
                  * A potência é expressa em **$pIC_{50} = -\log_{10}(IC_{50})$**.
                  * Quanto maior o $pIC_{50}$ (ex.: $\ge 6{,}5$, correspondendo a $IC_{50} \le 316\text{ nM}$),
                    mais potente é a molécula contra o alvo.
                * **❤️ Segurança Cardíaca (Bloqueio do Canal hERG):**
                  O canal de potássio cardíaco $K_v11.1$ (hERG) é essencial para a repolarização elétrica cardíaca.
                  * O bloqueio indesejado desse canal pode prolongar o intervalo QT e induzir arritmias ventriculares graves (*Torsades de Pointes*).
                  * O modelo estima a **Probabilidade $P(\text{hERG})$**: escores abaixo de **$30\%$** indicam perfil seguro (baixo risco),
                    enquanto valores a partir de **$70\%$** acionam alerta vermelho de cardiotoxicidade.
                """
            )
        with col_guide_xai:
            st.markdown(
                r"""
                #### 🔍 Explicabilidade Química SHAP (XAI)
                * **🧩 A IA Dissecando a Molécula em Fragmentos:**
                  Em vez de uma caixa-preta opaca, o sistema utiliza a metodologia **SHAP** (*SHapley Additive exPlanations*)
                  combinada a impressões digitais circulares (**ECFP4 / Morgan Fingerprints**) e descritores físico-químicos (RDKit).
                * **⚖️ Contribuição Atômica e Balanço Aditivo:**
                  * Cada subestrutura (fragmentos de anéis, átomos centrais, grupos funcionais) presente ou ausente
                    recebe uma pontuação de impacto individual sobre a predição final.
                  * **Barras Ciano/Verdes:** fragmentos que elevam a afinidade terapêutica $A_{2A}$ ou reduzem o risco $hERG$ (efeito protetor).
                  * **Barras Vermelhas:** fragmentos que atenuam a afinidade $A_{2A}$ ou que atuam como alertas cardiotóxicos $hERG$.
                * **🔬 Auditoria Farmacofórica Visual:**
                  Abaixo de cada gráfico, os cartões interativos exibem a imagem 2D do microambiente atômico responsável por cada decisão.
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
                width="stretch",
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
        # Mapeamento canônico dos atalhos rápidos para SMILES correspondentes
        PRESET_SMILES_MAP = {
            "Cafeína": REFERENCE_CONTROLS["☕ Cafeína (CHEMBL113)"]["smiles"],
            "Istradefilina": REFERENCE_CONTROLS["💊 Istradefilina (CHEMBL431770)"]["smiles"],
            "Terfenadina": REFERENCE_CONTROLS["⚠️ Terfenadina (CHEMBL22)"]["smiles"],
            "Taurina": REFERENCE_CONTROLS["🌿 Taurina (Endógeno)"]["smiles"],
            "L-Teanina": REFERENCE_CONTROLS["🍵 L-Teanina (Nutracêutico)"]["smiles"],
        }

        if "mode2_smiles_input" not in st.session_state:
            st.session_state["mode2_smiles_input"] = ""

        def _on_mode2_preset_change():
            chosen = st.session_state.get("mode2_preset_selector")
            if chosen in PRESET_SMILES_MAP:
                st.session_state["mode2_smiles_input"] = PRESET_SMILES_MAP[chosen]

        def _on_mode2_text_change():
            typed = st.session_state.get("mode2_smiles_input", "").strip()
            current_preset = st.session_state.get("mode2_preset_selector")
            if current_preset in PRESET_SMILES_MAP:
                if typed != PRESET_SMILES_MAP[current_preset]:
                    st.session_state["mode2_preset_selector"] = "(Personalizado)"

        col_input, col_preset = st.columns([3, 1])
        with col_preset:
            preset_choice = st.selectbox(
                "Atalhos rápidos:",
                ["(Personalizado)", "Cafeína", "Istradefilina", "Terfenadina", "Taurina", "L-Teanina"],
                index=0,
                key="mode2_preset_selector",
                on_change=_on_mode2_preset_change,
            )

        with col_input:
            user_smiles = st.text_input(
                "Insira o código SMILES da molécula:",
                key="mode2_smiles_input",
                on_change=_on_mode2_text_change,
                placeholder="Ex: CCN1C(=O)C2=C(N=C(N2C)/C=C/C3=CC(=C(C=C3)OC)OC)N(C1=O)CC",
            )
        active_smiles = user_smiles.strip()

    elif input_mode == "🎨 Desenhar Molécula (Ketcher)":
        try:
            from streamlit_ketcher import st_ketcher

            st.markdown("Desenhe ou edite a estrutura química no editor bidimensional abaixo e clique em **Apply**:")
            ketcher_smiles = st_ketcher("", height=450)
            active_smiles = ketcher_smiles.strip() if ketcher_smiles else ""
            active_name = "Composto Desenhado (Ketcher)"
        except Exception as e:
            st.warning(f"Editor Ketcher indisponível ({e}). Por favor, utilize a entrada por código SMILES.")
            active_smiles = st.text_input("SMILES:", value="", placeholder="Insira o código SMILES da molécula...")


    # -----------------------------------------------------------------------------
    # 10. Padronização, Featurização e Rastreamento de Bits (bitInfo RDKit)
    # -----------------------------------------------------------------------------
    if not active_smiles:
        st.info("💡 Por favor, selecione um controle de referência, insira um código SMILES ou desenhe uma molécula para iniciar a análise.")
        st.stop()

    canonical_smi = canonicalize_smiles(active_smiles)

    if canonical_smi is None:
        st.error(
            f"❌ **Estrutura Química Inválida:** O RDKit não pôde interpretar o código submetido (`{active_smiles}`). "
            "Verifique a valência e sintaxe do código SMILES."
        )
        st.stop()

    provenance = resolve_compound_provenance(canonical_smi)

    # Consulta automatizada de identificação no PubChem (reutilizada para Ketcher e SMILES manual)
    pubchem_info = identify_compound_pubchem(canonical_smi)
    if pubchem_info.get("status") == "SUCCESS" and pubchem_info.get("title"):
        if not provenance.get("is_reference_control"):
            pub_title = pubchem_info["title"]
            cid_num = pubchem_info.get("cid")
            provenance["name"] = f"{pub_title} (PubChem CID {cid_num})" if cid_num else pub_title
            if pubchem_info.get("iupac_name"):
                provenance["desc"] = (
                    f"Identificado automaticamente via PubChem (CID {cid_num}). "
                    f"Nome sistemático IUPAC: {pubchem_info['iupac_name']}."
                )

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

    # [Função canônica transferida para a fachada pública: classify_herg_operational_risk]


    herg_operational = classify_herg_operational_risk(prob_herg)
    herg_color_class = herg_operational["color_class"]
    herg_label = herg_operational["label"]
    herg_desc = herg_operational["desc"]

    render_html(build_control_provenance_html(provenance, pred_pIC50, prob_herg))

    # Cartão de Identificação Automatizada via PubChem (reutilizado para Ketcher e SMILES manual)
    pubchem_card_html = build_pubchem_card_html(pubchem_info)
    if pubchem_card_html:
        render_html(pubchem_card_html)


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
                    <span>1. Potência no Receptor A<sub>2A</sub></span>
                    <span style="font-size:0.78rem; color:#94A3B8;" title="Modelo de regressão supervisionada XGBoost treinado em pIC50">Regressão XGBoost ℹ️</span>
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
                    <span>2. Triagem de Segurança hERG (K<sub>v</sub>11.1)</span>
                    <span style="font-size:0.78rem; color:#94A3B8;" title="Classificador Random Forest treinado em bioensaios de inibição de potássio">Classificador RF ℹ️</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.4rem; margin-bottom:0.75rem;">
                    <span class="{herg_color_class}">{herg_label}</span>
                    <span class="{herg_sample_badge_cls}">{herg_sample_icon} {provenance["herg_sample_status"]}</span>
                </div>
                <div class="kpi-grid">
                    <div class="kpi-box">
                        <div class="kpi-label">Escore Estimado P(hERG)</div>
                        <div class="kpi-value" style="color:{herg_bar_color};">{prob_herg:.1%}</div>
                        <div style="font-size:0.75rem; color:#94A3B8;" title="Estimativa estatística de inibição do canal">Probabilidade de bloqueio ℹ️</div>
                    </div>
                    <div class="kpi-box">
                        <div class="kpi-label">Índice Complementar (1 - P_hERG)</div>
                        <div class="kpi-value kpi-value-accent">{safety_index:.1%}</div>
                        <div style="font-size:0.75rem; color:#94A3B8;" title="Margem de segurança relativa derivada do escore">Margem de segurança ℹ️</div>
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
                <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(130px, 1fr)); gap:0.75rem; margin-top:0.5rem;">
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
    # [Função canônica transferida para a fachada pública: build_multiobjective_figure]


    # [Função canônica transferida para a fachada pública: classify_herg_safety_axis]


    # [Função canônica transferida para a fachada pública: classify_multiobjective_quadrant]


    st.subheader("🗺️ Matriz de Decisão Multiobjetivo (Potência Predita A2A × Margem 1 - P_hERG)")

    x_cur = pred_pIC50
    y_cur = safety_index

    fig_mat, ax_mat = build_multiobjective_figure(x_cur, y_cur, active_name)
    st.pyplot(fig_mat, width="stretch")
    plt.close(fig_mat)

    quad_info = classify_multiobjective_quadrant(x_cur, y_cur)
    quad_badge_cls = quad_info["quad_badge_cls"]
    quad_title = quad_info["quad_title"]
    quad_text = quad_info["quad_text"]

    render_html(
        f"""
        <div class="metric-card" style="margin-top:0.75rem;">
            <div class="card-header-title" style="margin-bottom:0.6rem;">
                <span>🧭 Diagnóstico Operacional de Posicionamento Farmacológico</span>
                <span class="{quad_badge_cls}" style="font-size:0.90rem;">{quad_title}</span>
            </div>
            <p class="pharm-desc-text" style="font-size:0.92rem; margin-bottom:0.85rem; line-height:1.5;">
                {quad_text}
            </p>
            <div class="kpi-grid">
                <div class="kpi-box">
                    <div class="kpi-label">Eixo Horizontal (X) — Potência Predita A2A</div>
                    <div style="color:#FFFFFF; font-size:0.95rem; margin-top:0.2rem;">
                        <b>pIC50 predito = {x_cur:.2f}</b> ({ic50_str}) &nbsp;•&nbsp; 
                        <span style="color:#94A3B8; font-size:0.82rem;">Faixas: Inativo (&lt; 5,0) | Moderado (5,0–6,5) | Alta Potência (≥ 6,5)</span>
                    </div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Eixo Vertical (Y) — Margem de Segurança Derivada</div>
                    <div style="color:#FFFFFF; font-size:0.95rem; margin-top:0.2rem;">
                        <b>1 - P(hERG) = {y_cur:.1%}</b> (Escore hERG = {prob_herg:.1%}) &nbsp;•&nbsp; 
                        <span style="color:#94A3B8; font-size:0.82rem;">Faixas: Alto Risco (P ≥ 70%) | Moderado (30%–70%) | Baixo Risco (P &lt; 30%)</span>
                    </div>
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

            st.info("Auditoria Local de Atributos: Exibindo exclusivamente os 10 principais atributos (Top 10) de maior impacto nesta predição.")

            fig_sa, ax_sa = plt.subplots(figsize=(10.5, 4.6), facecolor="#0E1117")
            ax_sa.set_facecolor("#1E222D")
            bar_colors_a2a = ["#00D2BE" if imp >= 0 else "#FF4B4B" for imp in top_impacts_a2a]

            y_pos = np.arange(len(top_names_a2a))
            ax_sa.barh(y_pos, top_impacts_a2a, color=bar_colors_a2a, edgecolor="#FFFFFF", linewidth=0.6, alpha=0.92)
            ax_sa.set_yticks(y_pos)
            ax_sa.set_yticklabels(top_names_a2a, fontsize=10, fontweight="bold", color="#FFFFFF")
            ax_sa.invert_yaxis()
            ax_sa.axvline(0, color="#CBD5E1", linestyle="-", linewidth=1.1)
            ax_sa.set_xlabel(
                "Impacto SHAP no pIC50 (Ciano = Aumenta Afinidade A2A | Vermelho = Atenua Afinidade)",
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
            st.pyplot(fig_sa, width="stretch")
            plt.close(fig_sa)

            st.markdown("#### 🧩 Grade Visual de Farmacóforos e Descritores Decisivos (Receptor $A_{2A}$)")
            st.caption(
                "Os fragmentos moleculares abaixo são extraídos diretamente da molécula em análise via `bitInfo` do RDKit "
                "com microambiente 2D em Dark Mode (destaque ciano = átomo central do bit; âmbar = átomos aromáticos do ambiente circular ECFP4)."
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

            st.info("Auditoria Local de Atributos: Exibindo exclusivamente os 10 principais atributos (Top 10) de maior impacto nesta predição.")

            fig_sh, ax_sh = plt.subplots(figsize=(10.5, 4.6), facecolor="#0E1117")
            ax_sh.set_facecolor("#1E222D")
            bar_colors_herg = ["#FF4B4B" if imp >= 0 else "#00D2BE" for imp in top_impacts_herg]

            y_pos = np.arange(len(top_names_herg))
            ax_sh.barh(y_pos, top_impacts_herg, color=bar_colors_herg, edgecolor="#FFFFFF", linewidth=0.6, alpha=0.92)
            ax_sh.set_yticks(y_pos)
            ax_sh.set_yticklabels(top_names_herg, fontsize=10, fontweight="bold", color="#FFFFFF")
            ax_sh.invert_yaxis()
            ax_sh.axvline(0, color="#CBD5E1", linestyle="-", linewidth=1.1)
            ax_sh.set_xlabel(
                "Impacto SHAP em P(hERG) (Vermelho = Aumenta Risco Cardiotóxico | Ciano = Efeito Protetor)",
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
            st.pyplot(fig_sh, width="stretch")
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


# -----------------------------------------------------------------------------
# 8. Módulos e Páginas Provisórias da Navegação (Streamlit 1.65.0)
# -----------------------------------------------------------------------------
from src.ui.components import render_provisional_page

def page_visao_geral() -> None:
    render_provisional_page(
        page_name="Visão Geral",
        scheduled_stage="Etapa de Implementação 5",
        description="Apresentação do escopo científico, pipeline em 4 etapas, estado operacional dos modelos e fluxo de trabalho multiobjetivo.",
        icon="🏠",
    )

def page_nova_analise() -> None:
    from src.ui.pages.p2_nova_analise import render_nova_analise

    models_bundle = {
        "model_a2a": model_a2a,
        "model_herg": model_herg,
        "meta_a2a": meta_a2a,
        "meta_herg": meta_herg,
        "explainer_a2a": explainer_a2a,
        "explainer_herg": explainer_herg,
        "load_ad_reference": load_ad_reference_cached,
        "resolve_provenance": resolve_compound_provenance,
        "classify_herg": classify_herg_operational_risk,
        "classify_quadrant": classify_multiobjective_quadrant,
    }
    render_nova_analise(models_bundle=models_bundle)

    with st.expander("🔬 Modo de Transição: Acessar Execução Funcional Legada", expanded=False):
        if st.checkbox("Habilitar visualização legada completa (requer inferência adicional)", value=False, key="chk_enable_legacy_view"):
            render_legacy_analysis_view()
        else:
            st.info("A interface legada está em repouso para economizar processamento e evitar execuções redundantes de SHAP. Marque a opção acima para visualizá-la.")

def page_resultados() -> None:
    from src.ui.pages.p3_resultados import render_resultados
    render_resultados()

def page_interpretacao() -> None:
    from src.ui.pages.p4_interpretacao import render_interpretacao
    render_interpretacao()

def page_comparacao() -> None:
    from src.ui.pages.p5_comparacao import render_comparacao
    render_comparacao()

def page_como_utilizar() -> None:
    from src.ui.pages.p6_como_utilizar import render_como_utilizar
    render_como_utilizar()

def page_fundamentos() -> None:
    render_provisional_page(
        page_name="Fundamentos Científicos",
        scheduled_stage="Etapa de Implementação 5",
        description="Embasamento teórico sobre quimioinformática RDKit, representações Morgan/ECFP4, modelagem supervisionada e formulação SHAP.",
        icon="🔬",
    )

def page_governanca() -> None:
    render_provisional_page(
        page_name="Modelos e Governança",
        scheduled_stage="Etapa de Implementação 5",
        description="Transparência de métricas de treino/holdout, dados de refit 100%, advertência de escores não calibrados e escopo de aplicabilidade.",
        icon="⚖️",
    )

def page_sobre() -> None:
    render_provisional_page(
        page_name="Sobre o Projeto",
        scheduled_stage="Etapa de Implementação 5",
        description="Créditos acadêmicos, termos de licenciamento MIT, ambiente local de execução e disclaimer de uso para pesquisa.",
        icon="ℹ️",
    )


# -----------------------------------------------------------------------------
# 9. Arquitetura Multipágina Centralizada (st.navigation)
# -----------------------------------------------------------------------------
page_obj_visao_geral = st.Page(page_visao_geral, title="Visão Geral", icon="🏠", url_path="", default=True)
page_obj_nova_analise = st.Page(page_nova_analise, title="Nova Análise", icon="🧪", url_path="nova-analise")
page_obj_resultados = st.Page(page_resultados, title="Resultados", icon="📊", url_path="resultados")
page_obj_interpretacao = st.Page(page_interpretacao, title="Interpretação", icon="🔬", url_path="interpretacao")
page_obj_comparacao = st.Page(page_comparacao, title="Comparação", icon="⚖️", url_path="comparacao")
page_obj_como_utilizar = st.Page(page_como_utilizar, title="Como Utilizar", icon="📖", url_path="como-utilizar")
page_obj_fundamentos = st.Page(page_fundamentos, title="Fundamentos Científicos", icon="🔬", url_path="fundamentos")
page_obj_governanca = st.Page(page_governanca, title="Modelos e Governança", icon="⚖️", url_path="governanca")
page_obj_sobre = st.Page(page_sobre, title="Sobre o Projeto", icon="ℹ️", url_path="sobre")

pages_registry = {
    "": page_obj_visao_geral,
    "visao_geral": page_obj_visao_geral,
    "visao-geral": page_obj_visao_geral,
    "nova_analise": page_obj_nova_analise,
    "nova-analise": page_obj_nova_analise,
    "resultados": page_obj_resultados,
    "interpretacao": page_obj_interpretacao,
    "comparacao": page_obj_comparacao,
    "como_utilizar": page_obj_como_utilizar,
    "como-utilizar": page_obj_como_utilizar,
    "fundamentos": page_obj_fundamentos,
    "governanca": page_obj_governanca,
    "sobre": page_obj_sobre,
}
st.session_state["pages_registry"] = pages_registry

pages_navigation = {
    "ANÁLISE": [
        page_obj_visao_geral,
        page_obj_nova_analise,
        page_obj_resultados,
        page_obj_interpretacao,
        page_obj_comparacao,
    ],
    "APRENDER": [
        page_obj_como_utilizar,
        page_obj_fundamentos,
    ],
    "REFERÊNCIA": [
        page_obj_governanca,
        page_obj_sobre,
    ],
}

nav_controller = st.navigation(pages_navigation, position="sidebar")
nav_controller.run()
