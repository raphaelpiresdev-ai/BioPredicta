"""
Catálogos de Apresentação e Mapeamento Químico — MultiTarget BioPredict.

Centraliza dicionários de tradução didática dos bits ECFP4 e descritores 2D.
Preserva identificadores canônicos e distinção entre caracterização observacional
e hipóteses de literatura devidamente contextualizadas.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from rdkit import Chem

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
            if mol is not None and bit_info is not None and bit_id in bit_info:
                envs = bit_info[bit_id]
                if envs:
                    atom_idx, radius = envs[0]
                    if atom_idx < mol.GetNumAtoms():
                        sym = mol.GetAtomWithIdx(atom_idx).GetSymbol()
                        return f"Subambiente ECFP4 #{bit_id} (Centro {sym}, r={radius})"
            return f"Subestrutura ECFP4 (#{bit_id})" 
        except Exception:
            pass
    return feature_name


