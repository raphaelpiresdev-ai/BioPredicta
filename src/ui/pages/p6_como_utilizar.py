"""
Módulo da Página: Como Utilizar a Plataforma — MultiTarget BioPredict.

Tutorial interativo, didático e metodologicamente rigoroso estruturado em 7 módulos sequenciais,
com abordagem híbrida (camada essencial e aprofundamento técnico sob demanda), tabela única
de limiares operacionais e diretrizes científicas consolidadas.
"""

from __future__ import annotations

import streamlit as st

from src.ad_utils import (
    PROVENANCE_LEVEL_1_STRUCTURE,
    PROVENANCE_LEVEL_2_FEATURES,
    PROVENANCE_LEVEL_3_FINGERPRINT,
    PROVENANCE_LEVEL_4_NO_EXACT_MATCH,
)
from src.ui.components import get_page_target, render_html, render_page_header

# Alias retrocompativel
_get_page_target = get_page_target


def render_como_utilizar() -> None:
    """Renderiza a página completa do Tutorial de Utilização da plataforma."""
    # Cabeçalho padronizado da página
    render_page_header(
        title="Como Utilizar a Plataforma",
        subtitle="Guia prático e metodológico em 7 módulos: do desenho molecular à interpretação de decisões multiobjetivo.",
        icon="📖",
    )

    # Organização em abas funcionais (Camada Essencial, Manual Completo e Boas Práticas)
    tab_rapido, tab_manual, tab_diretrizes = st.tabs([
        "🚀 Início Rápido (3 Minutos)",
        "📖 Manual Completo (7 Módulos)",
        "⚠️ Diretrizes e Boas Práticas",
    ])

    # =========================================================================
    # ABA 1: INÍCIO RÁPIDO (CAMADA ESSENCIAL)
    # =========================================================================
    with tab_rapido:
        render_html("""
            <div class="metric-card" style="border-left: 4px solid #00D2BE; margin-bottom: 1.2rem;">
                <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.4rem;">
                    ⚡ Fluxo de Triagem em 4 Passos Rápidos
                </div>
                <p style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.5; margin: 0;">
                    Para pesquisadores que desejam testar a plataforma instantaneamente com moléculas-modelo de referência 
                    ou avaliar rapidamente uma nova série química.
                </p>
            </div>
            """)

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            render_html("""
                <div class="surface-n2" style="border-radius: 8px; padding: 14px; margin-bottom: 12px; height: 100%;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                        <span style="background: #00D2BE; color: #0E1117; font-weight: 800; border-radius: 50%; width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center; font-size: 0.82rem;">1</span>
                        <span style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem;">Insira ou Carregue a Molécula</span>
                    </div>
                    <p style="font-size: 0.84rem; color: #CBD5E1; line-height: 1.45; margin: 0 0 8px 0;">
                        Acesse a página <b>Nova Análise</b>. Utilize o expander superior para carregar com 1 clique um composto modelo 
                        (ex.: <i>Istradefilina</i> ou <i>Cafeína</i>), cole um código SMILES ou utilize o editor Ketcher 2D.
                    </p>
                    <div style="font-size: 0.77rem; color: #94A3B8; background: rgba(0,0,0,0.25); padding: 6px 10px; border-radius: 6px;">
                        💡 <b>Dica no Ketcher:</b> Ao desenhar, sempre clique em <code>Apply</code> no canto inferior direito para transferir a estrutura.
                    </div>
                </div>
                """)

        with col_p2:
            render_html("""
                <div class="surface-n2" style="border-radius: 8px; padding: 14px; margin-bottom: 12px; height: 100%;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                        <span style="background: #00D2BE; color: #0E1117; font-weight: 800; border-radius: 50%; width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center; font-size: 0.82rem;">2</span>
                        <span style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem;">Execute a Predição Multiobjetivo</span>
                    </div>
                    <p style="font-size: 0.84rem; color: #CBD5E1; line-height: 1.45; margin: 0 0 8px 0;">
                        Após a confirmação da validação química pelo RDKit, clique no botão principal 
                        <b>🚀 Analisar Molécula (Predição Multiobjetivo)</b>.
                    </p>
                    <div style="font-size: 0.77rem; color: #94A3B8; background: rgba(0,0,0,0.25); padding: 6px 10px; border-radius: 6px;">
                        ⏱️ O pipeline calcula os 1.030 descritores, executa os modelos XGBoost e Random Forest e calcula o AD em ~1,2 segundo.
                    </div>
                </div>
                """)

        col_p3, col_p4 = st.columns(2)
        with col_p3:
            render_html("""
                <div class="surface-n2" style="border-radius: 8px; padding: 14px; margin-bottom: 12px; height: 100%;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                        <span style="background: #00D2BE; color: #0E1117; font-weight: 800; border-radius: 50%; width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center; font-size: 0.82rem;">3</span>
                        <span style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem;">Avalie o Veredito e Indicadores</span>
                    </div>
                    <p style="font-size: 0.84rem; color: #CBD5E1; line-height: 1.45; margin: 0 0 8px 0;">
                        No <b>Painel de Resultados</b>, consulte a síntese executiva:
                        <br/>• <b>A2A:</b> pIC₅₀ ≥ 6,5 indica alta potência predita nanomolar.
                        <br/>• <b>hERG:</b> P(hERG=1) &lt; 30% indica baixo escore de classificação de bloqueio.
                        <br/>• <b>Matriz 2D:</b> O <i>Quadrante 1 (topo direito)</i> representa a região de compromisso favorável.
                    </p>
                    <div style="font-size: 0.77rem; color: #38BDF8; background: rgba(56,189,248,0.1); padding: 6px 10px; border-radius: 6px;">
                        🛡️ Verifique o badge do <b>Domínio de Aplicabilidade (AD)</b> para contextualizar a proximidade frente ao treino.
                    </div>
                </div>
                """)

        with col_p4:
            render_html("""
                <div class="surface-n2" style="border-radius: 8px; padding: 14px; margin-bottom: 12px; height: 100%;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                        <span style="background: #00D2BE; color: #0E1117; font-weight: 800; border-radius: 50%; width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center; font-size: 0.82rem;">4</span>
                        <span style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem;">Aprofunde com SHAP e Comparação</span>
                    </div>
                    <p style="font-size: 0.84rem; color: #CBD5E1; line-height: 1.45; margin: 0 0 8px 0;">
                        • Acesse <b>Interpretação</b> para inspecionar os fragmentos Morgan atômicos em SVG e os papéis farmacofóricos.
                        <br/>• Analise outras moléculas e acesse <b>Comparação</b> para comparar de 2 a 5 compostos lado a lado com matriz de Tanimoto e deltas físico-químicos.
                    </p>
                    <div style="font-size: 0.77rem; color: #F59E0B; background: rgba(245,158,11,0.1); padding: 6px 10px; border-radius: 6px;">
                        🔄 O histórico preserva até 10 moléculas automaticamente durante a sessão.
                    </div>
                </div>
                """)

        render_html("<div style='margin-top:1.2rem;'></div>")
        col_nav_q1, col_nav_q2 = st.columns([1.5, 2.5])
        with col_nav_q1:
            st.page_link(get_page_target("nova-analise"), label="🧪 Ir para Nova Análise Molecular", icon="🚀")
        with col_nav_q2:
            st.caption("Pronto para experimentar? Clique ao lado para abrir o editor e selecionar um controle.")

    # =========================================================================
    # ABA 2: MANUAL COMPLETO (7 MÓDULOS COM ABORDAGEM HÍBRIDA)
    # =========================================================================
    with tab_manual:
        render_html("""
            <p style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.5; margin-bottom: 1rem;">
                Consulte abaixo as instruções sequenciais detalhadas para cada etapa do ciclo de triagem molecular.
                Cada módulo traz uma <b>apresentação essencial imediata</b> e um <b>aprofundamento técnico sob demanda</b>.
            </p>
            """)

        # ---------------------------------------------------------------------
        # MÓDULO 1: Conhecendo a Plataforma
        # ---------------------------------------------------------------------
        with st.expander("📘 Módulo 1 — Conhecendo a Plataforma e o Problema Científico", expanded=True):
            st.markdown(
                """
                #### Apresentação Essencial
                O **MultiTarget BioPredict** é uma plataforma computacional concebida para realizar **triagem virtual multiobjetivo (*in silico*)** 
                de pequenas moléculas orgânicas a partir de representações químicas bidimensionais.
                
                No processo de descoberta de novos candidatos a bioativos e fármacos, a busca por alta potência em um alvo terapêutico 
                deve ser equilibrada com a segurança toxicológica precoce. O sistema avalia simultaneamente:
                
                1. 🎯 **Potência predita frente ao receptor $A_{2A}$ ($ADORA2A$):** Estimativa contínua de potência inibitória expressa em $pIC_{50}$ em um receptor purinérgico acoplado à proteína G relevante para modulação neurológica (ex.: Doença de Parkinson) e inflamatória.
                2. ❤️ **Escore de classificação hERG ($K_v11.1$):** Estimativa da propensão de bloqueio do canal de potássio cardíaco, cuja oclusão por pequenas moléculas está associada ao prolongamento do intervalo QT no eletrocardiograma e ao risco de arritmias ventriculares graves (*Torsades de Pointes*).
                """
            )

            render_html("""
                <div style="background: rgba(245,158,11,0.08); border-left: 4px solid #F59E0B; padding: 10px 14px; border-radius: 6px; margin: 10px 0;">
                    <span style="font-size: 0.84rem; color: #CBD5E1; line-height: 1.45; display: block;">
                        ⚠️ <b>Ressalva Metodológica de Escopo:</b> A plataforma destina-se estritamente à <b>triagem virtual preliminar in silico</b> 
                        para priorização racional de candidatos. As saídas computacionais não constituem comprovação experimental de atividade, segurança 
                        ou eficácia, e <b>não substituem ensaios experimentais in vitro (como patch clamp), toxicologia regulatória (GLP) nem avaliação clínica</b>.
                    </span>
                </div>
                """)

            with st.expander("🔬 Aprofundamento Técnico — Escopo Farmacológico e Modelos", expanded=False):
                st.markdown(
                    r"""
                    - **Alvo Biológico 1 (Potência $A_{2A}$):** Receptor $A_{2A}$ humano (ChEMBL `CHEMBL251`). O modelo prediz a potência inibitória $pIC_{50} = 9{,}0 - \log_{10}(IC_{50\text{ [nM]}})$, inversamente proporcional à concentração inibitória semi-máxima ($IC_{50}$), sem inferir a seletividade frente aos subtipos $A_1$, $A_{2B}$ ou $A_3$.
                    - **Alvo Biológico 2 (Cardiotoxicidade):** Canal de potássio $K_v11.1$ (TDC / Harvard Dataverse `CHEMBL240`). Classificador supervisionado treinado com limiar experimental canônico em $10\ \mu\text{M}$ ($Y \in \{0, 1\}$).
                    - **Abordagem Ortogonal:** Ambos os alvos são preditos de maneira autônoma por modelos independentes, permitindo visualizar o compromisso entre atividade biológica e risco cardíaco sem filtros combinados opacos.
                    """
                )

        # ---------------------------------------------------------------------
        # MÓDULO 2: Inserindo uma Molécula
        # ---------------------------------------------------------------------
        with st.expander("✏️ Módulo 2 — Inserindo uma Molécula (SMILES, Ketcher e Controles)", expanded=False):
            st.markdown(
                """
                #### Apresentação Essencial
                Na página **Nova Análise**, você pode inserir a estrutura da molécula através de três modalidades integradas:
                
                - **Compostos de Referência (1 Clique):** Expander superior com botões pré-configurados para testar o sistema instantaneamente com fármacos e moléculas documentadas:
                  - *Istradefilina:* Antagonista potente $A_{2A}$ e aprovado clinicamente, com escore hERG moderado.
                  - *Terfenadina:* Anti-histamínico de referência cardiotóxica (alto escore hERG, inativo em $A_{2A}$).
                  - *Cafeína:* Metilxantina clássica, potência moderada *in-sample* e baixo escore hERG.
                  - *Taurina / L-Teanina:* Pequenas moléculas de baixa potência basal frente a $A_{2A}$ e baixo escore hERG.
                - **Código SMILES:** Campo de texto para digitar ou colar notações lineares (simplificadas ou isoméricas).
                - **Editor Molecular Ketcher 2D:** Canvas gráfico interativo para desenhar estruturas átomo a átomo.
                """
            )

            render_html("""
                <div style="background: rgba(0,210,190,0.08); border-left: 4px solid #00D2BE; padding: 12px 14px; border-radius: 6px; margin: 10px 0;">
                    <div style="font-weight: 700; color: #FFFFFF; font-size: 0.88rem; margin-bottom: 4px;">
                        📌 Instrução Obrigatória: Como aplicar a estrutura desenhada no Ketcher
                    </div>
                    <span style="font-size: 0.84rem; color: #CBD5E1; line-height: 1.45; display: block;">
                        Ao finalizar o desenho no Ketcher, <b>você deve clicar no botão verde "Apply" localizado no canto inferior direito do editor</b>. 
                        Esse clique transfere o bloco químico para o sistema, que aciona o RDKit para padronizar e gerar o SMILES canônico correspondente, atualizando o campo de texto e gerando a prévia da estrutura validada.
                    </span>
                </div>
                """)

            with st.expander("🔬 Aprofundamento Técnico — Validação Sintática e Busca PubChem", expanded=False):
                st.markdown(
                    """
                    - **Validação Sintática em Tempo Real:** O RDKit verifica valências químicas (ex.: octeto de carbonos e nitrogênios), fechamento correto de anéis aromáticos e emparelhamento de parênteses de ramificação. Erros de sintaxe geram diagnósticos explicativos imediatos na tela.
                    - **Resolução de Identidade PubChem:** Simultaneamente à validação estrutural, a plataforma consulta de forma não-bloqueante a API do PubChem por InChIKey, recuperando o identificador CID, a nomenclatura IUPAC oficial e sinônimos comerciais.
                    """
                )

        # ---------------------------------------------------------------------
        # MÓDULO 3: Executando a Análise
        # ---------------------------------------------------------------------
        with st.expander("⚙️ Módulo 3 — Executando a Análise e Pipeline Computacional", expanded=False):
            st.markdown(
                """
                #### Apresentação Essencial
                Ao clicar no botão principal **🚀 Analisar Molécula (Predição Multiobjetivo)**, o sistema dispara um pipeline automatizado executado 100% no ambiente local em aproximadamente 1,2 segundo:
                
                1. **Padronização Molecular:** Higieniza a estrutura química via RDKit, removendo contraíons/sais inorgânicos e neutralizando cargas simples para gerar a forma canônica isomérica.
                2. **Geração de Características (1.030 atributos):** Calcula simultaneamente 6 descritores físico-químicos 2D fundamentais e 1.024 bits de impressões digitais circulares Morgan ECFP4.
                3. **Inferência Preditiva Dual:** O modelo XGBoost estima a potência em $A_{2A}$ e o modelo Random Forest estima o escore de bloqueio hERG.
                4. **Diagnóstico do Domínio de Aplicabilidade (AD):** Avalia de forma independente a distância química da consulta em relação aos dados de calibração de cada modelo.
                """
            )

            with st.expander("🔬 Aprofundamento Técnico — Detalhes da Featurização e Algoritmos", expanded=False):
                st.markdown(
                    """
                    - **Mecanismos RDKit:** Executa `Cleanup()`, seleção do maior fragmento orgânico com `LargestFragmentChooser(preferOrganic=True)` e desionização com `Uncharger()`.
                    - **Espaço Vetorial 1.030-D:**
                      - *6 Descritores 2D:* Massa Molecular (`MolWt`), Lipofilicidade (`MolLogP`), Área Polar Topológica (`TPSA`), Doadores de H (`NumHDonors`), Aceptores de H (`NumHAcceptors`) e Ligações Rotacionáveis (`NumRotatableBonds`).
                      - *1.024 Bits Morgan ECFP4:* Fingerprints circulares de raio 2 (diâmetro 4 ligações) forçados nativamente na tipagem `numpy.uint8` para economia de memória RAM.
                    - **Modelos de Produção:**
                      - *$A_{2A}$ Regressor:* XGBoost (`tree_method="hist"`, `n_estimators=250`, `learning_rate=0.08`, `max_depth=6`).
                      - *hERG Classifier:* Random Forest (`n_estimators=150`, `min_samples_split=4`).
                    """
                )

        # ---------------------------------------------------------------------
        # MÓDULO 4: Interpretando os Resultados
        # ---------------------------------------------------------------------
        with st.expander("📊 Módulo 4 — Interpretando os Resultados, Limiares e Domínio (AD)", expanded=False):
            st.markdown(
                """
                #### Apresentação Essencial
                O Painel de Resultados apresenta as estimativas dos dois alvos biológicos, classificadas por faixas operacionais rigorosas, acompanhadas pelo diagnóstico de cobertura estrutural.
                """
            )

            # Tabela Única Visual Consolidada de Limiares (Correção 2)
            st.markdown(
                r"""
                #### Tabela Unificada de Classificação e Limiares Operacionais
                | Alvo | Classificação | Limite Operacional | Referência Numérica / Conversão |
                | :--- | :--- | :--- | :--- |
                | **A2A** | Alta potência predita | $pIC_{50} \ge 6{,}5$ (pIC₅₀ ≥ 6,5) | Faixa nanomolar ($IC_{50} \le 316\text{ nM}$) |
                | **A2A** | Potência predita moderada | $5{,}0 \le pIC_{50} < 6{,}5$ (5,0 ≤ pIC₅₀ < 6,5) | Faixa micromolar ($316\text{ nM} < IC_{50} \le 10\ \mu\text{M}$) |
                | **A2A** | Baixa potência/inativo | $pIC_{50} < 5{,}0$ (pIC₅₀ < 5,0) | Faixa micromolar ou superior ($IC_{50} > 10\ \mu\text{M}$) |
                | **hERG** | Baixo escore de classificação | $P < 0{,}30$ (P(hERG=1) < 30%) | Saída positiva $< 30\%$ ($1 - P > 0{,}70$) |
                | **hERG** | Escore moderado | $0{,}30 \le P \le 0{,}70$ (30% até 70%, inclusive) | Saída positiva entre $30\%$ e $70\%$, inclusive ($0{,}30 \le 1 - P \le 0{,}70$) |
                | **hERG** | Alto escore de classificação | $P > 0{,}70$ (P(hERG=1) > 70%) | Saída positiva $> 70\%$ ($1 - P < 0{,}30$) |
                """
            )

            render_html("""
                <div style="background: rgba(255,255,255,0.03); border-left: 4px solid #FF4B4B; padding: 10px 14px; border-radius: 6px; margin: 10px 0;">
                    <span style="font-size: 0.83rem; color: #CBD5E1; line-height: 1.45; display: block;">
                        ⚠️ <b>Aviso de Interpretação da Saída hERG:</b> O escore P(hERG=1) representa estritamente a 
                        <b>fração de votos das árvores do classificador Random Forest para a classe positiva</b> (bloqueador in vitro no limiar de 10 µM). 
                        <b>Ele é um escore relativo de classificação in silico e NÃO uma probabilidade clínica calibrada de cardiotoxicidade, síncope ou arritmia em humanos.</b>
                    </span>
                </div>
                """)

            st.markdown(
                f"""
                #### Domínio de Aplicabilidade (AD): Os 4 Níveis de Proveniência
                O sistema avalia a confiabilidade contextual da inferência de forma totalmente independente para cada modelo:
                
                - **Nível 1 — {PROVENANCE_LEVEL_1_STRUCTURE}:** O SMILES canônico da consulta é idêntico a um composto da base de treino.
                - **Nível 2 — {PROVENANCE_LEVEL_2_FEATURES}:** O vetor numérico de 1.030 variáveis é idêntico (ΔX = 0), indicando paridade química informacional (comum entre estereoisômeros em representação 2D).
                - **Nível 3 — {PROVENANCE_LEVEL_3_FINGERPRINT}:** O fingerprint de 1.024 bits é idêntico (Tanimoto = 1,0000), mas há variação nos descritores físico-químicos (colisão de bits Morgan).
                - **Nível 4 — {PROVENANCE_LEVEL_4_NO_EXACT_MATCH}:** Indica que não foi encontrada uma correspondência exata nos critérios de proveniência utilizados pelo sistema. Esse resultado deve ser considerado na interpretação da análise, sem ser confundido com erro de processamento.
                """
            )

            with st.expander("🔬 Aprofundamento Técnico — Métrica S_max e Auto-Exclusão", expanded=False):
                st.markdown(
                    r"""
                    - **Similaridade Máxima ($S_{\max}$):** Mensura o coeficiente de Tanimoto da molécula de consulta em relação ao vizinho mais próximo da base de calibração. É calculado de forma vetorizada via `DataStructs.BulkTanimotoSimilarity`.
                    - **Regra de Auto-Exclusão:** Quando uma molécula corresponde aos Níveis 1 ou 2, o sistema calcula $S_{\max\text{ excluding match}}$ excluindo a correspondência exata para avaliar a densidade da vizinhança química. No Nível 3, o composto com fingerprint idêntico é preservado como vizinho mais próximo devido à divergência comprovada em suas propriedades físico-químicas.
                    """
                )

        # ---------------------------------------------------------------------
        # MÓDULO 5: Entendendo a Visualização Multiobjetivo
        # ---------------------------------------------------------------------
        with st.expander("🗺️ Módulo 5 — Entendendo a Visualização Multiobjetivo (A2A × hERG)", expanded=False):
            st.markdown(
                """
                #### Apresentação Essencial
                A **Matriz Cartesiana Multiobjetivo** posiciona a molécula em um plano bidimensional de decisão para orientar a análise de compromisso entre potência e perfil de segurança toxicológica:
                
                - **Eixo X (Horizontal):** Potência predita frente ao receptor $A_{2A}$ ($pIC_{50}$). Valores mais à direita indicam maior potência inibitória frente ao receptor.
                - **Eixo Y (Vertical):** Transformação linear derivada $1 - P(\text{hERG}=1)$. Valores mais altos refletem menor propensão de bloqueio cardíaco estimada pelo modelo.
                """
            )

            render_html("""
                <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1); border-radius: 6px; padding: 10px 12px; margin: 10px 0;">
                    <span style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.45; display: block;">
                        ⚠️ <b>Aviso Metodológico do Eixo Y:</b> A transformação linear $1 - P$ é adotada exclusivamente 
                        para orientar a geometria gráfica cartesiana, fazendo com que o topo represente menor bloqueio do canal e permitindo 
                        que a região de maior interesse convirja para o quadrante superior direito. <b>Ela não constitui uma medida clínica direta de segurança nem margem de segurança farmacológica in vivo.</b>
                    </span>
                </div>
                """)

            st.markdown(
                r"""
                #### Caracterização dos Quatro Quadrantes Cartesianos
                | Quadrante | Critérios Operacionais | Interpretação em Triagem Virtual |
                | :--- | :--- | :--- |
                | **🏆 Quadrante 1 (Zona Ideal)** | $pIC_{50} \ge 6{,}5$ e $P < 0{,}30$ ($Y > 0{,}70$) | **Perfil Computacional Favorável:** Alta potência predita nanomolar no alvo terapêutico aliada a baixo escore de classificação hERG. Moléculas de alta prioridade para progressão. |
                | **🌿 Quadrante 2 (Baixo Interesse)** | $pIC_{50} < 5{,}0$ e $P < 0{,}30$ ($Y > 0{,}70$) | **Baixa Potência Predita / Baixo Escore hERG:** Baixa propensão de bloqueio cardíaco predita, porém sem potência relevante frente ao receptor A2A. |
                | **❌ Quadrante 3 (Descarte)** | $pIC_{50} < 5{,}0$ e $P > 0{,}70$ ($Y < 0{,}30$) | **Descarte Prioritário:** Inativo no alvo e simultaneamente com alto escore de classificação de bloqueio do canal de potássio. |
                | **⚠️ Quadrante 4 (Alerta Crítico)** | $pIC_{50} \ge 6{,}5$ e $P > 0{,}70$ ($Y < 0{,}30$) | **Potente com Alerta:** Alta potência predita no receptor, mas com alto escore de bloqueio hERG. Exige desderivação química racional para suprimir alertas toxicológicos. |
                | **🟡 Zona de Otimização** | Faixas intermediárias ($5{,}0 \le pIC_{50} < 6{,}5$ e/ou $0{,}30 \le P \le 0{,}70$) | **Séries Promissoras:** Potência moderada ou escore intermediário, indicando compostos com potencial para otimização de fragmentos (*hit-to-lead*). |
                """
            )

            with st.expander("🔬 Aprofundamento Técnico — Geometria e Limites dos Eixos", expanded=False):
                st.markdown(
                    r"""
                    - **Limites Físicos do Gráfico:** O eixo X opera fixado no intervalo $[3{,}5;\ 11{,}5]$ para garantir estabilidade visual comparativa, enquanto o eixo Y opera em $[-0{,}05;\ 1{,}05]$.
                    - **Linhas Delimitadoras Canônicas:** Linhas pontilhadas verticais delimitam os limiares de $A_{2A}$ em $x = 5{,}0$ e $x = 6{,}5$. Linhas pontilhadas horizontais delimitam os limiares de hERG em $y = 0{,}70$ ($P = 0{,}30$) e $y = 0{,}30$ ($P = 0{,}70$).
                    """
                )

        # ---------------------------------------------------------------------
        # MÓDULO 6: Interpretabilidade com SHAP
        # ---------------------------------------------------------------------
        with st.expander("🧩 Módulo 6 — Interpretabilidade Química com SHAP (XAI)", expanded=False):
            st.markdown(
                r"""
                #### Apresentação Essencial
                A plataforma emprega o método **SHAP** (*SHapley Additive exPlanations*, via `TreeExplainer`) para auditar as decisões internas dos modelos de árvores. O algoritmo quantifica a contribuição de cada característica na predição da molécula:
                
                - **Top 10 de Atributos Mais Influentes:** Gráficos de barras horizontais ordenam as dez variáveis químicas mais determinantes no resultado.
                - **Contribuições Positivas ($+\phi_i$):** Atributos ou subestruturas que elevam o valor predito em relação à média do conjunto de calibração (aumentam a potência predita em $A_{2A}$ ou elevam o escore de bloqueio em hERG).
                - **Contribuições Negativas ($-\phi_i$):** Atributos que reduzem a estimativa do modelo em relação à média (diminuem a potência predita em $A_{2A}$ ou amortecem o escore de bloqueio em hERG).
                - **Papéis Farmacofóricos e Visualização Atômica SVG:** Tabela interativa categoriza os papéis farmacofóricos e destaca os fragmentos e subambientes Morgan diretamente na estrutura 2D.
                """
            )

            render_html("""
                <div style="background: rgba(239,68,68,0.08); border-left: 4px solid #EF4444; padding: 10px 14px; border-radius: 6px; margin: 10px 0;">
                    <span style="font-size: 0.83rem; color: #CBD5E1; line-height: 1.45; display: block;">
                        ⚠️ <b>Ressalva Científica sobre SHAP:</b> As contribuições SHAP refletem correlações estatísticas aditivas aprendidas 
                        pelo algoritmo de Machine Learning a partir do espaço de descritores. <b>Elas explicam a lógica matemática da inferência do modelo, 
                        mas não demonstram causalidade farmacológica nem comprovam interação biofísica direta com resíduos do receptor ou canal.</b>
                    </span>
                </div>
                """)

            with st.expander("🔬 Aprofundamento Técnico — Formulação Matemática e Inspeção SVG", expanded=False):
                st.markdown(
                    r"""
                    - **Decomposição Aditiva Local:** A predição final $f(x)$ satisfaz estritamente a propriedade de eficiência:
                      $$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i$$
                      Onde $\phi_0$ é o valor base esperado (*expected value*) e $\phi_i$ o impacto local do atributo $i$.
                    - **Inspeção de Subambientes Morgan com SVG:** O sistema destaca no desenho 2D da molécula o átomo central e a vizinhança topológica (raio 0 a 2). Para resolver colisões de bits Morgan (quando ambientes distintos mapeiam para o mesmo índice hash), o RDKit gera assinaturas canônicas de submoléculas garantindo a identificação estrutural exata.
                    """
                )

        # ---------------------------------------------------------------------
        # MÓDULO 7: Histórico e Comparação
        # ---------------------------------------------------------------------
        with st.expander("⚖️ Módulo 7 — Histórico de Sessão e Comparação Multimolecular", expanded=False):
            st.markdown(
                r"""
                #### Apresentação Essencial
                O MultiTarget BioPredict integra um gerenciador de histórico não-destrutivo que permite reter análises e comparar séries químicas:
                
                - **Capacidade e Política FIFO:** O histórico armazena até **10 análises concluídas** por sessão. Ao atingir 10 registros, a adição de uma nova molécula descarta automaticamente a mais antiga (*First-In, First-Out*).
                - **Deduplicação Inteligente:** Se você reenviar uma estrutura com o mesmo SMILES canônico, o registro existente é atualizado e reposicionado no final da lista como o mais recente.
                - **Recuperação de Análises:** O botão *"Restaurar como Ativa"* no histórico recarrega o composto completo no Painel de Resultados.
                - **Comparação Multimolecular (`p5_comparacao.py`):** Permite selecionar de **2 a 5 moléculas** do histórico, definir uma molécula de referência e avaliar simultaneamente a dispersão multiobjetivo, a matriz de similaridade de Tanimoto e a tabela de deltas físico-químicos ($\Delta\text{MolWt}$, $\Delta\text{LogP}$, $\Delta\text{TPSA}$, etc.).
                """
            )

            render_html("""
                <div style="background: rgba(0,210,190,0.08); border-left: 4px solid #00D2BE; padding: 10px 14px; border-radius: 6px; margin: 10px 0;">
                    <span style="font-size: 0.83rem; color: #CBD5E1; line-height: 1.45; display: block;">
                        💡 <b>Ressalva Quimioinformática Crítica: Similaridade Tanimoto vs. Scaffold</b><br/>
                        A matriz de similaridade utiliza fingerprints circulares Morgan ECFP4 (raio 2, 1024 bits). 
                        <b>Valores elevados de Tanimoto (&gt; 0,70) refletem alta concordância de subambientes circulares locais e compartilhamento de fragmentos, mas NÃO comprovam identidade ou conservação de scaffold químico central (esqueleto de Bemis-Murcko).</b> Duas moléculas com scaffolds centrais distintos podem apresentar alta similaridade de fragmentos periféricos, enquanto variações de substituintes em um mesmo scaffold podem reduzir o coeficiente de Tanimoto.
                    </span>
                </div>
                """)

            with st.expander("🔬 Aprofundamento Técnico — Cálculo de Deltas e Matriz Par a Par", expanded=False):
                st.markdown(
                    r"""
                    - **Matriz de Tanimoto:** Matriz simétrica de dimensão $N \times N$ ($2 \le N \le 5$) com diagonal unitária ($M_{ii} = 1{,}0000$), calculada via `DataStructs.BulkTanimotoSimilarity`.
                    - **Deltas Físico-Químicos:** Para cada descritor contínuo, calcula $\Delta = \text{Valor}_{\text{molécula}} - \text{Valor}_{\text{referência}}$. A molécula de referência apresenta identicamente $\Delta = 0{,}0$ em todas as propriedades.
                    """
                )

    # =========================================================================
    # ABA 3: DIRETRIZES E BOAS PRÁTICAS CONSOLIDADAS
    # =========================================================================
    with tab_diretrizes:
        render_html("""
            <div class="metric-card" style="border-left: 4px solid #38BDF8; margin-bottom: 1.2rem;">
                <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.4rem;">
                    🛡️ Princípios de Interpretação Científica e Boas Práticas
                </div>
                <p style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.5; margin: 0;">
                    Consolidação dos cuidados metodológicos essenciais para orientar campanhas de triagem virtual racional.
                </p>
            </div>
            """)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            render_html("""
                <div class="surface-n2" style="border-radius: 8px; padding: 14px; margin-bottom: 12px; height: 100%;">
                    <div style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem; margin-bottom: 8px;">
                        1. Invariância Quiral na Representação 2D
                    </div>
                    <p style="font-size: 0.84rem; color: #CBD5E1; line-height: 1.45; margin: 0;">
                        Os descritores 2D e o algoritmo Morgan ECFP4 empregados no pipeline são <b>aquírais</b>. 
                        Enantiômeros puros (que diferem exclusivamente na quiralidade tridimensional de centros quirais) 
                        geram o mesmo vetor de 1.030 atributos e receberão predições idênticas dos modelos. O motor de AD sinalizará isso como <b>Nível 2 (Correspondência de Features)</b>.
                    </p>
                </div>
                """)

        with col_d2:
            render_html("""
                <div class="surface-n2" style="border-radius: 8px; padding: 14px; margin-bottom: 12px; height: 100%;">
                    <div style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem; margin-bottom: 8px;">
                        2. Interpretação Metodológica da Similaridade Máxima (S_max)
                    </div>
                    <p style="font-size: 0.84rem; color: #CBD5E1; line-height: 1.45; margin: 0;">
                        O coeficiente <code>S_max</code> (calculado via similaridade de Tanimoto ECFP4 frente ao vizinho mais próximo da base de calibração) é uma métrica diagnóstica contínua entre 0 e 1, e o código de Domínio de Aplicabilidade não implementa um limiar rígido de 0,30. Valores de S_max inferiores a 0,30 foram observados em determinados compostos de baixa similaridade com as estruturas de referência, incluindo exemplos avaliados durante a validação do projeto. Essa observação exploratória sugere a necessidade de cautela adicional na interpretação, mas não constitui um limiar formal de validade ou rejeição da predição.
                    </p>
                </div>
                """)

        col_d3, col_d4 = st.columns(2)
        with col_d3:
            render_html("""
                <div class="surface-n2" style="border-radius: 8px; padding: 14px; margin-bottom: 12px; height: 100%;">
                    <div style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem; margin-bottom: 8px;">
                        3. Caráter Não-Bloqueante do Domínio de Aplicabilidade
                    </div>
                    <p style="font-size: 0.84rem; color: #CBD5E1; line-height: 1.45; margin: 0;">
                        O diagnóstico de AD é estritamente informativo e contextual. A plataforma não rejeita nem oculta 
                        predições de moléculas no Nível 4. O objetivo é fornecer transparência para que o pesquisador julgue a relevância da inferência no contexto da sua série química.
                    </p>
                </div>
                """)

        with col_d4:
            render_html("""
                <div class="surface-n2" style="border-radius: 8px; padding: 14px; margin-bottom: 12px; height: 100%;">
                    <div style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem; margin-bottom: 8px;">
                        4. Validação Mandatória de Bancada
                    </div>
                    <p style="font-size: 0.84rem; color: #CBD5E1; line-height: 1.45; margin: 0;">
                        Nenhuma decisão de desenvolvimento pré-clínico avançado, síntese em larga escala ou estudo regulatório 
                        deve se basear unicamente em escores computacionais in silico. A validação experimental por radioligand binding e patch clamp automatizado permanece mandatória.
                    </p>
                </div>
                """)

    # =========================================================================
    # RODAPÉ DE NAVEGAÇÃO E CONTINUIDADE
    # =========================================================================
    render_html("<div style='margin-top:2rem;'></div>")
    st.markdown("---")
    col_nav1, col_nav2, col_nav3 = st.columns([1.5, 1.5, 2.0])

    with col_nav1:
        st.page_link(
            get_page_target("nova-analise"),
            label="🧪 Iniciar Nova Análise",
            icon="🚀",
        )

    with col_nav2:
        st.page_link(
            get_page_target("comparacao"),
            label="⚖️ Comparar Moléculas",
            icon="🔄",
        )

    with col_nav3:
        st.page_link(
            get_page_target("fundamentos"),
            label="🧬 Fundamentos Científicos",
            icon="📚",
        )
