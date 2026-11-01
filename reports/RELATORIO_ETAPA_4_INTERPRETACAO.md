# RELATÓRIO TÉCNICO DE IMPLEMENTAÇÃO — ETAPA 4: INTERPRETAÇÃO QUÍMICA E COMPUTACIONAL

**Projeto:** MultiTarget BioPredict — Plataforma de Triagem Virtual Multiobjetivo ($A_{2A} \times \text{hERG}$)  
**Arquitetura:** Streamlit Multipágina Nativo (`st.navigation` + `st.Page`)  
**Data:** 31 de Outubro de 2026  
**Responsável Técnico:** Antigravity (Google DeepMind - Advanced Agentic Coding)  
**Status da Etapa:** Concluída com Sucesso (100% dos testes aprovados, invariância criptográfica preservada)

---

## 1. RESUMO EXECUTIVO

A **Etapa 4 — Implementação da Interpretação Química e Computacional** estabeleceu uma experiência didática, transparente e cientificamente rigorosa para a explicabilidade das predições multiobjetivo no MultiTarget BioPredict.

Diferente de visualizações genéricas de Feature Importance, a solução desenvolvida implementa o **Protocolo Visual Híbrido de 5 Campos Independentes**, acopla a recuperação direta de subambientes circulares via RDKit (`Morgan bitInfo`, `radius=2`, `nBits=1024`), trata de forma explícita as **colisões de hashing**, diferencia rigorosamente **interpretação computacional** de **interpretação química** e apresenta o **Domínio de Aplicabilidade (AD)** e os **6 descritores físico-químicos** sem extrapolações causais ou mecanísticas espúrias.

A página consome estritamente o snapshot pré-calculado em `st.session_state["current_analysis"]`, garantindo custo computacional nulo de inferência adicional ao usuário.

---

## 2. ARQUIVOS CRIADOS E MODIFICADOS

| Arquivo | Ação | Responsabilidade no Sistema |
| :--- | :--- | :--- |
| `src/ui/interpretation_engine.py` | **Criado** | Motor de quimioinformática para recuperação de ambientes Morgan, detecção de colisões, geração de fragmentos 2D em Dark Mode, estruturação dos 5 campos, ranqueamento SHAP e interpretação de AD. |
| `src/ui/pages/p4_interpretacao.py` | **Criado** | Módulo de apresentação da página **Interpretação** com 4 abas especializadas ($A_{2A}$, hERG, Domínio AD e Descritores 2D), filtros por tipo de contribuição e estado vazio resiliente. |
| `tests/test_ui_p4_interpretacao.py` | **Criado** | Suíte de testes automatizados com 10 testes cobrindo os 10 requisitos obrigatórios da Etapa 4. |
| `src/ui/pages/__init__.py` | **Modificado** | Exportação pública de `render_interpretacao` junto a `render_nova_analise` e `render_resultados`. |
| `src/ui/catalogs.py` | **Modificado** | Resolução de fallback seguro para denominação de subambientes Morgan desconhecidos (`resolve_translated_feature_name`). |
| `app.py` | **Modificado** | Registro da rota `st.Page(page_interpretacao, title="Interpretação", icon="🔬", url_path="interpretacao")` no grupo `ANÁLISE` de `st.navigation`. |
| `src/ui/pages/p3_resultados.py` | **Modificado** | Inclusão de botões e links diretos para aprofundamento na página Interpretação a partir do Painel de Resultados. |

---

## 3. ARQUITETURA E COMPONENTES IMPLEMENTADOS

### 3.1 Estrutura da Página `Interpretação` (`src/ui/pages/p4_interpretacao.py`)
A página organiza-se em três níveis hierárquicos:
1. **Banner Superior e Separação Epistemológica:**
   - Exibição de metadados do composto analisado (Nome comercial/PubChem/Controle, SMILES canônico, fórmula, peso molecular e total de bits ativos no vetor de 1.024).
   - Card epistemológico destacado diferenciando **Interpretação Computacional** (árvores de decisão, SHAP aditivo, métricas Tanimoto) de **Interpretação Química** (subestruturas observáveis, características moleculares e hipóteses da literatura).
2. **Abas Técnicas Especializadas:**
   - **Aba 1 (🎯 Receptor $A_{2A}$):** Métricas do modelo XGBoost, valor base $E[f(X)]$, deslocamento SHAP total ($\sum \phi_i$) e cartões de 5 campos filtráveis por Maior Magnitude, Contribuições Positivas e Contribuições Negativas.
   - **Aba 2 (🫀 Canal hERG):** Métricas do modelo Random Forest, probabilidade predita $P(\text{hERG})$, valor base da classe positiva, e cartões de 5 campos com foco em Alertas Cardiotóxicos (+) e Fatores Protetivos (-).
   - **Aba 3 (🌐 Domínio de Aplicabilidade):** Guia conceitual (identidade vs similaridade vs cobertura vs confiança estatística), comparativo visual $A_{2A} \times \text{hERG}$, identificação de proveniência (Níveis 1 a 4), $S_{\max}$ geral, $S_{\max}$ excluindo auto-match, mediana LOO de treino e estrutura 2D do vizinho mais próximo de bancada.
   - **Aba 4 (⚖️ Descritores Físico-Químicos):** Tabela comparativa dos 6 descritores globais (MolWt, MolLogP, NumHDonors, NumHAcceptors, TPSA, NumRotatableBonds) com valores calculados, impacto SHAP em cada alvo e expanders explicativos com ressalva contra generalizações universais.
3. **Rodapé de Navegação e Continuidade:**
   - Links rápidos bidirecionais para `nova-analise`, `resultados` e `fundamentos`.

---

## 4. ESTRUTURAÇÃO OBRIGATÓRIA DAS EXPLICAÇÕES: OS 5 CAMPOS INDEPENDENTES

Em estrito atendimento ao item 4 das diretrizes, cada atributo interpretado é decomposto em **cinco campos mutuamente exclusivos**:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. ATRIBUTO COMPUTACIONAL                                                              │
│    • Identificador Exato: fp_831 (ou MolLogP, fp_926, etc.)                           │
│    • Tipo de Atributo: Fingerprint de Morgan (ECFP4) / Descritor 2D                   │
│    • Denominação Didática: Aceptor de H em Anel Heteroaromático (N sp²)               │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. CONTRIBUIÇÃO SHAP (ϕ)                                                               │
│    • Valor Numérico Formatado: +0.4250 (ou -0.2100)                                    │
│    • Direção Textual Específica:                                                       │
│      - A2A: "Eleva a saída predita de afinidade (aumenta o pIC50 estimado)."            │
│      - hERG: "Reduz a saída associada à classe hERG positiva (diminui o escore)."      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. SUBAMBIENTE ESTRUTURAL                                                              │
│    • Subambiente Isolado 2D: Imagem gerada via DrawMorganBit (160x160 px, Dark Mode)   │
│    • Destaque Molecular 2D: Átomos e ligações realçados na molécula completa (Dark)   │
│    • Diagnóstico de Multiplicidade/Colisão: Notificação explícita caso haja ambiguidade│
│    • Caso Bit Ausente: "Atributo ausente na molécula avaliada (valor = 0)."            │
│    • Caso Descritor: "Propriedade físico-química global contínua (não fragmentária)."  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. DESCRIÇÃO QUÍMICA OBSERVACIONAL                                                     │
│    • Caracterização neutra e puramente estrutural dos átomos e ligações.               │
│    • Conformidade fp_926: Estritamente "Carbono alifático (raio 0)", sem menções a    │
│      "Sistema Aromático Condensado Substituído".                                       │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 5. HIPÓTESE DA LITERATURA CIENTÍFICA (CONDICIONAL)                                     │
│    • Apresentada APENAS quando houver registro documentado em catálogo científico.     │
│    • Caso contrário: "Sem hipótese bibliográfica documentada para este atributo."      │
│    • Ressalva mandatória: Hipótese correlacional; não constitui mecanismo comprovado.  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. ESTRATÉGIA DE RECUPERAÇÃO DOS AMBIENTES MORGAN E TRATAMENTO DE COLISÕES

### 5.1 Protocolo Técnico de Mapeamento
A recuperação dos subambientes moleculares utiliza a infraestrutura de `bitInfo` do RDKit sobre o objeto molecular já padronizado:
```python
bit_info = {}
AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024, bitInfo=bit_info)
```
Para cada bit ativo identificado pelo SHAP:
1. Se `bit_id not in bit_info`: o bit está inativo na molécula (valor 0). O impacto SHAP decorre da **ausência** dessa característica no cálculo das árvores de decisão.
2. Se `bit_id in bit_info`: extrai-se a tupla de ocorrências `((atom_idx, radius), ...)`.
3. Para cada ambiente individual:
   - Se `radius == 0`: o ambiente é o átomo central isolado (`atoms = [atom_idx]`, `bonds = []`).
   - Se `radius > 0`: executa-se `Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)` para recuperar os índices de ligações e compor o conjunto exato de átomos conectados.

### 5.2 Tratamento Rigoroso de Colisões de Hashing
O algoritmo verifica se os múltiplos ambientes mapeados para o mesmo bit compartilham o mesmo raio e símbolo químico:
- **Colisão de Hashing Genuína (`has_collision = True`):** Ambientes com raios distintos (ex.: raio 0 e raio 2) ou símbolos elementares diferentes dobrados no mesmo bit de 1.024 posições.
  - *Exemplo Real:* O bit `33` na Cafeína mapeia simultaneamente 3 átomos de carbono metílico em raio 0 (`C0`, `C9`, `C11`) e 1 carbono do anel purínico em raio 2 (`C5`). O sistema alerta: *"Ambiguidade de Hashing: 4 ambientes distintos mapeados para o mesmo bit fp_33 (colisão de hashing no vetor de 1.024 bits)"*.
- **Multiplicidade Simétrica (`has_collision = False`):** Múltiplas ocorrências com mesma tipologia estrutural.
  - *Exemplo Real:* O bit `121` na Cafeína mapeia 3 ligações $C-N$ simétricas em raio 1. O sistema alerta: *"Multiplicidade: 3 ocorrências simétricas ou estruturalmente equivalentes deste fragmento na molécula"*.
- **Seletor Interativo:** Em ambos os casos de ambiguidade, a interface disponibiliza um `st.selectbox` permitindo alternar a visualização 2D entre os ambientes individuais, destacando cada centro atômico separadamente.

---

## 6. ORIENTAÇÃO DO SINAL SHAP POR MODELO

As convenções de sinal e impacto foram implementadas com estrita fidelidade aos algoritmos subjacentes:

### 6.1 Modelo $A_{2A}$ (XGBoost Regressor)
- **Variável Explicada:** $pIC_{50} = -\log_{10}(IC_{50} \text{ [M]})$.
- **Sinal Positivo ($\phi_i > 0$):** Eleva o $pIC_{50}$ predito $\rightarrow$ Favorece maior afinidade/potência no receptor $A_{2A}$ (Semáforo Ciano/Verde: `#00D2BE`).
- **Sinal Negativo ($\phi_i < 0$):** Reduz o $pIC_{50}$ predito $\rightarrow$ Desfavorece a afinidade (Semáforo Vermelho: `#FF4B4B`).

### 6.2 Modelo hERG (Random Forest Classifier)
- **Variável Explicada:** Probabilidade da Classe 1 (Bloqueador do canal hERG / Risco Cardiotóxico).
- **Sinal Positivo ($\phi_i > 0$):** Eleva a probabilidade de bloqueio $\rightarrow$ Constitui **Alerta de Cardiotoxicidade** (Semáforo Vermelho: `#FF4B4B`).
- **Sinal Negativo ($\phi_i < 0$):** Reduz a probabilidade de bloqueio $\rightarrow$ Constitui **Fator Protetivo** que afasta a molécula da zona de cardiotoxicidade (Semáforo Ciano/Verde: `#00D2BE`).

---

## 7. DOMÍNIO DE APLICABILIDADE (AD) — TRANSPARÊNCIA SEM EXTRAPOLAÇÃO

A seção de Domínio de Aplicabilidade cumpre os requisitos epistemológicos da Fase 1:
1. **Diferenciação Didática:**
   - *Correspondência Estrutural:* Amostra de calibração in-sample com medição experimental (Níveis 1 e 2).
   - *Similaridade Química:* Sobreposição de fingerprints circulares Tanimoto ECFP4.
   - *Cobertura do Domínio:* Densidade do espaço químico amostrado no treino.
   - *Confiança Estatística:* Não vinculação de Tanimoto a intervalos de probabilidade ou certeza estatística.
2. **Auto-Exclusão Preservada:**
   - Exibição de $S_{\max}$ geral e $S_{\max}$ excluindo a auto-correspondência (disponível quando a molécula avaliada é um controle in-sample de treino, como Cafeína e Istradefilina).
3. **Comparação com Mediana LOO:**
   - Referência empírica Leave-One-Out fixa do conjunto de treino: $0,8219$ para $A_{2A}$ e $0,8308$ para hERG.
4. **Inspeção Visual do Vizinho:**
   - Renderização 2D em Dark Mode da estrutura química do vizinho de treino mais próximo, com seu ID ChEMBL e valor experimental.

---

## 8. SUÍTE DE TESTES AUTOMATIZADOS E RESULTADOS

Foi desenvolvida a suíte `tests/test_ui_p4_interpretacao.py`, cobrindo 100% dos requisitos de testes da Etapa 4:

| Teste | Objetivo | Resultado |
| :--- | :--- | :---: |
| `test_morgan_environment_recovery` | Extração de bitInfo preservando radius=2 e nBits=1024 | **PASSED** |
| `test_collision_handling_and_multiple_environments` | Detecção de colisão (bit 33) e multiplicidade (bit 121) | **PASSED** |
| `test_structural_fragment_highlights` | Renderização 2D de PNGs em base64 e resiliência | **PASSED** |
| `test_shap_sign_orientation_for_each_model` | Verificação do sinal e direção textual para A2A e hERG | **PASSED** |
| `test_physicochemical_descriptors_interpretation` | Quadro comparativo dos 6 descritores e status escalar | **PASSED** |
| `test_absence_of_shap_data` | Resiliência a arrays SHAP nulos, vazios ou incompletos | **PASSED** |
| `test_absence_of_recoverable_subenvironment` | Tratamento neutro de bits inativos (ausência na molécula) | **PASSED** |
| `test_absence_of_bibliographic_references` | Ausência de alucinações e conformidade textual do fp_926 | **PASSED** |
| `test_incomplete_ad_data_handling` | Resiliência a diagnósticos AD parciais ou em falha técnica | **PASSED** |
| `test_empty_state_page_rendering` | Renderização sem exceções no estado vazio (sem análise) | **PASSED** |

### Execução Completa da Suíte do Projeto
A suíte completa de testes de regressão, quimioinformática, AD e UI foi executada:
```
============================== 99 passed in 86.56s ==============================
- tests/test_ui_p4_interpretacao.py: 10 passed
- tests/test_ui_p3_resultados.py: 10 passed
- tests/test_ui_p2_nova_analise.py: 47 passed
- tests/test_controls_and_interpretation.py: 5 passed
- tests/test_multiobjective_plot.py: 3 passed
- tests/test_ad_performance.py: 1 passed
- tests/test_applicability_domain.py: 10 passed
- tests/test_featurization_regression.py: 7 passed
- tests/test_pubchem_identification.py: 6 passed
```
**Total:** **99 testes executados, 99 aprovados (100% de sucesso).**

---

## 9. VERIFICAÇÃO CRIPTOGRÁFICA DE INVARIÂNCIA (HASHES SHA-256)

Todos os 8 artefatos protegidos (modelos serializados, metadados JSON e bases de dados processadas) foram auditados via hash SHA-256 e permanecem **100% inalterados e idênticos** ao baseline:

| Arquivo Protegido | SHA-256 Hash Verificado | Status |
| :--- | :--- | :---: |
| `models/a2a_model.joblib` | `1907c7241edf64ffae3aa5c59a497037d427b1ead6b83abdf15abbfe3b2a4787` | **INVARIANTE** |
| `models/a2a_model_metadata.json` | `da3e05936977254e08c34dab6b21009c3da2c632647e062522662bd47a1d5f8f` | **INVARIANTE** |
| `models/herg_model.joblib` | `fcedbecb89843870fe1b7da9c501eb62b8e9e6a0a254a9ed8139c17d6ad154e6` | **INVARIANTE** |
| `models/herg_model_metadata.json` | `781f4b07b68eeeff8f7fe245aa7513073811bb78819fe16996902c4d3146115f` | **INVARIANTE** |
| `data/processed/a2a_curated.csv` | `88354901075bc9058b6ae6b5ad8920a4a710f10b2efd32abd8ab02541b977a3e` | **INVARIANTE** |
| `data/processed/herg_curated.csv` | `df7be71479e64c05f735b17b35a6fd888cc6ca869615e9bde50f489c715b3798` | **INVARIANTE** |
| `data/processed/a2a_featurized.parquet` | `ac006abb90b59ac74c6d3a14cad25ceee62c58cee44fa263a8a69f90928b31b7` | **INVARIANTE** |
| `data/processed/herg_featurized.parquet` | `612e115eba6e47788adb92a2edd067e89d5af9f5095dba0dfcad6fc817d7eb92` | **INVARIANTE** |

---

## 10. PENDÊNCIAS E PRÓXIMOS PASSOS (FASE 5)

A Etapa 4 encontra-se integralmente implementada, testada e pronta para avaliação. Nenhuma pendência técnica impeditiva foi identificada.

**Escopo para a Etapa 5 (Central de Ajuda, Governança e Páginas Institucionais):**
- Construção de `p1_visao_geral.py` (homepage e fluxo conceitual de 4 etapas).
- Construção de `p4_como_utilizar.py` (guia em 2 camadas: Essencial e Técnico).
- Construção de `p5_fundamentos.py` (fundamentos de RDKit, ECFP4, XGBoost, Random Forest, SHAP e AD).
- Construção de `p6_governanca.py` (tabela de limiares unificada, transparência de métricas e disclaimer de refit 100%).
- Construção de `p7_sobre.py` (autoria, licença MIT e ambiente).

---

## 11. CONCLUSÃO

A implementação da Etapa 4 eleva o MultiTarget BioPredict a um novo patamar de explicabilidade científica. Ao invés de meras listas de atributos opacos, o usuário dispõe agora de um motor interpretativo que visualiza subambientes atômicos reais, alerta sobre colisões de hashing, respeita estritamente o sinal farmacológico e delimita com rigor a fronteira entre predição matemática e inferência biológica.

**A execução foi interrompida e aguarda avaliação formal do responsável pelo projeto.**  
*(Nenhum commit ou push foi executado; a Etapa 5 não foi iniciada).*

