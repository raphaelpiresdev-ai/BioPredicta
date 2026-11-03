# RELATÓRIO TÉCNICO — FASE DE ESTABILIZAÇÃO FUNCIONAL
## Projeto MultiTarget BioPredict

**Data:** 01 de Novembro de 2026  
**Branch Git:** `feat/ajustes-funcionalidades`  
**Escopo:** Correção funcional e estabilização da navegação multipágina, renderização HTML, fluxo completo de Nova Análise (Acesso Rápido, SMILES, Ketcher e Histórico) e Comparação Multimolecular  
**Integridade dos Artefatos Protegidos:** 8/8 hashes SHA-256 verificados e 100% preservados  

---

## 1. CAUSA RAIZ DE CADA PROBLEMA

### 1.1. Erro de Navegação (`StreamlitPageNotFoundError`)
- **Mecanismo interno:** No Streamlit 1.65.0, a arquitetura multipágina definida em [`app.py`](file:///c:/Users/Raphael/Documents/BioPredicta/app.py#L2295-L2341) registra funções Python chamáveis (*callables*) em `st.Page(page_func, url_path=...)`. Internamente, páginas baseadas em *callables* possuem `script_path = ""` (não há script físico `.py` dedicado por rota).
- **Falha:** Chamadas a `st.page_link("nova-analise", ...)` ou `st.page_link("resultados", ...)` passavam strings literais. O método `ButtonMixin._page_link` tentava resolver a string comparando caminhos de arquivo em disco com `script_path` (`""`), falhando sistematicamente e lançando `StreamlitPageNotFoundError` em `p2_nova_analise.py`, `p3_resultados.py`, `p4_interpretacao.py`, `p5_comparacao.py` e `p6_como_utilizar.py`. Quando `st.page_link()` recebe a instância registrada de `st.Page`, o Streamlit acessa diretamente `page._script_hash`.

### 1.2. Exibição Literal de Código HTML nos Cartões de Resultados e Comparação
- **Mecanismo interno (CommonMark §4.4 e §4.6):** Strings HTML multilinha construídas com f-strings triplas (`f""" ... """`) dentro de funções Python indentadas continham de **8 a 28 espaços à esquerda** e linhas em branco internas, sendo enviadas diretamente a `st.markdown(..., unsafe_allow_html=True)` em vez da função utilitária [`render_html()`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/components.py#L56-L59). O parser Markdown do Streamlit interpreta linhas com 4 ou mais espaços de recuo (ou linhas subsequentes a linhas vazias em blocos HTML Tipo 6) como **blocos de código `<pre><code>`**, exibindo literalmente as tags `<div>`, `<span>` e até strings base64 de imagens PNG na tela.
- **Agravantes estruturais encontrados:**
  1. Em [`p5_comparacao.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p5_comparacao.py), cada cartão molecular era dividido em 3 chamadas separadas de `st.markdown()` (abertura `<div class="surface-n2">`, tag `<img>` indentada e fechamento órfão `</div>`), sendo que cada `st.markdown` cria um contêiner DOM isolado no React.
  2. Em [`p4_interpretacao.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p4_interpretacao.py), `_render_5_field_card` abria um `<div>` sem fechá-lo no cabeçalho e emitia um `st.markdown("</div>")` órfão ao final da função.
  3. Valores dinâmicos contendo o caractere `<` (ex.: `pIC50 < 5,0`, `P_hERG < 0,30` ou SMILES inválidos digitados pelo usuário) eram interpolados sem `html.escape()`, podendo truncar tags HTML adjacentes, e um bloco HTML bruto em `p3_resultados.py` continha notação LaTeX `$A_{2A}$` não processada dentro de `<span>`.

### 1.3. Falha nas Moléculas de Acesso Rápido e Fluxo de Nova Análise
- **Conflito de estado de widget no Streamlit:** Em [`p2_nova_analise.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p2_nova_analise.py), o campo `st.text_input("Código SMILES da Estrutura:", value=st.session_state.get("input_smiles_field", ""), key="field_smiles_text")` inicializava `st.session_state["field_smiles_text"] = ""` na primeira renderização. No ciclo de vida do Streamlit, uma vez que a `key` de um widget existe em `st.session_state`, o parâmetro `value=` passa a ser ignorado em reruns subsequentes.
- **Sobrescrita silenciosa no rerun:** Os botões de moléculas de acesso rápido (assim como os botões **"🗑️ Limpar"** e **"🔍 Ativar no Painel"** do histórico) atualizavam apenas `st.session_state["input_smiles_field"]`, sem atualizar `st.session_state["field_smiles_text"]`. No `st.rerun()` imediato, o `st.text_input` retornava o valor antigo `""` retido em `field_smiles_text` e a linha seguinte sobrescrevia `st.session_state["input_smiles_field"] = ""`, anulando completamente a seleção.
- **Preservação de Molfile V2000/V3000 do Ketcher:** Quando o usuário desenha uma estrutura no Ketcher e clica em **Apply**, `st_ketcher` pode retornar um bloco MDL Molfile V2000/V3000 (cujas 3 linhas de cabeçalho possuem espaços significativos exigidos pelo parser `Chem.MolFromMolBlock` do RDKit). Realizar `.strip()` prematuro na string retornada podia remover as linhas iniciais de cabeçalho do Molfile antes de `canonicalize_smiles()`.

### 1.4. Falha Funcional na Página Comparação (`p5_comparacao.py`)
A falha na página Comparação decorria da combinação de quatro fatores:
1. **Quebra no estado vazio (< 2 moléculas):** `st.page_link("nova-analise", ...)` lançava `StreamlitPageNotFoundError` imediatamente ao abrir a página com menos de 2 moléculas analisadas.
2. **Bloqueio de alimentação a montante:** Como os botões de acesso rápido em Nova Análise não carregavam estruturas e o link pós-análise para `resultados` também falhava com `StreamlitPageNotFoundError`, o fluxo natural para acumular 2+ moléculas no histórico ficava interrompido.
3. **Corrupção visual dos cartões comparativos:** Os cartões lado a lado da seção 2 dividiam um único `<div class="surface-n2">` em três `st.markdown` indentados, despejando código HTML bruto e a string base64 da imagem 2D na interface.
4. **Dessincronização dos widgets `comp_multiselect_molecules` e `comp_select_ref_molecule`:** Após a primeira visita à página Comparação, `st.session_state["comp_multiselect_molecules"]` ficava congelado (ignorando novas moléculas adicionadas posteriormente ao histórico FIFO ou mantendo SMILES já removidos além do limite de 10 itens), e desmarcar no `multiselect` a molécula atualmente selecionada em `comp_select_ref_molecule` causava falha de valor ausente nas opções do `selectbox`.

---

## 2. ARQUIVOS MODIFICADOS

### Arquivos de Produção Modificados
1. [`app.py`](file:///c:/Users/Raphael/Documents/BioPredicta/app.py)
2. [`src/ui/components.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/components.py)
3. [`src/ui/pages/p2_nova_analise.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p2_nova_analise.py)
4. [`src/ui/pages/p3_resultados.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p3_resultados.py)
5. [`src/ui/pages/p4_interpretacao.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p4_interpretacao.py)
6. [`src/ui/pages/p5_comparacao.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p5_comparacao.py)
7. [`src/ui/pages/p6_como_utilizar.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p6_como_utilizar.py)

### Arquivo de Teste Adicionado
8. [`tests/test_functional_stabilization.py`](file:///c:/Users/Raphael/Documents/BioPredicta/tests/test_functional_stabilization.py)

*(Observação: [`src/ui/history_manager.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/history_manager.py) foi inspecionado e mantido intacto, sem qualquer alteração em suas regras FIFO, deduplicação ou contrato de sessão).*

---

## 3. CORREÇÃO REALIZADA EM CADA ARQUIVO

1. **[`app.py`](file:///c:/Users/Raphael/Documents/BioPredicta/app.py#L2295-L2341):**
   - Instanciação explícita dos 9 objetos `st.Page` (`page_obj_visao_geral`, `page_obj_nova_analise`, `page_obj_resultados`, `page_obj_interpretacao`, `page_obj_comparacao`, `page_obj_como_utilizar`, `page_obj_fundamentos`, `page_obj_governanca`, `page_obj_sobre`).
   - Registro completo em `st.session_state["pages_registry"]` cobrindo todas as chaves canônicas (`url_path`) e aliases com hífen/sublinhado antes de `nav_controller.run()`.

2. **[`src/ui/components.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/components.py#L46-L105):**
   - Implementação e exportação de [`get_page_target(key: str)`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/components.py#L46-L54) para resolver o objeto `st.Page` registrado na sessão (com fallback seguro em testes unitários isolados).
   - Conversão de [`render_page_header`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/components.py#L77-L86) e [`render_provisional_page`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/components.py#L89-L105) para utilizar [`render_html()`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/components.py#L56-L59).
   - Inclusão de fallback `provenance.get("name")` em `compute_pairwise_tanimoto_matrix` e `_extract_mol_properties` para rotulagem consistente de controles offline na Comparação.

3. **[`src/ui/pages/p2_nova_analise.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p2_nova_analise.py):**
   - Criação de [`select_preset_molecule(smiles: str)`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p2_nova_analise.py#L90-L97), sincronizando simultaneamente `input_smiles_field`, `field_smiles_text` e `input_mode_selector = "✏️ Inserir SMILES"`, além de limpar `ketcher_molecular_canvas`.
   - Adição de sincronização pré-widget (`_pending_smiles_update` e `_pending_mode_update`) no início de [`render_nova_analise()`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p2_nova_analise.py#L569-L579) para permitir que os botões **"🗑️ Limpar"** e **"🔍 Ativar no Painel"** atualizem `field_smiles_text` sem violar a restrição de ciclo de vida de widgets do Streamlit.
   - Guarda contextual via `get_script_run_ctx() is not None` nos kwargs de `st.segmented_control` e `st.text_input` para eliminar conflitos entre `value`/`default` e `st.session_state` no servidor real, preservando compatibilidade com testes unitários *bare mode*.
   - Preservação integral (sem `.strip()` destrutivo) de blocos Molfile V2000/V3000 retornados por `st_ketcher` ao clicar em **Apply**, garantindo canonicalização RDKit e sincronização com a sessão.
   - Conversão de todos os cartões HTML multilinha para `render_html(...)`, aplicação de `html.escape()` em textos dinâmicos e substituição de `st.page_link` por `get_page_target("resultados")`.

4. **[`src/ui/pages/p3_resultados.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p3_resultados.py):**
   - Conversão de todos os 10 blocos HTML multilinha para `render_html(...)` com `html.escape()` nos rótulos dinâmicos (`pIC50 < 5,0`, `P_hERG < 0,30`, nomes e descrições).
   - Substituição de `$A_{2A}$` bruto dentro de `<span>` HTML por `A<sub>2A</sub>` e ajuste de lookup da descrição hERG (`desc` / `description`).
   - Atualização de todas as 5 chamadas `st.page_link()` para `get_page_target(...)`.

5. **[`src/ui/pages/p4_interpretacao.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p4_interpretacao.py):**
   - Conversão de todos os 25 blocos HTML para `render_html(...)`.
   - Correção do `<div>` não fechado / `</div>` órfão em `_render_5_field_card`, garantindo tags `<div>...</div>` perfeitamente balanceadas em cada componente.
   - Atualização de todas as 4 chamadas `st.page_link()` para `get_page_target(...)`.

6. **[`src/ui/pages/p5_comparacao.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p5_comparacao.py):**
   - Unificação dos 3 fragmentos separados de `st.markdown` dos cartões comparativos em um único bloco `<div class="surface-n2">...</div>` autocontido renderizado via `render_html(...)` (linhas 327–382), além dos cartões de estado vazio e notas metodológicas.
   - Sincronização pré-widget de `comp_multiselect_molecules` (removendo SMILES expirados pela fila FIFO e anexando automaticamente novas moléculas analisadas até o limite de 5, respeitando desmarcações manuais) e de `comp_select_ref_molecule` (redefinindo a referência caso ela seja removida da seleção ativa).
   - Atualização de `st.page_link()` no estado vazio para `get_page_target("nova-analise")` e sincronização de `_pending_smiles_update` no botão `"📌 Tornar Ativa"`.

7. **[`src/ui/pages/p6_como_utilizar.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p6_como_utilizar.py):**
   - Conversão dos 19 blocos HTML customizados nas 3 abas e 7 módulos para `render_html(...)`, preservando 100% do conteúdo científico e terminológico validado na Fase 5.1.
   - Atualização das 4 chamadas `st.page_link()` para `get_page_target(...)`.

8. **[`tests/test_functional_stabilization.py`](file:///c:/Users/Raphael/Documents/BioPredicta/tests/test_functional_stabilization.py):**
   - Suíte dedicada com 8 testes cobrindo auditoria AST global de `st.page_link`/`st.switch_page`, ausência de indentação `>= 4` espaços e balanceamento `<div>` em todas as páginas (estados vazio, inválido e ativo), presets de acesso rápido, botão Limpar, fluxo Ketcher Molfile V2000 + inferência, ativação a partir do histórico e sincronização de estado na Comparação.

---

## 4. TESTES EXECUTADOS E RESPECTIVOS RESULTADOS

| Etapa de Teste | Comando / Método Executado | Escopo Coberto | Resultado |
| :--- | :--- | :--- | :--- |
| **Testes Específicos das Páginas** | `pytest tests/test_ui_p2_nova_analise.py tests/test_ui_p3_resultados.py tests/test_ui_p4_interpretacao.py tests/test_ui_p5_comparacao.py tests/test_ui_p5_history.py tests/test_ui_p6_como_utilizar.py -q` | Todas as páginas modificadas e integração com `history_manager.py` | **131 passed** (0 falhas) |
| **Suíte de Estabilização Funcional** | `pytest tests/test_functional_stabilization.py -q` | Navegação AST global, auditoria CommonMark HTML, presets, Limpar, Ketcher V2000, Histórico e Comparação | **8 passed** (0 falhas) |
| **Suíte Completa (exceto benchmark)** | `pytest tests -m "not benchmark" -q` | Regressão completa de engenharia de dados, química, modelos, AD, SHAP e UI | **170 passed**, 1 deselected (0 falhas) |
| **Verificação E2E em Servidor Streamlit Real (HTTP + WebSocket Protobuf)** | `streamlit run app.py --server.headless true` acionado via cliente Tornado WebSocket (`/_stcore/stream`, `BackMsg` / `ForwardMsg`) | Inicialização Uvicorn/Streamlit real, navegação pelas 9 rotas, clique em Acesso Rápido (Cafeína e Istradefilina), inferência SMILES, troca para Ketcher + `Apply` Molfile V2000 (Taurina) + inferência, abertura de Resultados, Interpretação e Comparação com 3 moléculas na sessão e auditoria de zero exceções e zero HTML indentado nos `ForwardMsg` | **100% aprovado** (`exit code 0`, encerramento limpo) |

---

## 5. RESULTADO DA VERIFICAÇÃO VISUAL E DISTINÇÃO ENTRE TESTES AUTOMATIZADOS E NAVEGADOR

Em cumprimento estrito à diretriz de transparência técnica, diferencia-se claramente o que foi comprovado programaticamente daquilo que requer inspeção visual humana no navegador:

1. **O que foi comprovado em servidor Streamlit real (`streamlit run app.py` via protocolo HTTP + WebSocket Protobuf):**
   - O servidor real Uvicorn/Streamlit 1.65.0 subiu sem erros (`/_stcore/health == ok`) e registrou as 9 rotas (`''`, `'nova-analise'`, `'resultados'`, `'interpretacao'`, `'comparacao'`, `'como-utilizar'`, `'fundamentos'`, `'governanca'`, `'sobre'`).
   - Todas as 9 páginas foram abertas em sessão real sem nenhum elemento `exception` (`0` ocorrências de `StreamlitPageNotFoundError` ou `StreamlitAPIException`).
   - Todos os elementos `markdown` com `allow_html=True` emitidos pelo servidor ao vivo foram inspecionados linha a linha nos pacotes Protobuf `ForwardMsg`: **0 linhas vazias internas e 0 linhas com recuo >= 4 espaços**, confirmando que o CommonMark não converte mais os cartões em blocos `<pre><code>` de HTML literal.
   - O clique real via `WidgetState.trigger_value` nos botões de acesso rápido (**Cafeína** e **Istradefilina**) preencheu o `text_input` (`Código SMILES da Estrutura:`) no `ForwardMsg` devolvido pelo servidor e executou o pipeline completo de predição multiobjetivo.
   - A troca do `segmented_control` para **`🎨 Desenhar Molécula (Ketcher)`** instanciou o `component_instance` do Ketcher, recebeu um Molfile V2000 da **Taurina** via `json_value` (simulando o botão **Apply**), canonicalizou a estrutura no RDKit e concluiu a análise multiobjetivo.
   - As páginas **Resultados**, **Interpretação** e **Comparação** (com as 3 moléculas analisadas selecionadas automaticamente no `multiselect`) renderizaram todos os cartões, tabelas e gráficos sem exceções.

2. **O que NÃO foi inspecionado por renderização óptica humana e requer validação manual no navegador gráfico:**
   - Como o ambiente de execução do agente é *headless* (sem navegador gráfico interativo controlado visualmente por usuário humano), **a confirmação visual final de layout CSS, alinhamento estético dos cartões Dark Mode, interação manual com o canvas JavaScript dentro do `<iframe>` do Ketcher pelo mouse e cliques manuais nos links do navegador deve ser validada manualmente pelo avaliador** executando `streamlit run app.py` e abrindo `http://localhost:8501` no navegador.

---

## 6. CONFIRMAÇÃO DOS OITO HASHES SHA-256 DE REFERÊNCIA

Verificação criptográfica executada após todas as modificações e baterias de testes:

| Arquivo Protegido | Hash SHA-256 Calculado | Status |
| :--- | :--- | :--- |
| `models/a2a_model.joblib` | `1907C7241EDF64FFAE3AA5C59A497037D427B1EAD6B83ABDF15ABBFE3B2A4787` | ✅ 100% ÍNTEGRO |
| `models/a2a_model_metadata.json` | `DA3E05936977254E08C34DAB6B21009C3DA2C632647E062522662BD47A1D5F8F` | ✅ 100% ÍNTEGRO |
| `models/herg_model.joblib` | `FCEDBECB89843870FE1B7DA9C501EB62B8E9E6A0A254A9ED8139C17D6AD154E6` | ✅ 100% ÍNTEGRO |
| `models/herg_model_metadata.json` | `781F4B07B68EEEFF8F7FE245AA7513073811BB78819FE16996902C4D3146115F` | ✅ 100% ÍNTEGRO |
| `data/processed/a2a_curated.csv` | `88354901075BC9058B6AE6B5AD8920A4A710F10B2EFD32ABD8AB02541B977A3E` | ✅ 100% ÍNTEGRO |
| `data/processed/herg_curated.csv` | `DF7BE71479E64C05F735B17B35A6FD888CC6CA869615E9BDE50F489C715B3798` | ✅ 100% ÍNTEGRO |
| `data/processed/a2a_featurized.parquet` | `AC006ABB90B59AC74C6D3A14CAD25CEEE62C58CEE44FA263A8A69F90928B31B7` | ✅ 100% ÍNTEGRO |
| `data/processed/herg_featurized.parquet` | `612E115EBA6E47788ADB92A2EDD067E89D5AF9F5095DBA0DFCAD6FC817D7EB92` | ✅ 100% ÍNTEGRO |

Nenhum modelo, metadado, base curada, arquivo Parquet, hiperparâmetro, conjunto de 1030 features, limiar de classificação (`5,0`/`6,5` para $A_{2A}$ e `0,30`/`0,70` para hERG) ou cálculo de $pIC_{50}$, $P(\text{hERG}=1)$, AD ou SHAP foi alterado.

---

## 7. ITENS FORA DO ESCOPO OBSERVADOS (DOCUMENTADOS E NÃO MODIFICADOS)

1. **Páginas Provisórias Agendadas no Roteiro UX/UI:** As rotas `Visão Geral` (`/`), `Fundamentos Científicos` (`/fundamentos`), `Modelos e Governança` (`/governanca`) e `Sobre o Projeto` (`/sobre`) continuam exibindo o componente informativo `render_provisional_page` (cujo HTML também foi estabilizado via `render_html`), pois a implementação de seu conteúdo definitivo pertence às próximas etapas planejadas da Central de Ajuda e Institucional.
2. **Modo de Transição Legado em `Visão Geral` (`chk_enable_legacy_view`):** Em [`app.py`](file:///c:/Users/Raphael/Documents/BioPredicta/app.py#L2231-L2243), a rota provisória `Visão Geral` mantém uma caixa de seleção opcional para exibir o painel monolítico legado durante a transição entre fases; mantida inalterada para não realizar refatoração estrutural fora do escopo.
3. **Dependência de Rede para Identificação PubChem:** Quando sem conexão externa com a API PUG REST da NCBI/PubChem, o pipeline opera normalmente em modo resiliente (*offline fallback*), exibindo aviso não bloqueante e executando integralmente as predições locais RDKit/XGBoost/Random Forest/AD/SHAP.

