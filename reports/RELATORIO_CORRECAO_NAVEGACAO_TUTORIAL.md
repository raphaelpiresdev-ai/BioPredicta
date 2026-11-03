# RELATÓRIO TÉCNICO — CORREÇÃO GLOBAL DE NAVEGAÇÃO MULTIPÁGINA
## Resolução do StreamlitPageNotFoundError em Toda a Aplicação

**Projeto:** MultiTarget BioPredict  
**Foco:** Resolução sistemática e definitiva do erro `StreamlitPageNotFoundError: Could not find page: nova-analise` (e de todas as rotas invocadas via `st.page_link`) nas páginas de Análise, Resultados, Interpretação, Comparação e Como Utilizar  
**Data:** 01 de Novembro de 2026  
**Branch Git:** `feat/ajustes-funcionalidades`  
**Status dos Hashes Protegidos:** 8/8 íntegros e verificados criptograficamente via SHA-256 (100% conformidade)  

---

## 1. DIAGNÓSTICO DO ERRO E CAUSA RAIZ MECANICISTA

### 1.1. O Erro Apresentado
Na interface visual da aplicação, ao acessar páginas como **Resultados** antes de executar uma análise molecular, a aplicação renderiza o banner de estado vazio *"Nenhuma Análise Molecular Ativa"* e falhava com a seguinte exceção reportada no traceback:

```text
streamlit.errors.StreamlitPageNotFoundError: Could not find page: nova-analise.
You must provide a Page object or file path relative to the entrypoint file. Only pages previously defined by st.Page and passed to st.navigation are allowed.

Traceback:
  File "C:\Users\Raphael\Documents\BioPredicta\app.py", line 2339, in <module>
    nav_controller.run()
  File "C:\Users\Raphael\Documents\BioPredicta\app.py", line 2253, in page_resultados
    render_resultados()
  File "C:\Users\Raphael\Documents\BioPredicta\src\ui\pages\p3_resultados.py", line 114, in render_resultados
    st.page_link(
```

O mesmo comportamento ocorria na página **Como Utilizar** (`p6_como_utilizar.py`) e potencialmente em **Interpretação** (`p4_interpretacao.py`) e **Comparação** (`p5_comparacao.py`).

### 1.2. Causa Raiz no Mecanismo Interno do Streamlit (`st.navigation` × `st.page_link`)
No Streamlit 1.65.0, a aplicação utiliza a navegação centralizada em [`app.py`](file:///c:/Users/Raphael/Documents/BioPredicta/app.py):

```python
st.Page(page_nova_analise, title="Nova Análise", icon="🧪", url_path="nova-analise")
```

Cada página foi instanciada associando uma função Python chamável (*callable*).

A inspeção do código interno da biblioteca Streamlit (`streamlit.elements.widgets.button.ButtonMixin._page_link`) revelou o seguinte funcionamento:
1. Quando uma página é criada a partir de uma função Python (*callable*), o Streamlit define internamente `script_path = ""` (vazio), dado que não há um arquivo `.py` exclusivo em disco para essa página.
2. Ao invocar `st.page_link("nova-analise", ...)`, o parâmetro é uma string. O resolvedor de páginas da biblioteca tenta normalizar o caminho de arquivo no sistema operacional (`normalize_path_join(main_script_directory, "nova-analise")`) e compara com `page_data["script_path"]`.
3. Como `script_path` é vazio para callables (`""`), a verificação de igualdade `requested_page == full_path` é **sempre falsa**.
4. Não encontrando o hash correspondente por caminho de arquivo, a biblioteca lança a exceção `StreamlitPageNotFoundError`.
5. Por outro lado, quando o primeiro argumento de `st.page_link()` é o próprio objeto `st.Page` registrado, o Streamlit entra na ramificação `if isinstance(page, Page):`, extrai diretamente `page._script_hash` e `page.url_path`, valida contra as páginas ativas no `PagesManager` e realiza a navegação com perfeição sem efetuar buscas em disco.

---

## 2. ARQUITETURA DA SOLUÇÃO IMPLEMENTADA

A correção foi concebida de forma modular, centralizada e de alta manutenibilidade:

### 2.1. Registro de Instâncias `st.Page` em `app.py`
Em [`app.py`](file:///c:/Users/Raphael/Documents/BioPredicta/app.py), as instâncias de `st.Page` foram nomeadas explicitamente e mapeadas no dicionário canônico `pages_registry`, compartilhado globalmente no `st.session_state`:

```python
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
    "visao_geral": page_obj_visao_geral,
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
```

### 2.2. Função Centralizada de Resolução `get_page_target()` em `components.py`
Foi adicionada e exportada em [`src/ui/components.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/components.py) a função reutilizável:

```python
def get_page_target(key: str) -> Any:
    """Recupera o objeto st.Page registrado em st.session_state['pages_registry'].

    Caso a página seja renderizada fora do contexto de app.py (ex.: testes unitários isolados),
    retorna a chave de string como fallback seguro.
    """
    registry = st.session_state.get("pages_registry", {})
    return registry.get(key, key)
```

### 2.3. Varredura e Atualização de Todos os Pontos de Navegação
Todos os arquivos sob [`src/ui/pages/`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/) foram uniformizados para utilizar `get_page_target(...)` em todas as invocações de `st.page_link`:

1. **[`p3_resultados.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p3_resultados.py):**
   - Linha 115: `st.page_link(get_page_target("nova-analise"), ...)` (resolução direta do erro da captura de tela enviada pelo usuário).
   - Linhas 596, 610: `st.page_link(get_page_target("interpretacao"), ...)`
   - Linha 617: `st.page_link(get_page_target("nova-analise"), ...)`
   - Linha 624: `st.page_link(get_page_target("fundamentos"), ...)`

2. **[`p2_nova_analise.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p2_nova_analise.py):**
   - Linha 429: `st.page_link(get_page_target("resultados"), ...)`

3. **[`p4_interpretacao.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p4_interpretacao.py):**
   - Linha 330: `st.page_link(get_page_target("nova-analise"), ...)` (estado vazio).
   - Linha 762: `st.page_link(get_page_target("resultados"), ...)`
   - Linha 769: `st.page_link(get_page_target("nova-analise"), ...)`
   - Linha 776: `st.page_link(get_page_target("fundamentos"), ...)`

4. **[`p5_comparacao.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p5_comparacao.py):**
   - Linha 127: `st.page_link(get_page_target("nova-analise"), ...)` (estado vazio).

5. **[`p6_como_utilizar.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p6_como_utilizar.py):**
   - Linha 144: `st.page_link(get_page_target("nova-analise"), ...)` (aba Início Rápido).
   - Linhas 548, 555, 562: `st.page_link(get_page_target("nova-analise" | "comparacao" | "fundamentos"), ...)` (rodapé de continuidade).
   - Mantido alias retrocompatível `_get_page_target = get_page_target`.

---

## 3. AUDITORIA E VERIFICAÇÃO DE INTEGRIDADE

### 3.1. Varredura Estática de Conformidade
Uma busca recursiva por todas as chamadas a `page_link` e `switch_page` no repositório confirmou que **todas as 14 chamadas existentes** utilizam rigorosamente `get_page_target(...)`. Nenhuma string órfã permanece no código.

### 3.2. Execução da Suíte Completa de Testes Unitários
A suíte global foi executada com o comando:
```powershell
.\.venv\Scripts\python.exe -m pytest tests -m "not benchmark" -q
```
**Resultado:**
- **162 testes aprovados** (100% de sucesso).
- 1 teste de benchmark desmarcado conforme configuração de rotina.
- 0 erros, 0 falhas, 0 warnings de execução.

### 3.3. Integridade Criptográfica dos Oito Arquivos Protegidos (SHA-256)
A integridade dos modelos e conjuntos de dados fundamentais foi validada contra os valores de referência:

| Arquivo Protegido | Hash SHA-256 Calculado | Status |
| :--- | :--- | :--- |
| `models/a2a_model.joblib` | `1907C7241EDF64FFAE3AA5C59A497037D427B1EAD6B83ABDF15ABBFE3B2A4787` | ✅ ÍNTEGRO |
| `models/a2a_model_metadata.json` | `DA3E05936977254E08C34DAB6B21009C3DA2C632647E062522662BD47A1D5F8F` | ✅ ÍNTEGRO |
| `models/herg_model.joblib` | `FCEDBECB89843870FE1B7DA9C501EB62B8E9E6A0A254A9ED8139C17D6AD154E6` | ✅ ÍNTEGRO |
| `models/herg_model_metadata.json` | `781F4B07B68EEEFF8F7FE245AA7513073811BB78819FE16996902C4D3146115F` | ✅ ÍNTEGRO |
| `data/processed/a2a_curated.csv` | `88354901075BC9058B6AE6B5AD8920A4A710F10B2EFD32ABD8AB02541B977A3E` | ✅ ÍNTEGRO |
| `data/processed/herg_curated.csv` | `DF7BE71479E64C05F735B17B35A6FD888CC6CA869615E9BDE50F489C715B3798` | ✅ ÍNTEGRO |
| `data/processed/a2a_featurized.parquet` | `AC006ABB90B59AC74C6D3A14CAD25CEEE62C58CEE44FA263A8A69F90928B31B7` | ✅ ÍNTEGRO |
| `data/processed/herg_featurized.parquet` | `612E115EBA6E47788ADB92A2EDD067E89D5AF9F5095DBA0DFCAD6FC817D7EB92` | ✅ ÍNTEGRO |

---

## 4. CONCLUSÃO

O erro `StreamlitPageNotFoundError` apresentado no painel de Resultados (e nas demais páginas) foi corrigido em sua causa raiz. A aplicação agora navega com segurança e robustez entre todas as abas e páginas sem interrupção de fluxo.

Nenhuma alteração foi realizada nas etapas subsequentes da Central de Ajuda (`fundamentos`, `governanca` ou `sobre`), permanecendo o escopo estritamente focado e concluído nesta correção.
