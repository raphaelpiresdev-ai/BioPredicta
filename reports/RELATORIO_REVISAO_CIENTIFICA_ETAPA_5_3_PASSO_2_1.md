# RELATÓRIO TÉCNICO — ETAPA 5.3 (PASSO 2.1)
## Verificação de Precisão Científica da Página de Comparação

**Projeto:** MultiTarget BioPredict  
**Fase:** Etapa 5.3 — Passo 2.1 (Revisão Científica Pontual)  
**Data:** 31 de Outubro de 2026  
**Branch Git:** `feat/ajustes-funcionalidades`  
**Status da Suíte de Testes:** 143 testes aprovados (100% de sucesso, 0 falhas)  
**Status dos Hashes Protegidos:** 8/8 íntegros e verificados criptograficamente  

---

## 1. INTRODUÇÃO E ESCOPO

Este relatório documenta a revisão pontual de rigor científico realizada na página de comparação multimolecular ([`src/ui/pages/p5_comparacao.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p5_comparacao.py)) e nos seus respectivos testes ([`tests/test_ui_p5_comparacao.py`](file:///c:/Users/Raphael/Documents/BioPredicta/tests/test_ui_p5_comparacao.py)).

Nenhuma funcionalidade, cálculo de similaridade, algoritmo de featurização ou modelo preditivo foi modificado ou refatorado. A intervenção concentrou-se estritamente na fidelidade terminológica e precisão quimioinformática das legendas e descrições da interface.

---

## 2. DIAGNÓSTICO DOS TRÊS PONTOS REVISADOS

### Ponto 1: Similaridade Tanimoto e Interpretação de Scaffold
- **Diagnóstico:** A nota metodológica anterior citava de forma vaga que coeficientes próximos a 1,0 indicavam "homologia estrutural direta". Embora a matriz utilize rigorosamente Morgan ECFP4 (raio 2, 1024 bits), valores elevados de similaridade de Tanimoto mensuram a sobreposição de fragmentos circulares locais e não garantem identidade ou conservação estrita do *scaffold* (esqueleto central de Murcko / Bemis-Murcko).
- **Ação:** A legenda foi refinada para esclarecer que similaridade $> 0,70$ reflete alta concordância de subambientes circulares locais, sem implicar necessariamente identidade ou conservação de scaffold.

### Ponto 2: Terminologia dos Níveis de Domínio de Aplicabilidade (AD)
- **Diagnóstico:** A função `_render_ad_badge` continha descrições manuais informais como `"Nível 1 (Estrutura idêntica)"` e `"Nível 4 (Interpolação / Extrapolação)"`. Na implementação canônica de governança em [`src/ad_utils.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ad_utils.py), os níveis são rigorosamente definidos pelas constantes:
  - Nível 1: `"Correspondência de estrutura"` (`PROVENANCE_LEVEL_1_STRUCTURE`)
  - Nível 2: `"Correspondência de features (vetor 1.030-D)"` (`PROVENANCE_LEVEL_2_FEATURES`)
  - Nível 3: `"Fingerprint idêntico (1.024 bits)"` (`PROVENANCE_LEVEL_3_FINGERPRINT`)
  - Nível 4: `"Sem correspondência exata conhecida"` (`PROVENANCE_LEVEL_4_NO_EXACT_MATCH`)
  A atribuição de "Interpolação / Extrapolação" para o Nível 4 extrapolava a definição canônica, pois o Nível 4 apenas atesta a ausência de correspondência exata nos dados de calibração, quantificando a proximidade via $S_{\max}$ frente ao vizinho mais próximo.
- **Ação:** As constantes canônicas de `src.ad_utils` foram importadas diretamente para `p5_comparacao.py`, eliminando qualquer termo improvisado e garantindo 100% de fidelidade conceitual.

### Ponto 3: Terminologia do Resultado hERG
- **Diagnóstico:** A saída do modelo hERG deve ser inequivocamente apresentada como a saída probabilística do classificador supervisionado Random Forest para a classe hERG positiva ($P(\text{hERG}=1)$), e nunca como probabilidade clínica calibrada ou incidência de eventos adversos cardíacos.
- **Ação:** Todos os rótulos de interface e notas explicativas foram ajustados para registrar expressamente `Saída P(hERG=1)` e advertir que o escore reflete a propensão do modelo a partir do espaço de descritores e não probabilidade clínica de cardiotoxicidade.

---

## 3. TRECHOS ANTERIORES E POSTERIORES (DIFF CONCEITUAL)

### 3.1. Badges de Domínio de Aplicabilidade (AD)
**Arquivo:** [`src/ui/pages/p5_comparacao.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p5_comparacao.py#L44-L68)

```python
# ANTES:
    if level == 1:
        color = "#00D2BE"
        desc = "Nível 1 (Estrutura idêntica)"
    elif level == 2:
        color = "#38BDF8"
        desc = "Nível 2 (Features 1030-D)"
    elif level == 3:
        color = "#F59E0B"
        desc = "Nível 3 (Fingerprint idêntico)"
    elif level == 4:
        color = "#CBD5E1"
        desc = "Nível 4 (Interpolação / Extrapolação)"

# DEPOIS:
    if level == 1:
        color = "#00D2BE"
        desc = f"Nível 1 — {PROVENANCE_LEVEL_1_STRUCTURE}"
    elif level == 2:
        color = "#38BDF8"
        desc = f"Nível 2 — {PROVENANCE_LEVEL_2_FEATURES}"
    elif level == 3:
        color = "#F59E0B"
        desc = f"Nível 3 — {PROVENANCE_LEVEL_3_FINGERPRINT}"
    elif level == 4:
        color = "#CBD5E1"
        desc = f"Nível 4 — {PROVENANCE_LEVEL_4_NO_EXACT_MATCH}"
```

### 3.2. Nota Metodológica da Matriz de Tanimoto (ECFP4)
**Arquivo:** [`src/ui/pages/p5_comparacao.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p5_comparacao.py#L342-L352)

```python
# ANTES:
    st.caption(
        "Coeficientes de similaridade par-a-par baseados em fingerprints circulares de Morgan (raio 2, 1024 bits). "
        "A diagonal é estritamente 1,000 (auto-similaridade). Coeficientes próximos a 1,0 indicam homologia estrutural direta."
    )

# DEPOIS:
    st.caption(
        "Coeficientes de similaridade par-a-par calculados via Tanimoto sobre fingerprints Morgan (ECFP4, raio 2, 1024 bits). "
        "A diagonal é estritamente 1,000 (auto-similaridade). Valores superiores a 0,70 indicam alta similaridade estrutural "
        "baseada na sobreposição de ambientes circulares locais, sem implicar necessariamente conservação ou identidade de scaffold central."
    )
```

### 3.3. Nota Científica e Rótulos da Predição hERG
**Arquivo:** [`src/ui/pages/p5_comparacao.py`](file:///c:/Users/Raphael/Documents/BioPredicta/src/ui/pages/p5_comparacao.py#L206-L218)

```python
# ANTES:
        <div style="background:#1E222D; border:1px solid #333B4D; border-radius:8px; padding:10px 14px; font-size:0.83rem; color:#94A3B8; margin-top:6px;">
            <strong style="color:#00D2BE;">Nota Científica:</strong> O eixo vertical representa o <em>Índice Complementar</em> <code>1 - P(hERG)</code>, 
            onde maiores valores denotam menor escore de saída do classificador. 
            O escore <code>P(hERG=1)</code> reflete a propensão do modelo e não constitui probabilidade clínica calibrada de cardiotoxicidade.
        </div>

# DEPOIS:
        <div style="background:#1E222D; border:1px solid #333B4D; border-radius:8px; padding:10px 14px; font-size:0.83rem; color:#94A3B8; margin-top:6px;">
            <strong style="color:#00D2BE;">Nota Científica:</strong> O eixo vertical representa o <em>Índice Complementar</em> <code>1 - P(hERG)</code>, 
            derivado da saída probabilística do classificador para a classe positiva (<code>P(hERG=1)</code>). 
            Esse escore expressa a propensão estimada pelo modelo supervisionado a partir dos descritores e fingerprints moleculares e 
            <strong>não</strong> constitui probabilidade clínica calibrada de cardiotoxicidade nem estimativa direta de incidência de eventos adversos cardíacos.
        </div>
```

---

## 4. RESULTADOS DOS TESTES AUTOMATIZADOS

### 4.1. Suíte Específica de Comparação (`tests/test_ui_p5_comparacao.py`)
- Adicionadas verificações explícitas de não-afirmação de conservação estrita de scaffold (`test_tanimoto_methodological_note_does_not_claim_scaffold_identity`) e assertivas de que as strings canônicas de `ad_utils` estão presentes nos badges (`test_ad_badge_a2a_and_herg_separate`).
- **Resultado:** **28 testes executados e 100% aprovados** em 13.35s.

### 4.2. Suíte Global de Regressão do Projeto
Comando executado: `pytest tests -v -m "not benchmark"`

```
================ 143 passed, 1 deselected in 65.12s (0:01:05) =================
```
- Total de testes coletados: 144
- Testes aprovados: **143 (100%)**
- Falhas / Erros: **0**
- Testes desmarcados: 1 (benchmark de featurização)

---

## 5. VERIFICAÇÃO CRIPTOGRÁFICA DOS 8 HASHES SHA-256

Todos os 8 arquivos protegidos do projeto permanecem 100% íntegros e inalterados:

| Arquivo Protegido | Algoritmo | Hash SHA-256 Verificado | Integridade |
| :--- | :---: | :---: | :---: |
| `models/a2a_model.joblib` | SHA-256 | `1907c7241edf64ffae3aa5c59a497037d427b1ead6b83abdf15abbfe3b2a4787` | Íntegro |
| `models/a2a_model_metadata.json` | SHA-256 | `da3e05936977254e08c34dab6b21009c3da2c632647e062522662bd47a1d5f8f` | Íntegro |
| `models/herg_model.joblib` | SHA-256 | `fcedbecb89843870fe1b7da9c501eb62b8e9e6a0a254a9ed8139c17d6ad154e6` | Íntegro |
| `models/herg_model_metadata.json` | SHA-256 | `781f4b07b68eeeff8f7fe245aa7513073811bb78819fe16996902c4d3146115f` | Íntegro |
| `data/processed/a2a_curated.csv` | SHA-256 | `88354901075bc9058b6ae6b5ad8920a4a710f10b2efd32abd8ab02541b977a3e` | Íntegro |
| `data/processed/herg_curated.csv` | SHA-256 | `df7be71479e64c05f735b17b35a6fd888cc6ca869615e9bde50f489c715b3798` | Íntegro |
| `data/processed/a2a_featurized.parquet` | SHA-256 | `ac006abb90b59ac74c6d3a14cad25ceee62c58cee44fa263a8a69f90928b31b7` | Íntegro |
| `data/processed/herg_featurized.parquet` | SHA-256 | `612e115eba6e47788adb92a2edd067e89d5af9f5095dba0dfcad6fc817d7eb92` | Íntegro |

---

## 6. DECLARAÇÃO DE CONFORMIDADE COM AS RESTRIÇÕES

1. Nenhum modelo, metadado ou dataset foi alterado.
2. Nenhuma predição ou valor de SHAP foi recalculado.
3. As funções de cálculo de Tanimoto e AD permaneceram rigorosamente idênticas.
4. `history_manager.py` e `p2_nova_analise.py` permaneceram intactos.
5. A função `build_multiobjective_figure()` foi preservada.
6. Nenhum commit foi executado.

*Execução do Passo 2.1 concluída e interrompida. Aguardando autorização explícita para os próximos passos.*
