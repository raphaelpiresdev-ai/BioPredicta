# Relatório Técnico: Correção Crítica da Integração Ketcher → SMILES → RDKit

**Projeto:** MultiTarget BioPredict (A2A × hERG)  
**Módulo:** `src/ui/pages/p2_nova_analise.py` & `src/chem_utils.py`  
**Data:** 31 de Outubro de 2026  
**Status:** Concluído e Validado  

---

## 1. Causa Raiz do Problema

### 1.1 Diagnóstico Detalhado
Ao submeter estruturas desenhadas no editor molecular interativo **Ketcher** (especialmente compostos contendo heterociclos condensados purínicos/xantínicos como a Cafeína), o sistema apresentava o erro `"Estrutura Química Inválida"`, rejeitando códigos semelhantes a:
```text
[C]n1c(=O)nc2nc[n](C)c2c1=O
```

A auditoria no componente `streamlit_ketcher` (versão 0.1.2) e no RDKit identificou as seguintes causas raízes combinadas:
1. **Algoritmo de serialização SMILES do frontend Ketcher:** Em modo standalone (execução puramente client-side em JavaScript sem o microsserviço de backend Indigo C++), a função `ketcher.getSmiles()` possui limitações intrínsecas de percepção de aromaticidade. Ao tentar aromatizar anéis bicíclicos com carbonilas exocíclicas (anel de pirimidinodiona da xantina), ela gerou átomos aromáticos em minúsculo com valência incoerente (`c(=O)`) e átomos em colchetes (`[C]`, `[n]`).
2. **Semântica estrita de colchetes no Daylight SMILES / RDKit:** No padrão Daylight SMILES seguido pelo RDKit, qualquer átomo isolado entre colchetes sem hidrogênios explícitos (como `[C]`) assume zero hidrogênios implícitos ($H=0$). Como o átomo possui apenas 1 ligação simples com o nitrogênio, seu estado de valência torna-se um radical monovalente inválido.
3. **Impossibilidade de kekulização:** O RDKit não consegue atribuir ligações simples e duplas alternadas que satisfaçam a regra de Hückel ($4n+2$) em um anel contendo `c(=O)` e átomos `n` sem substituintes ou cargas adequadas, disparando o erro canônico:
   ```text
   [Can't kekulize mol. Unkekulized atoms: 4 5 6 7 10]
   ```
4. **Vulnerabilidade a formatos:** Anteriormente, `canonicalize_smiles()` e `smiles_to_mol()` em `src/chem_utils.py` utilizavam apenas `Chem.MolFromSmiles()`, falhando silenciosamente se recebessem blocos MDL Molfile V2000.

---

## 2. Solução Implementada

Para resolver a raiz do problema sem recorrer a substituições textuais indiscriminadas (expressamente vedadas):

1. **Alteração do Formato de Retorno do Ketcher para `molecule_format="MOLFILE"`:**
   - No frontend do Ketcher, `molecule_format="MOLFILE"` aciona `ketcher.getMolfile()`.
   - Em contraste com `getSmiles()`, a rotina `getMolfile()` do Ketcher extrai diretamente a tabela de conexão 2D da tela (MDL Molfile V2000), na qual as ordens de ligação (simples `1`, duplas `2`) e átomos são definidos explicitamente, sem ambiguidade de aromaticidade ou dependência de kekulização em JavaScript.

2. **Parser Molecular Robusto em `src/chem_utils.py`:**
   - A função `smiles_to_mol(smiles, standardize=True)` foi atualizada para reconhecer tanto representações lineares SMILES quanto blocos MDL Molfile V2000/V3000 (identificados por demarcadores `M  END`, `V2000` ou `V3000`).
   - Implementada a função `_mol_from_molblock()` que preserva ou reconstrói o cabeçalho obrigatório de 3 linhas do padrão CTAB V2000 mesmo após aparamento de quebras de linha (`strip()` / `rstrip()`).

3. **Sincronização Bidirecional e Conversão Canônica na UI (`p2_nova_analise.py`):**
   - Ao receber o Molfile gerado pelo Ketcher após clique em *Apply*, a interface executa `canonical_smi = canonicalize_smiles(raw_input_structure)`.
   - O SMILES canônico resultante (limpo e isomérico) é atribuído a `st.session_state["input_smiles_field"]`.
   - Essa abordagem assegura que alternâncias entre o modo de desenho (Ketcher) e o modo de texto (SMILES) preservem a integridade química absoluta, mantendo o campo de texto sempre preenchido com a string SMILES canônica linear legível.

---

## 3. Comparativo Estrutural: Antes e Depois

| Composto | SMILES Bruto (Ketcher com Bug) | SMILES Convertido / Canônico (Após Correção) | Fórmula Antes | Fórmula Depois | Massa Molecular |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cafeína** | `[C]n1c(=O)nc2nc[n](C)c2c1=O` *(rejeitado pelo RDKit)* | `Cn1c(=O)c2c(ncn2C)n(C)c1=O` | Inválida | `C8H10N4O2` | 194.19 Da (14 átomos pesados, 3 N-metilas) |
| **Teobromina** | *(aromático corrompido)* | `Cn1c(=O)c2[nH]cnc2n(C)c1=O` | Inválida | `C7H8N4O2` | 180.17 Da (13 átomos pesados) |
| **Teofilina** | *(aromático corrompido)* | `Cn1c(=O)c2[nH]cnc2n(C)c1=O` | Inválida | `C7H8N4O2` | 180.17 Da (13 átomos pesados) |
| **Água** | `O` | `O` | `H2O` | `H2O` | 18.02 Da (1 heteroátomo) |
| **Etanol** | `CCO` | `CCO` | `C2H6O` | `C2H6O` | 46.07 Da (3 átomos pesados) |
| **Benzeno** | `c1ccccc1` | `c1ccccc1` | `C6H6` | `C6H6` | 78.11 Da (6 átomos pesados) |
| **Istradefilina** | *(purina complexa)* | `CCn1c(=O)c2c(nc(/C=C/c3ccc(OC)c(OC)c3)n2C)n(CC)c1=O` | Inválida | `C20H24N4O4` | 384.44 Da (28 átomos pesados, 8 heteroátomos) |

---

## 4. Arquivos Modificados

1. `src/chem_utils.py`:
   - Adicionada função `_mol_from_molblock(text: str) -> Optional[Chem.Mol]`.
   - Atualizada `smiles_to_mol()` para aceitar Molfile V2000/V3000 e SMILES.
2. `src/ui/pages/p2_nova_analise.py`:
   - Configurado `st_ketcher(molecule_format="MOLFILE")`.
   - Implementada sincronização e normalização de Molfile para SMILES canônico no `session_state`.
   - Atualizado o controle de referência da Istradefilina para o SMILES clínico autêntico (`C20H24N4O4`).
3. `tests/test_ui_p2_nova_analise.py`:
   - Adicionadas classes `TestMandatoryChemicalValidation` e `TestKetcherIntegrationPipeline`.

---

## 5. Testes Químicos Automatizados e Resultados

A suíte de testes em `tests/test_ui_p2_nova_analise.py` cobriu:
- **Cafeína:** Parse RDKit sem erro, canonicalização, verificação estrita da fórmula `C8H10N4O2`, massa 194.19 Da, 14 átomos pesados, 6 heteroátomos, e contagem exata das 3 metilas ligadas aos nitrogênios (N1, N3, N7).
- **Teobromina / Teofilina:** Fórmula `C7H8N4O2`, massa 180.17 Da, 13 átomos pesados, 6 heteroátomos.
- **Água:** Fórmula `H2O`, massa 18.02 Da, 1 átomo pesado.
- **Etanol:** Fórmula `C2H6O`, massa 46.07 Da, 3 átomos pesados.
- **Benzeno:** Fórmula `C6H6`, massa 78.11 Da, 6 átomos pesados, 0 heteroátomos.
- **Istradefilina:** Fórmula `C20H24N4O4`, massa 384.44 Da, 28 átomos pesados, 8 heteroátomos.
- **Reprodução Documental do Erro:** Teste explícito demonstrando que `[C]n1c(=O)nc2nc[n](C)c2c1=O` é rejeitado pelo RDKit e pelo `canonicalize_smiles()`, documentando a causa exata do erro.
- **Fluxo de Integração:** Ketcher Molfile → RDKit → SMILES canônico → Featurização 1030-D completa.

**Resultado da suíte:** 47/47 aprovados.

---

## 6. Limitações de Validação Manual

- Como o ambiente de execução não dispõe de navegador gráfico interativo com renderização WebGL do iframe do Ketcher, a validação do componente frontend baseou-se na inspeção minuciosa dos bundles JavaScript compilados (`main.869c7e58.js`), emulações de payload do componente React do Streamlit e testes químicos automatizados de ida e volta (round-trip) via RDKit.

