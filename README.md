# Geração de Dados Sintéticos via VAE e Auditoria Lógica Neuro-Simbólica

Uma abordagem neuro-simbólica para predição de risco cardiovascular com foco em privacidade, explicabilidade e segurança clínica factual.

---

## Sobre o Projeto

Projeto final da disciplina **XAI com Raciocínio Automatizado (2026.1)** do Programa de Pós-Graduação em Ciência da Computação (PPGCC) — Instituto Federal do Ceará (IFCE).

O trabalho propõe um pipeline neuro-simbólico completo que integra quatro componentes:

- **TVAE (SDV):** geração de dados sintéticos para preservar privacidade (LGPD)
- **Random Forest:** classificação de risco cardiovascular
- **SHAP:** explicabilidade global para identificar variáveis clínicas dominantes
- **Z3 SMT Solver:** auditoria lógica formal para detectar violações nas predições

---

## Motivação

Modelos de IA com alta acurácia estatística podem violar axiomas médicos elementares em situações de borda — o que denominamos **falso negativo fatal**. Um modelo com F1-Score de 92% ainda pode classificar um paciente com isquemia severa como sem risco cardiovascular.

Este projeto demonstra que métricas estatísticas sozinhas são insuficientes para garantir segurança clínica, e que a camada de raciocínio simbólico formal via Z3 é indispensável para IA confiável em saúde.

---

## Pipeline

```
Dados Reais (Cleveland UCI)
        |
        v
    TVAE (SDV)
    Geração de 300 instâncias sintéticas
    Score de Qualidade: 78,9%
        |
        v
    Random Forest
    Treinado nos dados sintéticos
    Acurácia: 93,33% | F1: 92,00%
        |
        v
    SHAP
    Ranking global: oldpeak > thal > thalach > age
        |
        v
    Z3 SMT Solver
    Auditoria em lote: 296/300 consistentes (98,7%)
    4 violações fatais detectadas
```

---

## Resultados

### Qualidade dos Dados Sintéticos (TVAE)

| Métrica | Score |
|---|---|
| Score Geral de Qualidade (SDV) | 78,9% |
| Fidelidade Individual (Column Shapes) | 83,67% |
| Fidelidade Cruzada (Column Pair Trends) | 74,13% |
| Similaridade da Variável Alvo | 94,79% |

### Desempenho Comparativo do Classificador

| Métrica | Dados Reais | Dados Sintéticos | Diferença |
|---|---|---|---|
| Acurácia | 88,52% | 93,33% | +4,81% |
| Precisão | 83,87% | 88,46% | +4,59% |
| Recall | 92,86% | 95,83% | +2,97% |
| F1-Score | 88,14% | 92,00% | +3,86% |

### Auditoria Z3

| Resultado | Valor |
|---|---|
| Pacientes auditados | 300 |
| Predições consistentes | 296 (98,7%) |
| Violações lógicas detectadas | 4 (1,3%) |
| Axioma mais violado | R1: oldpeak > 2,0 |

### Violações Detectadas (Falsos Negativos Fatais)

| Paciente | Idade | Oldpeak | Thalach | Thal | Regra Violada |
|---|---|---|---|---|---|
| #177 | 61 anos | 2,0 | 174 | 3 | Idade + Oldpeak |
| #198 | 66 anos | 2,1 | 156 | 3 | Oldpeak Elevado |
| #262 | 58 anos | 3,5 | 169 | 3 | Oldpeak Elevado |
| #282 | 56 anos | 2,0 | 161 | 7 | Thal + Oldpeak |

---

## Axiomas Clínicos Formalizados (Z3)

```python
# R1 — Depressão ST severa indica isquemia ativa
Implies(oldpeak > 2.0, risk_prediction == 1)

# R2 — Defeito cardíaco com depressão ST moderada
Implies(And(Or(thal == 6, thal == 7), oldpeak > 1.0), risk_prediction == 1)

# R3 — Idoso com frequência cardíaca máxima baixa
Implies(And(age > 60, thalach < 120), risk_prediction == 1)

# R4 — Meia-idade com depressão ST moderada
Implies(And(age > 55, oldpeak > 1.5), risk_prediction == 1)
```

---

## Estrutura do Repositório

```
vaes-xai-cardiovascular/
├── DADOS_VAEs_XAI_FINAL.ipynb    # Notebook principal executado
├── apresentacao_vaes_xai.pptx    # Slides da apresentação
└── README.md                     # Este arquivo
```

---

## Como Executar

Abra o notebook `DADOS_VAEs_XAI_FINAL.ipynb` no Google Colab e execute as células em ordem.

As dependências são instaladas automaticamente na primeira célula:

```python
!pip install sdv shap z3-solver -q
```

O treinamento do TVAE pode levar entre 1 e 3 minutos no Colab CPU.

---

## Dependências

```
sdv >= 1.0
shap >= 0.42
z3-solver >= 4.12
scikit-learn >= 1.2
pandas >= 1.5
numpy >= 1.23
matplotlib >= 3.6
torch >= 2.0
```

---

## Reprodutibilidade

Todo o experimento usa `SEED = 42` propagado em numpy, torch e todos os estimadores sklearn.

---

## Dataset

**Cleveland Heart Disease Dataset** — UCI Machine Learning Repository  
303 pacientes | 13 variáveis clínicas | Target binarizado (0 = Sem Risco, 1 = Com Risco)  
Link: https://archive.ics.uci.edu/ml/datasets/heart+disease

---

## Autores

**Wesnei de Paiva Batista** — wesnei.paiva@ifce.edu.br  
**Thiago Alves** (Docente) — thiago.alves@ifce.edu.br

PPGCC — Instituto Federal do Ceará (IFCE) | 2026.1

---

## Licença

MIT License
