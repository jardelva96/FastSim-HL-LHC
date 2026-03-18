# FastSim HL-LHC

Repositorio de demonstracao tecnica para candidatura da bolsa TT-IV-A (FAPESP)
do projeto "Desafios de Computacao para a Fase II da Atualizacao do CMS"
(processo 2022/02950-5).

## O que este projeto entrega

- simulacao rapida sintetica de eventos com detector em grafo;
- treino de dois modelos (`graph_cvae` e baseline `mlp_ae`);
- avaliacao com metricas de reconstrucao e vies energetico;
- validacao fisica (perfil longitudinal, closure e erro por pileup);
- geracao condicionada de eventos sinteticos;
- interface web (Streamlit) para treino, validacao e benchmark;
- benchmark automatico e geracao de pacote para inscricao.

## Melhorias implementadas

- **Geracao de dados vetorizada** para acelerar criacao de eventos;
- **Geometria configuravel** (`n_layers`, `cells_per_layer`);
- **Treino mais robusto** com early stopping, scheduler e grad clipping;
- **Metricas mais completas** (`mse`, vies/resolucao de energia, `time_mae`);
- **Benchmark comparativo** (`graph_cvae` vs `mlp_ae`) para evidenciar criterio tecnico;
- **Pacote de candidatura** gerado automaticamente com resumo + rascunho de email;
- **Artefatos de experimento** (`history.json`, `train_config.json`, `train_summary.json`).

## Estrutura

```
.
|-- src/fastsim_tt4a/
|   |-- data.py
|   |-- model.py
|   |-- metrics.py
|   |-- train.py
|   |-- evaluate.py
|   |-- validate.py
|   |-- analysis.py
|   |-- benchmark.py
|   |-- submission.py
|   `-- dashboard.py
|-- tests/
|   |-- test_data.py
|   |-- test_model.py
|   `-- test_metrics.py
|-- .github/workflows/ci.yml
|-- pyproject.toml
`-- README.md
```

## Setup rapido (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev,ui]
```

## Treino (CLI)

```powershell
fastsim-train --epochs 20 --num-events 6000 --batch-size 64 --n-layers 6 --cells-per-layer 16
fastsim-train --model-type mlp_ae --epochs 20 --num-events 6000 --out-dir artifacts/mlp
```

Saidas em `artifacts/`:

- `model.pt`
- `history.json`
- `train_config.json`
- `train_summary.json`

## Avaliacao (CLI)

```powershell
fastsim-eval --checkpoint artifacts/model.pt --num-events 1000 --out-json artifacts/eval.json
```

## Validacao fisica detalhada (CLI)

```powershell
fastsim-validate --checkpoint artifacts/model.pt --num-events 1200 --out-json artifacts/validation.json
```

Saida inclui:

- perfil por camada (energia true/pred/erro);
- perfil por faixa de pileup (bias/resolucao);
- closure energetico e mapas medios de energia/tempo.

## Interface (Streamlit)

```powershell
fastsim-dashboard
```

Na interface:

- aba **Treino**: configura hiperparametros e roda treino;
- aba **Historico**: carrega e plota `history.json`;
- aba **Evento**: inspeciona energia/tempo verdadeiros e reconstruidos.
- aba **Validacao Fisica**: metricas e mapas detalhados;
- aba **Geracao**: amostragem condicionada por energia/pileup;
- aba **Benchmark**: comparacao entre modelos;
- aba **Selecao**: gera pacote final da candidatura.

## Benchmark para selecao

```powershell
fastsim-benchmark --out-dir artifacts/benchmark
```

Saidas:

- `artifacts/benchmark/benchmark_results.json`
- `artifacts/benchmark/benchmark_report.md`

## Gerar pacote para enviar

```powershell
fastsim-submission --candidate-name "Seu Nome" --eval-json artifacts/eval.json --benchmark-json artifacts/benchmark/benchmark_results.json
```

Saida:

- `artifacts/application_packet.md` (resumo tecnico + rascunho de e-mail)

## Testes e qualidade

```powershell
ruff check src tests
pytest
```

## Como usar na candidatura

- destaque o fluxo ponta a ponta (dados -> treino -> avaliacao -> interface);
- mostre comparacao entre `graph_cvae` e `mlp_ae` (isso sinaliza metodo cientifico);
- inclua no email os principais numeros de `train_summary.json` e `eval.json`;
- inclua resultado de `validation.json` para evidenciar leitura fisica do modelo;
- mostre que o codigo esta pronto para trocar dados sinteticos por dados reais.
