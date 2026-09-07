# Triagem Automática de Laudos Médicos

Tech Challenge Fase 3 (POSTECH) — sistema de triagem automática de exames de texto
(laudos médicos) que classifica a urgência do caso em **normal / atenção / urgente**,
servido via API REST (FastAPI) em container Docker.

Especificação completa do desafio em [`MLET - Tech Challenge Fase 3.pdf`](MLET%20-%20Tech%20Challenge%20Fase%203.pdf).

Este README cobre as **Etapas 1 e 2** do desafio (as demais etapas — monitoramento e
otimização de latência — ainda não foram implementadas neste repositório).

## Etapa 1 — Decisão Arquitetural e API Inicial

### Estratégia de deploy em nuvem: batch vs. real-time

O cenário é a triagem de laudos médicos em um hospital: os textos chegam **um a um**,
conforme os exames são concluídos, e a equipe clínica precisa saber a urgência
**imediatamente** para priorizar o atendimento. Isso descarta uma arquitetura batch
(ex.: processar laudos acumulados a cada N minutos/horas) — o valor do sistema está
justamente em reduzir o tempo entre "laudo pronto" e "urgência conhecida". Por isso a
arquitetura escolhida é **real-time**, servida como uma API REST síncrona.

### Arquitetura proposta na AWS

```
Cliente (sistema hospitalar)
        │  HTTPS
        ▼
  Application Load Balancer
        │
        ▼
  ECS Fargate (serviço da API FastAPI, containers stateless)
        │
        ▼
  Container: uvicorn + FastAPI + modelo carregado em memória
```

- **Amazon ECR**: armazena a imagem Docker da API (a mesma imagem construída neste
  repositório). O pipeline de CI/CD (Etapa 2) faz build + push para o ECR a cada
  merge na branch principal.
- **Amazon ECS (Fargate)**: roda a API sem gerenciar servidores/instâncias EC2
  diretamente. Fargate foi escolhido em vez de EC2 porque a carga é imprevisível
  (picos de exames), e Fargate escala o número de tasks conforme a demanda sem
  necessidade de provisionar capacidade fixa. Para um cenário de produção real com
  carga mais previsível/alta, uma migração para EKS ou EC2 com auto scaling seria
  uma otimização de custo a considerar.
- **Application Load Balancer**: distribui requisições entre as tasks do ECS e
  expõe HTTPS para o cliente.
- **Orquestração de retreino (Airflow)**: fora do caminho crítico de latência da
  API. Proposta: **Amazon MWAA** (Managed Workflows for Apache Airflow) rodando a
  DAG de treino/retreino (Etapa 2) periodicamente ou disparada por evento (ex.:
  chegada de novos dados rotulados em um bucket S3). Alternativa mais barata para
  ambientes de estudo/baixo orçamento: Airflow self-hosted em uma única instância
  EC2 ou task ECS, já que o volume de retreino aqui é baixo.
- **Monitoramento** (Etapa 3, ainda não implementada): Prometheus + Grafana rodando
  localmente via Docker Compose para desenvolvimento; em produção na AWS, o
  equivalente seria Amazon Managed Service for Prometheus + Amazon Managed Grafana,
  ou CloudWatch Container Insights como alternativa nativa.

### Por que essa escolha

- Real-time via API é o único jeito de atender ao requisito clínico de resposta
  imediata por laudo.
- ECS Fargate remove a necessidade de gerenciar servidores e escala
  automaticamente com a demanda de triagem, sem exigir uma equipe de
  infraestrutura dedicada — adequado ao escopo de um hospital que quer focar no
  modelo, não na operação de servidores.
- Separar a API (caminho de latência crítica) do Airflow (orquestração de
  retreino, fora do caminho crítico) evita que jobs de treino pesados
  compitam por recursos com as requisições de triagem em produção.

## Dataset e mapeamento de urgência

Dataset: [Medical Abstracts TC Corpus](https://github.com/sebischair/Medical-Abstracts-TC-Corpus)
(11.550 amostras de treino + 2.888 de teste). O dataset original classifica a
**categoria da doença** (`condition_label`, 1 a 5), não a urgência clínica. Como
simplificação didática para este exercício, mapeamos cada categoria para um nível
de urgência:

| `condition_label` | Categoria original | Urgência mapeada |
|---|---|---|
| 4 | Cardiovascular diseases | **urgente** |
| 1 | Neoplasms | **atenção** |
| 3 | Nervous system diseases | **atenção** |
| 2 | Digestive system diseases | **normal** |
| 5 | General pathological conditions | **normal** |

**Importante:** esse mapeamento é uma aproximação para fins do desafio, não uma
classificação de urgência clinicamente validada — em um cenário real, o
mapeamento deveria ser definido por profissionais de saúde a partir de dados
rotulados por urgência real.

## Modelo

Pipeline scikit-learn: `TfidfVectorizer` (uni+bigramas, stopwords em inglês,
20k features) + `LogisticRegression`. Escolhido por ser leve e rápido — favorece
a latência da API e será a base para a otimização (ex.: exportação para ONNX) na
Etapa 4. Acurácia no conjunto de teste: **~61%** (esperado dado que o rótulo de
urgência é uma aproximação a partir de categoria de doença, não urgência real).

## Estrutura do projeto

```
├── src/
│   ├── ml/            # carregamento de dados, treino e predição do modelo
│   └── api/            # API FastAPI (schemas + endpoints)
├── scripts/
│   ├── download_data.py    # baixa o dataset
│   └── measure_latency.py  # mede latência do endpoint /predict
├── tests/               # testes automatizados (Etapa 2)
├── .github/workflows/   # pipeline CI/CD (Etapa 2)
├── airflow/dags/        # DAG de treino/retreino (Etapa 2)
├── data/                 # dataset baixado (não versionado)
├── models/               # modelo treinado (não versionado)
└── Dockerfile
```

## Como executar

### 1. Ambiente local (sem Docker)

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows (Git Bash); no Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py   # baixa o dataset para data/
python -m src.ml.train             # treina o modelo e salva em models/

uvicorn src.api.main:app --reload --port 8000
```

### 2. Docker

```bash
docker build -t triagem-laudos:latest .
docker run -p 8000:8000 triagem-laudos:latest
```

> O modelo (`models/urgency_classifier.joblib`) precisa existir antes do build —
> rode `python -m src.ml.train` localmente primeiro (ele não é versionado no git).

### Testando a API

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient presents with acute chest pain and shortness of breath."}'
```

### Medindo latência

```bash
python scripts/measure_latency.py --url http://localhost:8000 --n 200
```

## Baseline de latência (Etapa 1)

Medido localmente contra o container Docker, 200 requisições ao endpoint
`/predict` (após warm-up):

| Métrica | Valor |
|---|---|
| Média | 10,27 ms |
| p50 | 6,25 ms |
| p95 | 29,15 ms |
| p99 | 30,47 ms |
| min / max | 4,98 ms / 30,69 ms |

Esse baseline será comparado com o modelo otimizado (ex.: ONNX Runtime) na
Etapa 4.

## Etapa 2 — CI/CD e Pipeline Automatizado

### Testes automatizados

`tests/test_data.py` valida o mapeamento de urgência e o carregamento do CSV
(usando um CSV temporário, sem depender do dataset completo). `tests/test_api.py`
valida os endpoints `/health` e `/predict` com um modelo minúsculo treinado em
memória durante o teste — assim a suíte roda rápido e não depende de baixar o
dataset nem de ter um modelo já treinado.

```bash
pip install -r requirements-dev.txt
pytest -v
ruff check .
```

### GitHub Actions

Workflow em [`.github/workflows/ci.yml`](.github/workflows/ci.yml), disparado em
todo `push`/`pull_request`, com dois jobs independentes:

- **Lint**: `ruff check .`
- **Testes**: `pytest -v`

### DAG do Airflow

DAG em [`airflow/dags/train_dag.py`](airflow/dags/train_dag.py) com duas tasks
encadeadas:

1. `ingest_data` — baixa/atualiza o dataset (`scripts/download_data.py`).
2. `train_and_save_model` — treina o pipeline e salva o modelo (`src/ml/train.py`).

Ambas as tasks usam `do_xcom_push=False`: nenhuma delas precisa repassar dado
para a próxima via XCom, e o retorno de `train()` é um objeto `Pipeline` do
scikit-learn, que não é serializável em JSON (o Airflow tentaria publicá-lo como
XCom por padrão e a task falharia).

Como este repositório não roda um Airflow local, a DAG foi validada com a
imagem oficial do Airflow via Docker (útil para o desenvolvedor reproduzir):

```bash
docker run --rm -v "$(pwd):/opt/airflow/project" \
  -e AIRFLOW_HOME=/tmp/airflow_home \
  -e AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/project/airflow/dags \
  -e AIRFLOW__CORE__LOAD_EXAMPLES=False \
  apache/airflow:2.10.4-python3.11 bash -c "
    pip install -q pandas scikit-learn joblib requests &&
    airflow db migrate &&
    cd /opt/airflow/project &&
    airflow dags test triagem_laudos_train_dag 2026-01-01
  "
```

Resultado obtido: `DagRun ... state=success` — as duas tasks rodaram e o
modelo foi treinado e salvo com sucesso dentro do container.
