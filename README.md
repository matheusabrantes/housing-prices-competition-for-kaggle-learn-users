# Housing Prices Competition - Kaggle

Este projeto contém a solução para a competição "Housing Prices Competition for Kaggle Learn Users" do Kaggle.

## Estrutura do Projeto

```
.
├── data/                   # Dados brutos
├── notebooks/             # Jupyter notebooks para análise exploratória
├── src/
│   ├── data/             # Scripts para processamento de dados
│   ├── features/         # Scripts para engenharia de features
│   ├── models/           # Scripts para treinamento e avaliação de modelos
│   └── utils/            # Funções utilitárias
└── pyproject.toml        # Configuração do Poetry
```

## Configuração do Ambiente

1. Instale o Poetry (se ainda não tiver instalado):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone o repositório:
```bash
git clone [URL_DO_REPOSITÓRIO]
cd housing-prices-competition
```

3. Instale as dependências:
```bash
poetry install
```

4. Ative o ambiente virtual:
```bash
poetry shell
```

## Modelos Implementados

- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Stacking (combinação dos modelos acima)

## Métricas de Avaliação

- RMSE (Root Mean Square Error)
- R² Score

## Como Executar

1. Processamento dos dados:
```bash
python src/data/process_data.py
```

2. Treinamento dos modelos:
```bash
python src/models/train_models.py
```

3. Geração de previsões:
```bash
python src/models/predict.py
``` 