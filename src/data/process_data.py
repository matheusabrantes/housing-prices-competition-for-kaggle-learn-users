import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from ..features.advanced_features import create_advanced_features

def load_data():
    """Carrega os dados de treino e teste."""
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train, test

def handle_missing_values(df):
    """Trata valores ausentes no dataset."""
    # Identifica colunas numéricas e categóricas
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Imputa valores ausentes em colunas numéricas com a mediana
    numeric_imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
    
    # Imputa valores ausentes em colunas categóricas com 'Missing'
    categorical_imputer = SimpleImputer(strategy='constant', fill_value='Missing')
    df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
    
    return df

def encode_categorical_features(X_train, X_test):
    """Codifica features categóricas usando LabelEncoder."""
    # Identifica colunas categóricas
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    
    # Concatena os dados para garantir que todos os valores categóricos sejam considerados
    combined_data = pd.concat([X_train[categorical_cols], X_test[categorical_cols]], axis=0)
    
    # Cria e treina os encoders
    encoders = {}
    for col in categorical_cols:
        encoders[col] = LabelEncoder()
        encoders[col].fit(combined_data[col].astype(str))
    
    # Aplica a codificação apenas nas colunas categóricas
    for col in categorical_cols:
        X_train[col] = encoders[col].transform(X_train[col].astype(str))
        X_test[col] = encoders[col].transform(X_test[col].astype(str))
    
    return X_train, X_test

def create_features(df):
    """Cria novas features a partir das existentes."""
    # Área total do porão
    df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(0)
    
    # Área total da casa
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'].fillna(0)
    
    # Idade da casa
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    
    # Idade da garagem
    df['GarageAge'] = df['YrSold'] - df['GarageYrBlt']
    df['GarageAge'] = df['GarageAge'].fillna(0)
    
    # Total de banheiros
    df['TotalBath'] = df['FullBath'] + (0.5 * df['HalfBath'].fillna(0))
    
    # Qualidade geral da casa
    df['OverallQual'] = df['OverallQual'].fillna(df['OverallQual'].median())
    
    # Condição geral da casa
    df['OverallCond'] = df['OverallCond'].fillna(df['OverallCond'].median())
    
    return df

def prepare_data():
    """Prepara os dados para treinamento."""
    # Carrega os dados
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    
    # Separa features e target
    X_train = train_data.drop(['Id', 'SalePrice'], axis=1)
    y_train = train_data['SalePrice']
    X_test = test_data.drop(['Id'], axis=1)
    test_ids = test_data['Id']
    
    # Processa os dados
    X_train, X_test = process_features(X_train, X_test)
    
    return X_train, y_train, X_test, test_ids

def process_features(X_train, X_test):
    """Processa as features dos dados de treino e teste."""
    # Identifica colunas numéricas e categóricas
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    
    # Preenche valores faltantes
    X_train, X_test = handle_missing_values(X_train, X_test, numeric_cols, categorical_cols)
    
    # Cria features básicas
    X_train, X_test = create_basic_features(X_train, X_test)
    
    # Codifica features categóricas
    X_train, X_test = encode_categorical_features(X_train, X_test)
    
    # Cria features avançadas
    X_train = create_advanced_features(X_train)
    X_test = create_advanced_features(X_test)
    
    return X_train, X_test

def handle_missing_values(X_train, X_test, numeric_cols, categorical_cols):
    """Preenche valores faltantes nas features numéricas e categóricas."""
    # Preenche valores faltantes numéricos com a mediana
    numeric_imputer = SimpleImputer(strategy='median')
    X_train[numeric_cols] = numeric_imputer.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = numeric_imputer.transform(X_test[numeric_cols])
    
    # Preenche valores faltantes categóricos com 'Missing'
    categorical_imputer = SimpleImputer(strategy='constant', fill_value='Missing')
    X_train[categorical_cols] = categorical_imputer.fit_transform(X_train[categorical_cols])
    X_test[categorical_cols] = categorical_imputer.transform(X_test[categorical_cols])
    
    return X_train, X_test

def create_basic_features(X_train, X_test):
    """Cria features básicas para os dados."""
    for df in [X_train, X_test]:
        # Área total da casa
        df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF'].fillna(0) + df['TotalBsmtSF'].fillna(0)
        
        # Total de banheiros
        df['TotalBath'] = df['FullBath'] + (0.5 * df['HalfBath']) + \
                         df['BsmtFullBath'].fillna(0) + (0.5 * df['BsmtHalfBath'].fillna(0))
        
        # Idade da casa
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        
        # Anos desde a última reforma
        df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
        
        # Qualidade geral da casa (média de OverallQual e OverallCond)
        df['OverallScore'] = (df['OverallQual'] + df['OverallCond']) / 2
        
        # Área do porão por área total
        df['BsmtSF_Ratio'] = df['TotalBsmtSF'] / df['TotalSF']
        
        # Área da garagem por área total
        df['GarageSF_Ratio'] = df['GarageArea'] / df['TotalSF']
    
    return X_train, X_test

if __name__ == "__main__":
    X_train, y_train, X_test, test_ids = prepare_data()
    print("Dados processados com sucesso!")
    print(f"Shape do conjunto de treino: {X_train.shape}")
    print(f"Shape do conjunto de teste: {X_test.shape}")
