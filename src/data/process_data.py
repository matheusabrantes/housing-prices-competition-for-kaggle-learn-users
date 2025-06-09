import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

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
    """Codifica features categóricas usando Label Encoding, garantindo que todas as categorias estejam presentes."""
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Junta treino e teste para garantir todas as categorias
        all_values = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
        le.fit(all_values)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le
    return X_train, X_test, label_encoders

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
    train, test = load_data()
    
    # Separa features e target
    y_train = train['SalePrice']
    X_train = train.drop(['Id', 'SalePrice'], axis=1)
    X_test = test.drop(['Id'], axis=1)
    
    # Processa os dados
    X_train = handle_missing_values(X_train)
    X_test = handle_missing_values(X_test)
    
    X_train = create_features(X_train)
    X_test = create_features(X_test)
    
    X_train, X_test, label_encoders = encode_categorical_features(X_train, X_test)
    
    return X_train, y_train, X_test, test['Id']

if __name__ == "__main__":
    X_train, y_train, X_test, test_ids = prepare_data()
    print("Dados processados com sucesso!")
    print(f"Shape do conjunto de treino: {X_train.shape}")
    print(f"Shape do conjunto de teste: {X_test.shape}") 