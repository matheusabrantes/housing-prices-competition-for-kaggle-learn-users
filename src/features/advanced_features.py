import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler

def create_polynomial_features(df, degree=2):
    """Cria features polinomiais para variáveis numéricas importantes."""
    numeric_cols = ['GrLivArea', 'TotalBsmtSF', 'OverallQual', 'YearBuilt', 'GarageArea']
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(df[numeric_cols])
    poly_df = pd.DataFrame(poly_features, columns=[f'poly_{i}' for i in range(poly_features.shape[1])])
    return pd.concat([df, poly_df], axis=1)

def create_interaction_features(df):
    """Cria features de interação entre variáveis importantes."""
    df['Qual_GrLivArea'] = df['OverallQual'] * df['GrLivArea']
    df['Qual_YearBuilt'] = df['OverallQual'] * df['YearBuilt']
    df['Qual_TotalBsmtSF'] = df['OverallQual'] * df['TotalBsmtSF']
    df['YearBuilt_GrLivArea'] = df['YearBuilt'] * df['GrLivArea']
    return df

def create_cluster_features(df, n_clusters=5):
    """Cria features de clusterização baseadas em características similares."""
    cluster_cols = ['GrLivArea', 'TotalBsmtSF', 'OverallQual', 'YearBuilt', 'GarageArea']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[cluster_cols])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['HouseCluster'] = kmeans.fit_predict(scaled_data)
    return df

def transform_numeric_features(df):
    """Aplica transformações nas features numéricas."""
    # Log transform para features com distribuição assimétrica
    skewed_cols = ['GrLivArea', 'TotalBsmtSF', 'GarageArea', '1stFlrSF', '2ndFlrSF']
    for col in skewed_cols:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col])
    
    # Box-Cox transform para features que podem se beneficiar
    boxcox_cols = ['SalePrice'] if 'SalePrice' in df.columns else []
    for col in boxcox_cols:
        df[f'{col}_boxcox'], _ = boxcox(df[col] + 1)
    
    return df

def create_advanced_features(df):
    """Aplica todas as transformações de features avançadas."""
    df = create_polynomial_features(df)
    df = create_interaction_features(df)
    df = create_cluster_features(df)
    df = transform_numeric_features(df)
    return df
