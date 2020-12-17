import pandas as pd
from sklearn.decomposition import PCA


def fit(df):
    pca = PCA(n_components=3)
    pca.fit(df)
    loadings = (
        pd.DataFrame(index=df.columns)
        .assign(Level=pca.components_[0, :])
        .assign(Slope=pca.components_[1, :])
        .assign(Curvature=pca.components_[2, :])
    )
    factors = (
        pd.DataFrame()
        .assign(Level=(df * loadings.Level).sum(axis=1))
        .assign(Slope=(df * loadings.Slope).sum(axis=1))
        .assign(Curvature=(df * loadings.Curvature).sum(axis=1))
    )
    factors = factors.cumsum(axis=0)
    return pca, loadings, factors
