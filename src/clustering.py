import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def cluster_products(df):
    g = df.groupby("Product Name").agg(
        avg_lead_time=("Shipping Lead Time","mean"),
        lead_time_std=("Shipping Lead Time","std"),
        avg_units=("Units","mean"),
        avg_profit=("Gross Profit","mean"),
    ).fillna(0)
    if len(g) < 3:
        return g.assign(Cluster=0, Silhouette=None)
    X = StandardScaler().fit_transform(g)
    best_k, best_score = 2, -1
    for k in range(2, min(6, len(g)-1)+1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score = k, score
    g["Cluster"] = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X)
    g["Silhouette"] = best_score
    return g.reset_index()
