"""
Project 4: Customer Segmentation (K-Means Clustering)

Goal: group shop customers into segments by their age, income and spending,
WITHOUT any labels telling us the "right" groups.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def make_customers(seed=0):
    """Generate customers from 4 hidden groups (the model never sees the group)."""
    rng = np.random.default_rng(seed)
    groups = [
        # (count, age mean, income mean (k$), spending score mean)
        (60, 25, 30, 75),   # young, low income, big spenders
        (60, 45, 85, 80),   # wealthy, big spenders
        (60, 50, 90, 20),   # wealthy, careful savers
        (70, 35, 50, 45),   # average customers
    ]
    rows = []
    for count, age, income, score in groups:
        rows.append(np.column_stack([
            rng.normal(age, 5, count),
            rng.normal(income, 8, count),
            rng.normal(score, 8, count),
        ]))
    data = np.vstack(rows)
    df = pd.DataFrame(data, columns=["age", "annual_income_k", "spending_score"])
    df["spending_score"] = df["spending_score"].clip(1, 100)
    return df.round(0)


def main():
    df = make_customers()
    print(df.describe().round(1), "\n")

    # 1. Scale features so that no column dominates just because its numbers are bigger
    X = StandardScaler().fit_transform(df)

    # 2. Elbow method + silhouette score to choose the number of clusters
    k_range = range(2, 9)
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, km.labels_))
        print(f"k={k}: inertia={km.inertia_:7.1f}  silhouette={silhouettes[-1]:.3f}")
    best_k = k_range[int(np.argmax(silhouettes))]
    print(f"\nBest k by silhouette score: {best_k}\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(list(k_range), inertias, marker="o")
    ax1.set(title="Elbow method", xlabel="k", ylabel="Inertia (lower = tighter)")
    ax2.plot(list(k_range), silhouettes, marker="o", color="green")
    ax2.set(title="Silhouette score", xlabel="k", ylabel="Score (higher = better)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "choose_k.png")

    # 3. Final model
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    df["segment"] = kmeans.fit_predict(X)

    # 4. Describe each segment in business terms
    profile = df.groupby("segment").agg(
        customers=("age", "size"),
        avg_age=("age", "mean"),
        avg_income_k=("annual_income_k", "mean"),
        avg_spending=("spending_score", "mean"),
    ).round(1)
    print("Segment profiles:\n", profile, "\n")

    # 5. Visualise
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(df["annual_income_k"], df["spending_score"],
                          c=df["segment"], cmap="viridis", alpha=0.7)
    plt.xlabel("Annual income (k$)")
    plt.ylabel("Spending score (1-100)")
    plt.title("Customer segments")
    plt.legend(*scatter.legend_elements(), title="Segment")
    plt.savefig(OUTPUT_DIR / "segments.png")
    print(f"Charts saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
