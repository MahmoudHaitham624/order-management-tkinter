"""
Project 2: Iris Flower Classification (KNN vs Decision Tree)

Goal: identify an iris flower's species from 4 measurements of its petals
and sepals.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    # 1. Load the classic Iris dataset (150 flowers, 3 species)
    iris = load_iris(as_frame=True)
    df = iris.frame
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
    print(df.head(), "\n")
    print("Flowers per species:\n", df["species"].value_counts(), "\n")

    X = iris.data
    y = iris.target

    # stratify=y keeps the same species mix in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 2. Find a good k for K-Nearest Neighbours
    print("KNN accuracy for different k:")
    k_values = range(1, 21)
    scores = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
        scores.append(accuracy_score(y_test, knn.predict(X_test)))
        print(f"  k={k:2d}: {scores[-1]:.3f}")
    best_k = k_values[scores.index(max(scores))]
    print(f"Best k = {best_k}\n")

    plt.figure()
    plt.plot(list(k_values), scores, marker="o")
    plt.xlabel("k (number of neighbours)")
    plt.ylabel("Test accuracy")
    plt.title("Choosing k for KNN")
    plt.savefig(OUTPUT_DIR / "knn_choose_k.png")

    # 3. Train both models
    models = {
        f"KNN (k={best_k})": KNeighborsClassifier(n_neighbors=best_k),
        "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(f"===== {name} =====")
        print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
        print(classification_report(y_test, preds,
                                    target_names=iris.target_names))

    # 4. A decision tree can be read as simple if/else rules
    print("Decision tree rules:")
    print(export_text(models["Decision Tree"],
                      feature_names=list(X.columns)))

    # 5. Confusion matrix: which species get mixed up?
    ConfusionMatrixDisplay.from_estimator(
        models["Decision Tree"], X_test, y_test,
        display_labels=iris.target_names, cmap="Blues",
    )
    plt.title("Decision Tree - confusion matrix")
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png")

    # 6. Classify a new flower
    new_flower = [[5.0, 3.4, 1.5, 0.2]]  # sepal len, sepal wid, petal len, petal wid
    new_df = pd.DataFrame(new_flower, columns=X.columns)
    species = iris.target_names[models["Decision Tree"].predict(new_df)[0]]
    print(f"New flower {new_flower[0]} is predicted to be: {species}")
    print(f"\nCharts saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
