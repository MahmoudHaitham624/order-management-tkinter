"""
Project 5: Handwritten Digit Recognition (Logistic Regression vs SVM)

Goal: recognise handwritten digits 0-9 from tiny 8x8 pixel images.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    # 1. Load 1,797 images of 8x8 pixels (each pixel is 0-16)
    digits = load_digits()
    print(f"Images: {digits.images.shape}, flattened features: {digits.data.shape}")
    print("The first image as numbers:\n", digits.images[0].astype(int), "\n")

    # Show some sample images
    fig, axes = plt.subplots(2, 5, figsize=(8, 4))
    for ax, img, label in zip(axes.ravel(), digits.images, digits.target):
        ax.imshow(img, cmap="gray_r")
        ax.set_title(f"Label: {label}")
        ax.axis("off")
    fig.savefig(OUTPUT_DIR / "sample_digits.png")

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42,
        stratify=digits.target,
    )

    # 3. Compare two models with 5-fold cross-validation
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000)),
        "SVM (RBF kernel)": make_pipeline(StandardScaler(), SVC(gamma="scale")),
    }
    for name, model in models.items():
        cv = cross_val_score(model, X_train, y_train, cv=5)
        print(f"{name:20s} cross-val accuracy: {cv.mean():.3f} (+/- {cv.std():.3f})")

    # 4. Train the best model and test it
    best = models["SVM (RBF kernel)"].fit(X_train, y_train)
    preds = best.predict(X_test)
    print(f"\nSVM test accuracy: {accuracy_score(y_test, preds):.3f}")

    ConfusionMatrixDisplay.from_predictions(y_test, preds, cmap="Greens")
    plt.title("Digits - confusion matrix")
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png")

    # 5. Look at the mistakes: that is where you learn the most
    wrong = np.where(preds != y_test)[0]
    print(f"Misclassified {len(wrong)} of {len(y_test)} test images")
    if len(wrong):
        n = min(len(wrong), 10)
        fig, axes = plt.subplots(1, n, figsize=(1.6 * n, 2.2), squeeze=False)
        for ax, i in zip(axes[0], wrong[:n]):
            ax.imshow(X_test[i].reshape(8, 8), cmap="gray_r")
            ax.set_title(f"true {y_test[i]}\npred {preds[i]}", fontsize=9)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "mistakes.png")
    print(f"Charts saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
