# Project 2: Iris Flower Classification
# We teach the computer to recognise 3 types of iris flower
# from 4 measurements of the flower's petals and sepals.

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay


# ---------- Step 1: Load the flower data ----------
# This dataset comes free with scikit-learn: 150 flowers, 3 species.
iris = load_iris()

X = iris.data    # the 4 measurements of each flower
y = iris.target  # the species of each flower, as a number: 0, 1 or 2

print("Measurements:", iris.feature_names)
print("Species:", iris.target_names)
print("Number of flowers:", len(X))
print("The first flower:", X[0], "is a", iris.target_names[y[0]])
print()


# ---------- Step 2: Split into training data and testing data ----------
# stratify=y makes sure all 3 species appear fairly in both parts.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


# ---------- Step 3: Model 1, K-Nearest Neighbours (KNN) ----------
# KNN idea: to classify a new flower, find the 5 most similar flowers
# it has seen, and pick the species most of them have.
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

knn_predictions = knn.predict(X_test)
print("KNN accuracy:", accuracy_score(y_test, knn_predictions))
print()


# ---------- Step 4: Try different numbers of neighbours ----------
print("Trying different numbers of neighbours (k):")
for k in [1, 3, 5, 7, 9, 15]:
    test_model = KNeighborsClassifier(n_neighbors=k)
    test_model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, test_model.predict(X_test))
    print("  k =", k, " accuracy =", round(accuracy, 3))
print()


# ---------- Step 5: Model 2, Decision Tree ----------
# A decision tree learns a list of yes/no questions,
# like "Is the petal shorter than 2.45 cm?"
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

tree_predictions = tree.predict(X_test)
print("Decision Tree accuracy:", accuracy_score(y_test, tree_predictions))
print()

# We can print the questions the tree learned.
# class 0 = setosa, class 1 = versicolor, class 2 = virginica
print("The rules the tree learned:")
print(export_text(tree, feature_names=iris.feature_names))


# ---------- Step 6: Classify a new flower ----------
# sepal length, sepal width, petal length, petal width (in cm)
new_flower = [[5.0, 3.4, 1.5, 0.2]]
answer = tree.predict(new_flower)
print("The new flower is a:", iris.target_names[answer[0]])


# ---------- Step 7: Draw a confusion matrix ----------
# Rows = the real species, columns = what the model guessed.
# Numbers on the diagonal are correct guesses. Anything else is a mistake.
ConfusionMatrixDisplay.from_predictions(
    y_test, tree_predictions, display_labels=iris.target_names
)
plt.title("Decision Tree: correct vs wrong guesses")
plt.show()
