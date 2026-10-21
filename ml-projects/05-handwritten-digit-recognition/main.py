# Project 5: Handwritten Digit Recognition
# We teach the computer to read handwritten numbers from 0 to 9.

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay


# ---------- Step 1: Load the images ----------
# 1,797 small pictures of handwritten digits. Each picture is 8 x 8 pixels.
# Each pixel is a number from 0 (white) to 16 (black).
digits = load_digits()

print("Number of pictures:", len(digits.images))
print()
print("This is what the first picture looks like as numbers:")
print(digits.images[0])
print("It is the digit:", digits.target[0])
print()

# Show the first picture as an image
plt.imshow(digits.images[0], cmap="gray_r")
plt.title("This is a " + str(digits.target[0]))
plt.show()


# ---------- Step 2: Get the inputs and the answers ----------
# digits.data is each 8 x 8 picture flattened into one row of 64 numbers.
X = digits.data
y = digits.target  # the real digit (0 to 9) for each picture


# ---------- Step 3: Split into training data and testing data ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ---------- Step 4: Model 1, Logistic Regression ----------
# max_iter gives the model enough steps to finish learning
model1 = LogisticRegression(max_iter=5000)
model1.fit(X_train, y_train)
predictions1 = model1.predict(X_test)
print("Logistic Regression accuracy:", round(accuracy_score(y_test, predictions1), 3))


# ---------- Step 5: Model 2, Support Vector Machine (SVM) ----------
model2 = SVC()
model2.fit(X_train, y_train)
predictions2 = model2.predict(X_test)
print("SVM accuracy:", round(accuracy_score(y_test, predictions2), 3))
print()


# ---------- Step 6: Look at the mistakes ----------
# Go through every test picture and print the ones the SVM got wrong
print("Pictures the SVM got wrong:")
mistakes = 0
for i in range(len(y_test)):
    if predictions2[i] != y_test[i]:
        print("  Real digit:", y_test[i], " SVM guessed:", predictions2[i])
        mistakes = mistakes + 1
print("Total mistakes:", mistakes, "out of", len(y_test))


# ---------- Step 7: Draw a confusion matrix ----------
# This shows which digits get confused with each other
ConfusionMatrixDisplay.from_predictions(y_test, predictions2)
plt.title("SVM: real digit vs guessed digit")
plt.show()
