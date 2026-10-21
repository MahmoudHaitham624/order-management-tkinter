# Project 5: Handwritten Digit Recognition

**Type:** Image classification (multi-class, 10 classes)
**Algorithms:** Logistic Regression, Support Vector Machine (SVM)

## What you'll learn
- How an **image becomes data**: an 8×8 picture becomes 64 numbers
- **Multi-class classification**: 10 possible answers instead of 2
- **Cross-validation**: a fairer way to compare models than a single split
- **SVMs**: a strong classic model for small datasets
- **Error analysis**: looking at the images the model got wrong

## The data
scikit-learn's built-in `digits` dataset: 1,797 handwritten digits, each 8×8
pixels with values from 0 to 16.

## Run it
```bash
python main.py
```
Output: cross-validation scores, test accuracy, `outputs/sample_digits.png`,
`outputs/confusion_matrix.png`, and `outputs/mistakes.png`.

## Try it yourself
1. Add `KNeighborsClassifier` and `RandomForestClassifier` to `models` and compare.
2. Tune the SVM's `C` value (0.1, 1, 10) with `GridSearchCV`.
3. Train on only 10% of the data. How much does accuracy drop?
4. Next step: try the 28×28 **MNIST** dataset with `fetch_openml("mnist_784")`
   or a small neural network (`MLPClassifier`).
