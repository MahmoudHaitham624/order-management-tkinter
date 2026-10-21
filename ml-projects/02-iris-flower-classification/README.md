# Project 2: Iris Flower Classification

**Type:** Classification (predicting a category)
**Algorithms:** K-Nearest Neighbours (KNN), Decision Tree

## What you'll learn
- The difference between **classification** and regression
- **KNN**: "you are what your closest neighbours are"
- **Decision Trees**: a model you can read as if/else rules
- **Hyperparameter tuning**: trying different `k` values to find the best one
- Metrics: **accuracy**, **precision**, **recall**, and the **confusion matrix**
- Why `stratify` matters when splitting

## The data
The famous Iris dataset: 150 flowers, 3 species (setosa, versicolor,
virginica), and 4 measurements (sepal length/width, petal length/width). It ships with scikit-learn.

## Run it
```bash
python main.py
```
Output: accuracy for k = 1 to 20, a report for each model, readable tree rules,
`outputs/knn_choose_k.png`, and `outputs/confusion_matrix.png`.

## Try it yourself
1. Change the tree's `max_depth` to 1, then to 10. Which depth overfits?
2. Use only the 2 petal features. Is accuracy still good?
3. Make a scatter plot of petal length vs petal width, coloured by species.
4. Add a third model, `LogisticRegression`, and compare all three.
