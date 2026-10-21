# 5 Machine Learning Projects for Beginners

Five small projects, each teaching a different core ML idea. They use only
scikit-learn's built-in datasets or generated data, so they run offline.

| # | Project | ML Type | Algorithm(s) | Key Concepts |
|---|---------|---------|--------------|--------------|
| 1 | [House Price Prediction](01-house-price-prediction) | Regression | Linear Regression | train/test split, MAE, RMSE, R² |
| 2 | [Iris Flower Classification](02-iris-flower-classification) | Classification | KNN, Decision Tree | accuracy, confusion matrix, choosing *k* |
| 3 | [Spam Message Detector](03-spam-message-detector) | Text Classification | Naive Bayes | bag-of-words, TF-IDF, precision/recall |
| 4 | [Customer Segmentation](04-customer-segmentation) | Clustering (unsupervised) | K-Means | scaling, elbow method, cluster profiles |
| 5 | [Handwritten Digit Recognition](05-handwritten-digit-recognition) | Image Classification | Logistic Regression, SVM | pixels as features, model comparison |

## Setup

```bash
cd ml-projects
pip install -r requirements.txt
```

## Run a project

```bash
python 01-house-price-prediction/main.py
```

Each script prints its results to the terminal and saves its charts in an
`outputs/` folder inside the project.

## Suggested learning order

1. **Project 1**: the basic workflow: load data → split → train → evaluate.
2. **Project 2**: classification and comparing two models.
3. **Project 3**: turning text into numbers.
4. **Project 4**: learning without labels.
5. **Project 5**: images as data, and putting it all together.

Each project's README ends with **"Try it yourself"** exercises. Do them; they
are where most of the learning happens.
