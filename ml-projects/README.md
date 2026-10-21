# 5 Machine Learning Projects for Beginners

Five small projects, each teaching one big ML idea. The code is written to be
as easy to read as possible: it runs from top to bottom, with a comment above
every step.

| # | Project | What the computer learns to do |
|---|---------|--------------------------------|
| 1 | [House Price Prediction](01-house-price-prediction) | Guess a **number** (a price) |
| 2 | [Iris Flower Classification](02-iris-flower-classification) | Pick a **category** (a flower type) |
| 3 | [Spam Message Detector](03-spam-message-detector) | Read **text** and decide: spam or not |
| 4 | [Customer Segmentation](04-customer-segmentation) | Find **groups** by itself, with no answers given |
| 5 | [Handwritten Digit Recognition](05-handwritten-digit-recognition) | Read **pictures** of numbers |

Do them in order. Each one builds on the one before.

## Setup

```bash
cd ml-projects
pip install -r requirements.txt
```

## Run a project

```bash
python 01-house-price-prediction/main.py
```

Results print in the terminal. Charts open in a window, and the script
continues once you **close the chart window**.

## The recipe almost every ML project follows

Every project here follows these same steps. Once you know them, ML code gets
much easier to read.

1. **Get data**: a table of examples.
2. **Split X and y**: `X` is the information we give the model, and `y` is the answer we want it to predict.
3. **Split train and test**: the model learns from the *train* part. We keep the *test* part secret so we can check the model afterwards, like exam questions.
4. **Create a model**: `model = SomeModel()`
5. **Train it**: `model.fit(X_train, y_train)`
6. **Test it**: `model.predict(X_test)`, then compare the guesses with the real answers.
7. **Use it**: predict something brand new.

## Word list

| Word | Meaning |
|------|---------|
| **Model** | The "brain" that learns patterns from data |
| **Feature** | One piece of input information (for example, house size) |
| **Target / label** | The answer we want to predict (for example, the price) |
| **Training** | Letting the model learn from examples (`fit`) |
| **Prediction** | The model's guess (`predict`) |
| **Accuracy** | The share of guesses that are correct (0.95 = 95%) |
| **Regression** | Predicting a number |
| **Classification** | Predicting a category |
| **Clustering** | Finding groups without being given any answers |
| **Overfitting** | Memorising the training data instead of learning the real pattern |
