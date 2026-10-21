# Project 1: House Price Prediction
# We teach the computer to guess a house's price
# from its size, number of bedrooms, age, and distance to the city.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# ---------- Step 1: Make some house data ----------
# We create 500 pretend houses using random numbers.
np.random.seed(42)  # makes the "random" numbers the same every time you run

size = np.random.uniform(50, 300, 500)      # size in square metres (50 to 300)
bedrooms = np.random.randint(1, 6, 500)     # 1 to 5 bedrooms
age = np.random.uniform(0, 50, 500)         # age in years (0 to 50)
distance = np.random.uniform(1, 30, 500)    # km from the city centre (1 to 30)

# This is the "real" price rule. The computer does NOT see this rule.
# Its job is to discover it just by looking at the data.
luck = np.random.normal(0, 25000, 500)      # some houses sell a bit higher or lower
price = 50000 + 1500 * size + 10000 * bedrooms - 1000 * age - 3000 * distance + luck

# Put everything in a table (a "DataFrame")
houses = pd.DataFrame({
    "size": size,
    "bedrooms": bedrooms,
    "age": age,
    "distance": distance,
    "price": price,
})
print("The first 5 houses:")
print(houses.head())
print()


# ---------- Step 2: Separate the inputs from the answer ----------
# X = the information we give the model (the "features")
# y = the answer we want it to predict (the "target")
X = houses[["size", "bedrooms", "age", "distance"]]
y = houses["price"]


# ---------- Step 3: Split into training data and testing data ----------
# 80% of houses are used to learn. 20% are hidden away to test the model later,
# just like a teacher keeps exam questions secret.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Houses to learn from:", len(X_train))
print("Houses to test on:", len(X_test))
print()


# ---------- Step 4: Create the model and train it ----------
model = LinearRegression()
model.fit(X_train, y_train)  # "fit" means "learn from this data"


# ---------- Step 5: See what the model learned ----------
# Compare these with the real rule in Step 1. They should be close!
print("What the model learned:")
print("  Each extra m2 adds:       $", round(model.coef_[0]))
print("  Each extra bedroom adds:  $", round(model.coef_[1]))
print("  Each extra year of age:   $", round(model.coef_[2]))
print("  Each extra km from city:  $", round(model.coef_[3]))
print()


# ---------- Step 6: Test the model on the hidden houses ----------
predictions = model.predict(X_test)

error = mean_absolute_error(y_test, predictions)  # average size of the mistakes
score = r2_score(y_test, predictions)             # 1.0 means perfect

print("On average, the guess is off by: $", round(error))
print("R2 score (1.0 is perfect):", round(score, 3))
print()


# ---------- Step 7: Guess the price of a brand-new house ----------
new_house = pd.DataFrame({
    "size": [120],
    "bedrooms": [3],
    "age": [10],
    "distance": [5],
})
guess = model.predict(new_house)
print("Predicted price for the new house: $", round(guess[0]))


# ---------- Step 8: Draw a chart ----------
# Each dot is one test house. If the model were perfect,
# every dot would sit exactly on a straight diagonal line.
plt.scatter(y_test, predictions)
plt.xlabel("Real price")
plt.ylabel("Predicted price")
plt.title("Real vs predicted house prices")
plt.show()
