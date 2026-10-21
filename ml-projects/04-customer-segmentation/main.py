# Project 4: Customer Segmentation
# We ask the computer to split shop customers into groups of similar people.
# We never tell it what the groups are. It must find them by itself.
# This is called "unsupervised learning".

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ---------- Step 1: Make some customer data ----------
# We secretly create 4 kinds of customers, 60 of each.
# np.random.normal(average, spread, how_many) makes numbers around an average.
np.random.seed(0)

# Group A: young, low income, spend a lot
age_a = np.random.normal(25, 5, 60)
income_a = np.random.normal(30, 8, 60)
spending_a = np.random.normal(75, 8, 60)

# Group B: middle-aged, high income, spend a lot
age_b = np.random.normal(45, 5, 60)
income_b = np.random.normal(85, 8, 60)
spending_b = np.random.normal(80, 8, 60)

# Group C: older, high income, spend very little
age_c = np.random.normal(50, 5, 60)
income_c = np.random.normal(90, 8, 60)
spending_c = np.random.normal(20, 8, 60)

# Group D: average in everything
age_d = np.random.normal(35, 5, 60)
income_d = np.random.normal(50, 8, 60)
spending_d = np.random.normal(45, 8, 60)

# Join the 4 groups together into one table of 240 customers
customers = pd.DataFrame({
    "age": np.concatenate([age_a, age_b, age_c, age_d]),
    "income": np.concatenate([income_a, income_b, income_c, income_d]),
    "spending": np.concatenate([spending_a, spending_b, spending_c, spending_d]),
})
print("The first 5 customers:")
print(customers.head())
print()


# ---------- Step 2: Scale the data ----------
# Age, income and spending use different units and ranges.
# Scaling puts every column on the same scale, so none of them counts more
# just because its numbers are bigger.
scaler = StandardScaler()
X = scaler.fit_transform(customers)


# ---------- Step 3: Find a good number of groups (the "elbow method") ----------
# For each number of groups, "inertia" says how spread out each group is.
# Smaller is better, but it always shrinks as we add groups.
# We look for the point where it stops dropping a lot: the "elbow".
group_counts = [1, 2, 3, 4, 5, 6, 7, 8]
inertias = []

for k in group_counts:
    test_model = KMeans(n_clusters=k, n_init=10, random_state=42)
    test_model.fit(X)
    inertias.append(test_model.inertia_)
    print("Groups:", k, " inertia:", round(test_model.inertia_, 1))

plt.plot(group_counts, inertias, marker="o")
plt.xlabel("Number of groups")
plt.ylabel("Inertia (lower is tighter)")
plt.title("Elbow method: look for the bend")
plt.show()


# ---------- Step 4: Create the final model with 4 groups ----------
# The elbow in the chart is at 4, so we use 4 groups.
model = KMeans(n_clusters=4, n_init=10, random_state=42)
model.fit(X)

# Add a new column to the table that says which group each customer is in
customers["group"] = model.labels_


# ---------- Step 5: Describe each group ----------
# The average age, income and spending for each group
print()
print("Average customer in each group:")
print(customers.groupby("group").mean().round(1))
print()
print("Number of customers in each group:")
print(customers["group"].value_counts())


# ---------- Step 6: Draw the groups ----------
# Each colour is one group the computer found.
plt.scatter(customers["income"], customers["spending"], c=customers["group"])
plt.xlabel("Income (thousands of $)")
plt.ylabel("Spending score")
plt.title("Customer groups found by K-Means")
plt.show()
