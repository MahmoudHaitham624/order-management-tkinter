"""
Project 1: House Price Prediction (Linear Regression)

Goal: predict a house's price from its size, number of bedrooms, age and
distance to the city centre.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save charts to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def make_house_data(n=500, seed=42):
    """Generate a realistic-looking housing dataset."""
    rng = np.random.default_rng(seed)
    size_m2 = rng.uniform(50, 300, n)
    bedrooms = rng.integers(1, 6, n)
    age_years = rng.uniform(0, 50, n)
    distance_km = rng.uniform(1, 30, n)

    # The "true" pricing rule, plus random noise
    price = (
        50_000
        + 1_500 * size_m2
        + 10_000 * bedrooms
        - 1_000 * age_years
        - 3_000 * distance_km
        + rng.normal(0, 25_000, n)
    )
    return pd.DataFrame({
        "size_m2": size_m2.round(1),
        "bedrooms": bedrooms,
        "age_years": age_years.round(1),
        "distance_km": distance_km.round(1),
        "price": price.round(0),
    })


def main():
    # 1. Load data
    df = make_house_data()
    print("First 5 rows:\n", df.head(), "\n")
    print("Summary statistics:\n", df.describe().round(1), "\n")

    # 2. Separate features (X) from target (y)
    X = df.drop(columns="price")
    y = df["price"]

    # 3. Split: 80% for training, 20% kept aside for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("What the model learned (price change per +1 unit):")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"  {feature:12s}: {coef:>10,.0f}")
    print(f"  {'intercept':12s}: {model.intercept_:>10,.0f}\n")

    # 5. Evaluate on unseen test data
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    print(f"MAE  (average error): ${mae:,.0f}")
    print(f"RMSE (punishes big errors): ${rmse:,.0f}")
    print(f"R²   (1.0 = perfect): {r2:.3f}\n")

    # 6. Predict a brand-new house
    new_house = pd.DataFrame([{
        "size_m2": 120, "bedrooms": 3, "age_years": 10, "distance_km": 5,
    }])
    print(f"Predicted price for {new_house.iloc[0].to_dict()}: "
          f"${model.predict(new_house)[0]:,.0f}")

    # 7. Visualise predictions vs. reality
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, predictions, alpha=0.6)
    lims = [y.min(), y.max()]
    plt.plot(lims, lims, "r--", label="Perfect prediction")
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.title("House prices: actual vs predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png")
    print(f"\nChart saved to {OUTPUT_DIR / 'actual_vs_predicted.png'}")


if __name__ == "__main__":
    main()
