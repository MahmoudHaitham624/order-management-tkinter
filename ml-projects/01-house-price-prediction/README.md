# Project 1: House Price Prediction

**Type:** Regression (predicting a number)
**Algorithm:** Linear Regression

## What you'll learn
- The standard ML workflow: **load → split → train → evaluate → predict**
- Why we keep a **test set** the model never sees during training
- How to read regression metrics:
  - **MAE**: on average, how many dollars off we are
  - **RMSE**: like MAE, but big mistakes count extra
  - **R²**: share of the price variation the model explains (1.0 is perfect)
- How to read **coefficients**: how much the price changes per extra m², bedroom, and so on

## The data
500 generated houses with `size_m2`, `bedrooms`, `age_years`, `distance_km`,
and `price`. The data comes from a known formula plus noise, so you can check
whether the model recovers the true rule (+1,500 per m², +10,000 per bedroom,
−1,000 per year of age, −3,000 per km).

## Run it
```bash
python main.py
```
Output: metrics in the terminal, plus `outputs/actual_vs_predicted.png`.

## Try it yourself
1. Change `test_size` to 0.5. Do the metrics get worse?
2. Increase the noise (`rng.normal(0, 25_000, n)`) to 100,000. What happens to R²?
3. Remove the `size_m2` column from `X`. How much worse is the model?
4. Replace `LinearRegression` with `DecisionTreeRegressor` from `sklearn.tree` and compare.
