# Project 1: House Price Prediction

**The question:** if I know a house's size, bedrooms, age and distance from
the city, can the computer guess its price?

**Type:** Regression (the answer is a number)
**Model:** Linear Regression

## The idea in one sentence
Linear Regression finds how much each feature adds to or takes away from the
price, for example "+$1,500 for every extra m²".

## How the code works, step by step

| Step | What happens |
|------|--------------|
| 1 | We make 500 pretend houses with random numbers. Prices follow a hidden rule plus some random "luck". |
| 2 | `X` = size, bedrooms, age, distance (the inputs). `y` = price (the answer). |
| 3 | We split: 400 houses to learn from, 100 kept secret for testing. |
| 4 | `model.fit(...)` is where the learning happens. |
| 5 | We print what the model learned. Compare it with the hidden rule in Step 1: the numbers are very close! |
| 6 | We test on the 100 secret houses and measure how wrong the guesses are. |
| 7 | We guess the price of a brand-new house. |
| 8 | A chart of real vs predicted prices. Dots close to a straight line mean good guesses. |

## Reading the results
- **"Off by $21,000 on average"**: a typical guess is about $21k too high or too low.
- **R2 score of 0.94**: the model explains 94% of why prices differ. 1.0 would be perfect.

## Run it
```bash
python main.py
```

## Try it yourself
1. Change `test_size=0.2` to `0.5`. Does the error get bigger?
2. Make the luck bigger: change `25000` to `100000`. What happens to the R2 score?
3. Remove `"size"` from `X` (and from `new_house`). How much worse is the model?
