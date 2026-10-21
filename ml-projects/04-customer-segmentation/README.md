# Project 4: Customer Segmentation

**The question:** can the computer split shop customers into groups of
similar people, without us telling it what the groups are?

**Type:** Clustering (unsupervised learning, with no `y` answers at all)
**Model:** K-Means

## The idea in one sentence
K-Means drops *k* centre points, gives each customer to the nearest centre,
moves each centre to the middle of its customers, and repeats until nothing
changes.

## How the code works, step by step

| Step | What happens |
|------|--------------|
| 1 | We secretly make 4 kinds of customers (60 each), then mix them into one table. |
| 2 | **Scale** the data so age, income and spending all count equally. |
| 3 | **Elbow method**: try 1 to 8 groups and draw a chart. The bend (the "elbow") shows the best number. |
| 4 | Train the final model with 4 groups. Each customer gets a group number. |
| 5 | Print the average customer in each group. You'll see the 4 kinds from Step 1! |
| 6 | Draw the customers coloured by group. |

## Notice: there is no train/test split here
We have no right answers to test against. We judge the groups by looking at
them and asking: do they make sense?

## Run it
```bash
python main.py
```

## Try it yourself
1. Change `n_clusters=4` to `3`, then `6`. How do the groups change?
2. Give each group a name, like "Young big spenders".
3. Bonus: this repo's `shop3.db` has real customers and orders. Work out each
   customer's total spend with pandas, then group them!
