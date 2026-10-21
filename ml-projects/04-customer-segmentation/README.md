# Project 4: Customer Segmentation

**Type:** Clustering (unsupervised learning, with no labels)
**Algorithm:** K-Means

## What you'll learn
- **Unsupervised learning**: finding structure when no "right answer" is given
- How **K-Means** works: place *k* centres, assign points, move the centres, repeat
- Why **feature scaling** (`StandardScaler`) matters for distance-based models
- Choosing *k* with the **elbow method** and the **silhouette score**
- Turning clusters into **business insight** with segment profiles

## The data
250 generated shop customers with `age`, `annual_income_k`, and
`spending_score`. They come from 4 hidden groups. See whether K-Means finds them.

> Idea: this repo's `shop3.db` holds real customers and orders. A great
> follow-up is to compute total spend and order count per customer from that
> database with pandas, then cluster those.

## Run it
```bash
python main.py
```
Output: scores for k = 2 to 8, segment profiles, `outputs/choose_k.png`, and
`outputs/segments.png`.

## Try it yourself
1. Remove the `StandardScaler` step. Do the segments change?
2. Force `n_clusters=3` and then `6`. How do the profiles change?
3. Give each segment a marketing name (e.g. "Big spenders").
4. Try `AgglomerativeClustering` from `sklearn.cluster` and compare.
