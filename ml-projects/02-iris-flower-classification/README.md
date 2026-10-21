# Project 2: Iris Flower Classification

**The question:** if I measure a flower's petals and sepals, can the computer
tell which of 3 species it is?

**Type:** Classification (the answer is a category)
**Models:** K-Nearest Neighbours (KNN) and Decision Tree

## The idea in one sentence each
- **KNN:** look at the 5 most similar flowers you have seen, and copy their species.
- **Decision Tree:** learn a list of yes/no questions, like "Is the petal shorter than 2.45 cm?"

## How the code works, step by step

| Step | What happens |
|------|--------------|
| 1 | Load 150 real flowers. `X` = 4 measurements, `y` = species (0, 1 or 2). |
| 2 | Split into training and testing. `stratify=y` keeps all 3 species fairly mixed in both parts. |
| 3 | Train KNN and check its accuracy. |
| 4 | Try KNN with different numbers of neighbours (`k`) to see which works best. |
| 5 | Train a Decision Tree and print the yes/no rules it learned. You can read them! |
| 6 | Classify a brand-new flower. |
| 7 | A confusion matrix shows which species the model mixes up. |

## Reading the results
- **Accuracy 0.97** means 97% of the test flowers were guessed correctly.
- In the confusion matrix, numbers on the diagonal (top-left to bottom-right) are correct guesses.

## Run it
```bash
python main.py
```

## Try it yourself
1. Change the tree's `max_depth=3` to `1`. What happens to accuracy?
2. Add `k = 50` to the list in Step 4. Why does it do worse?
3. Change the measurements of `new_flower` and see if the species changes.
