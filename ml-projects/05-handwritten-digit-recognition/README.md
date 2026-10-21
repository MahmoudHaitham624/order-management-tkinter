# Project 5: Handwritten Digit Recognition

**The question:** can the computer look at a picture of a handwritten number
and say which digit (0 to 9) it is?

**Type:** Image classification (10 possible answers)
**Models:** Logistic Regression and Support Vector Machine (SVM)

## The big new idea: a picture is just numbers
Each picture is 8 × 8 = 64 pixels. Each pixel is a number from 0 (white) to
16 (black). We line the 64 numbers up in one row, and that row is the input,
the same as any other table.

## How the code works, step by step

| Step | What happens |
|------|--------------|
| 1 | Load 1,797 pictures. Print one as numbers and show it as an image. |
| 2 | `X` = the 64 pixel numbers of each picture, `y` = the real digit. |
| 3 | Split into training and testing. |
| 4 | Train Logistic Regression and check its accuracy. |
| 5 | Train an SVM and check its accuracy. It does better! |
| 6 | Print every picture the SVM got wrong. |
| 7 | A confusion matrix shows which digits get mixed up (for example, 8 and 1). |

## Reading the results
About 99% of test pictures are read correctly by the SVM, so only a few
mistakes out of 360.

## Run it
```bash
python main.py
```

## Try it yourself
1. Change `test_size=0.2` to `0.9`, so the model learns from only 10% of pictures. How much worse is it?
2. Add a third model: `KNeighborsClassifier` (from `sklearn.neighbors`).
3. Show one of the wrong pictures with `plt.imshow(X_test[i].reshape(8, 8), cmap="gray_r")`.
