# Project 3: Spam Message Detector

**Type:** Text classification (Natural Language Processing)
**Algorithm:** Multinomial Naive Bayes with TF-IDF features

## What you'll learn
- How to turn **text into numbers**:
  - **Bag-of-words**: count how often each word appears
  - **TF-IDF**: give more weight to words that are rare across messages
- **Stop words**: removing common words like "the" and "is"
- **Pipelines**: chaining preprocessing and model into one object
- **Precision vs recall**: is it worse to mark a real message as spam, or to let spam through?
- How to inspect which words the model finds most "spammy"

## The data
40 example messages (20 spam, 20 ham) are written directly in `main.py`, so
you can read and edit every one.
For a bigger real dataset, look up the *SMS Spam Collection* on Kaggle or UCI
(about 5,500 messages).

## Run it
```bash
python main.py
```
Output: a bag-of-words demo, metrics, the top spam words, predictions for new
messages, and `outputs/confusion_matrix.png`.

## Try it yourself
1. Add 10 more messages of your own to `SPAM` and `HAM`. Does accuracy change?
2. Replace `TfidfVectorizer` with `CountVectorizer`. Which works better?
3. Add `ngram_range=(1, 2)` to the vectorizer so it also learns word pairs.
4. Download the real SMS Spam Collection, load it with `pd.read_csv`, and retrain.
