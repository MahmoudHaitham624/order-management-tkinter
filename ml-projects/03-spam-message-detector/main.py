"""
Project 3: Spam Message Detector (Naive Bayes + TF-IDF)

Goal: decide whether a text message is spam or a normal message ("ham").
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

SPAM = [
    "WINNER! You have won a free iPhone, click here to claim now",
    "Congratulations! You won $1000 cash prize, call now",
    "URGENT: your account is locked, verify your password at this link",
    "Get cheap loans today, no credit check, apply now",
    "Free entry to win a holiday, text WIN to 80085",
    "You have been selected for a free gift card, claim today",
    "Limited offer! Buy one get one free, click the link",
    "Earn money from home fast, no experience needed",
    "Claim your prize now before it expires, reply YES",
    "Exclusive deal just for you, 90% off, order now",
    "Your number won the lottery, send your bank details",
    "Hot singles in your area, click to chat free",
    "Act now! Lowest price on meds, free shipping",
    "You are a lucky winner, call this number to collect your reward",
    "Final notice: claim your free vacation package today",
    "Make $500 a day with this simple trick, click here",
    "Free ringtones! Text TONE to receive yours now",
    "Your loan is pre-approved, click to get cash instantly",
    "Win a brand new car, enter the free draw now",
    "Urgent prize waiting for you, call now to claim cash",
]

HAM = [
    "Hey, are we still meeting for lunch tomorrow?",
    "Can you send me the notes from today's lecture?",
    "I'll be home late tonight, don't wait for dinner",
    "Happy birthday! Hope you have a great day",
    "Did you finish the assignment for data handling?",
    "Mum says call her when you get a chance",
    "The meeting moved to 3pm, see you there",
    "Thanks for helping me move yesterday",
    "What time does the movie start?",
    "Can you pick up some milk on the way home?",
    "Good luck on your exam tomorrow!",
    "I left my keys at your place, can I come get them",
    "Let's study together at the library this weekend",
    "The order arrived today, everything looks fine",
    "Are you coming to the football match on Saturday?",
    "Running 10 minutes late, sorry",
    "Don't forget we have class at 9 tomorrow",
    "Just landed, will call you from the hotel",
    "Can you review my project code when you are free?",
    "Dinner was great, let's do it again soon",
]


def main():
    # 1. Build the dataset
    df = pd.DataFrame({
        "text": SPAM + HAM,
        "label": ["spam"] * len(SPAM) + ["ham"] * len(HAM),
    })
    print(df.sample(5, random_state=1), "\n")

    # 2. See how text becomes numbers (bag-of-words)
    demo = CountVectorizer()
    counts = demo.fit_transform(["free prize now", "call me now"])
    print("Bag-of-words demo")
    print("  vocabulary:", demo.get_feature_names_out())
    print("  vectors:\n", counts.toarray(), "\n")

    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.3, random_state=42,
        stratify=df["label"],
    )

    # 4. A pipeline chains "text -> TF-IDF numbers -> Naive Bayes"
    model = make_pipeline(
        TfidfVectorizer(lowercase=True, stop_words="english"),
        MultinomialNB(),
    )
    model.fit(X_train, y_train)

    # 5. Evaluate
    preds = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(classification_report(y_test, preds))

    ConfusionMatrixDisplay.from_predictions(y_test, preds, cmap="Oranges")
    plt.title("Spam detector - confusion matrix")
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png")

    # 6. Which words scream "spam"?
    vectorizer = model.named_steps["tfidfvectorizer"]
    nb = model.named_steps["multinomialnb"]
    spam_idx = list(nb.classes_).index("spam")
    ham_idx = list(nb.classes_).index("ham")
    spamminess = nb.feature_log_prob_[spam_idx] - nb.feature_log_prob_[ham_idx]
    words = vectorizer.get_feature_names_out()
    top = sorted(zip(spamminess, words), reverse=True)[:10]
    print("Top 10 spam words:", [w for _, w in top], "\n")

    # 7. Try your own messages
    tests = [
        "Click now to claim your free cash prize",
        "Are you free for coffee after class?",
        "You won a gift card, reply to collect",
        "Please send me the report by Friday",
    ]
    for msg, label, proba in zip(tests, model.predict(tests),
                                 model.predict_proba(tests)):
        print(f"[{label.upper():4s} {max(proba):.0%}] {msg}")
    print(f"\nChart saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
