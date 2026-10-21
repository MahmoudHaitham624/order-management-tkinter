# Project 3: Spam Message Detector
# We teach the computer to tell spam messages apart from normal messages.

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay


# ---------- Step 1: Write some example messages ----------
spam_messages = [
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
    "Act now! Lowest price on meds, free shipping",
    "You are a lucky winner, call this number to collect your reward",
    "Final notice: claim your free vacation package today",
    "Make $500 a day with this simple trick, click here",
    "Free ringtones! Text TONE to receive yours now",
    "Your loan is pre-approved, click to get cash instantly",
    "Win a brand new car, enter the free draw now",
    "Urgent prize waiting for you, call now to claim cash",
    "You won a free trip, click the link to claim",
]

normal_messages = [
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

# Put all messages in one list, and make a matching list of labels.
# ["spam"] * 20 makes a list with the word "spam" written 20 times.
messages = spam_messages + normal_messages
labels = ["spam"] * len(spam_messages) + ["normal"] * len(normal_messages)
print("Total messages:", len(messages))
print()


# ---------- Step 2: Split into training data and testing data ----------
X_train, X_test, y_train, y_test = train_test_split(
    messages, labels, test_size=0.3, random_state=42, stratify=labels
)


# ---------- Step 3: Turn words into numbers ----------
# A model can only work with numbers, not words.
# CountVectorizer counts how many times each word appears in a message.
vectorizer = CountVectorizer()

# Learn the list of words from the TRAINING messages, then count them
X_train_numbers = vectorizer.fit_transform(X_train)

# Count the same words in the TEST messages (no new learning here)
X_test_numbers = vectorizer.transform(X_test)

print("The model knows", len(vectorizer.get_feature_names_out()), "different words")
print()


# ---------- Step 4: Create the model and train it ----------
# Naive Bayes learns how often each word shows up in spam vs normal messages.
# Words like "free", "win" and "claim" will point strongly to spam.
model = MultinomialNB()
model.fit(X_train_numbers, y_train)


# ---------- Step 5: Test the model ----------
predictions = model.predict(X_test_numbers)
print("Accuracy:", accuracy_score(y_test, predictions))
print()


# ---------- Step 6: Try it on brand-new messages ----------
new_messages = [
    "Click now to claim your free cash prize",
    "Are you free for coffee after class?",
    "You won a gift card, reply to collect",
    "Please send me the report by Friday",
]
new_numbers = vectorizer.transform(new_messages)  # words -> numbers again
new_predictions = model.predict(new_numbers)

for i in range(len(new_messages)):
    print(new_predictions[i], "-->", new_messages[i])


# ---------- Step 7: Draw a confusion matrix ----------
ConfusionMatrixDisplay.from_predictions(y_test, predictions)
plt.title("Spam detector: correct vs wrong guesses")
plt.show()
