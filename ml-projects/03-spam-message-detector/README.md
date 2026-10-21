# Project 3: Spam Message Detector

**The question:** can the computer read a message and tell if it is spam?

**Type:** Text classification
**Model:** Naive Bayes

## The big new idea: words must become numbers
Models only understand numbers. `CountVectorizer` builds a list of all the
words it has seen, then turns each message into counts:

```
Words:            call  free  now  prize
"free prize now"   0     1     1     1
"call me now"      1     0     1     0
```

**Naive Bayes** then learns which words appear more often in spam ("free",
"win", "claim") and which appear more often in normal messages.

## How the code works, step by step

| Step | What happens |
|------|--------------|
| 1 | 20 spam and 20 normal messages are written in the code, plus a matching list of labels. |
| 2 | Split into training and testing messages. |
| 3 | Turn the words into numbers. `fit_transform` learns the words from the training messages; `transform` only counts them in the test messages. |
| 4 | Train the Naive Bayes model. |
| 5 | Check accuracy on the test messages. |
| 6 | Try 4 brand-new messages. Remember to turn them into numbers first! |
| 7 | A confusion matrix of correct vs wrong guesses. |

## Run it
```bash
python main.py
```

## Try it yourself
1. Add your own messages to `new_messages` in Step 6. Can you trick the model?
2. Add 5 more messages to each list. Does accuracy change?
3. Replace `CountVectorizer` with `TfidfVectorizer` (import it from the same place).
   It gives less weight to very common words.
