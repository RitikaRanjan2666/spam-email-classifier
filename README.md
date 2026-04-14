# 📧 Spam Email Classifier

A machine learning project that classifies messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) techniques.

Built by **Ritika Ranjan** | B.Tech CSE (AI/ML), CV Raman Global University

---

## 🚀 Demo

```
Message   : Congratulations! You've won a $500 Amazon gift card. Click here to claim.
Result    : 🚨 SPAM
Confidence: 98.3%

Message   : Hey, are we still meeting for lunch today?
Result    : ✅ HAM (Not Spam)
Confidence: 99.1%
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| scikit-learn | ML models (Naive Bayes, Logistic Regression) |
| TF-IDF Vectorizer | Text → numbers |
| Pandas & NumPy | Data handling |
| Matplotlib & Seaborn | Visualizations |

---

## 📁 Project Structure

```
spam-classifier/
│
├── spam_classifier.py     # Main script — training + prediction
├── requirements.txt       # All dependencies
├── data/
│   ├── insights.png       # Class distribution & message length charts
│   └── confusion_matrices.png  # Model performance charts
└── README.md
```

---

## ⚙️ How to Run

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/spam-email-classifier.git
cd spam-email-classifier
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the classifier**
```bash
python spam_classifier.py
```

---

## 📊 Model Performance

| Model | Accuracy |
|-------|----------|
| Naive Bayes (MultinomialNB) | ~97% |
| Logistic Regression | ~98% |

---

## 🧠 How It Works

1. **Load Data** — Uses the SMS Spam Collection dataset (~5,500 messages)
2. **EDA** — Explores class balance and message length patterns
3. **TF-IDF Vectorization** — Converts text into numerical features, ignoring common words
4. **Model Training** — Trains both Naive Bayes and Logistic Regression
5. **Evaluation** — Compares accuracy, precision, recall, and F1-score
6. **Prediction** — Classifies any new input message in real time

---

## 📬 Connect

- LinkedIn: [linkedin.com/in/ritikaranjan-540076338](https://www.linkedin.com/in/ritikaranjan-540076338)
- GitHub: github.com/YOUR_USERNAME
