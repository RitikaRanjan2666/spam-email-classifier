# =====================================================
# Spam Email Classifier
# Author: Ritika Ranjan
# Tech: Python, scikit-learn, Naive Bayes, Logistic Regression
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings('ignore')


# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────

def load_data():
    """Load the SMS Spam dataset."""
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    try:
        df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
        print(f"✅ Dataset loaded: {df.shape[0]} messages")
    except Exception:
        # Fallback: small built-in sample if internet is unavailable
        print("⚠️  Could not fetch online dataset. Using built-in sample.")
        data = {
            'label': ['ham','ham','spam','spam','ham','spam','ham','ham','spam','spam'],
            'message': [
                'Hey, are you coming to the meeting tomorrow?',
                'Sure, I will be there at 10am.',
                'Congratulations! You won a $1000 gift card. Call now!',
                'FREE entry in 2 a weekly competition to win FA Cup final',
                'Can you please call me when you are free?',
                'WINNER! You have been selected for a cash prize of £500.',
                'Are you free this weekend for a movie?',
                'The homework is due next Monday, do not forget.',
                'Urgent! Your mobile number has been awarded $2000.',
                'You have won a lottery. Send bank details to claim prize.'
            ]
        }
        df = pd.DataFrame(data)
    return df


# ── 2. EXPLORE DATA ───────────────────────────────────────────────────────────

def explore_data(df):
    """Print basic dataset statistics."""
    print("\n📊 Dataset Overview")
    print("-" * 40)
    print(df['label'].value_counts())
    print(f"\nSpam %  : {df['label'].value_counts(normalize=True)['spam']*100:.1f}%")
    print(f"Ham  %  : {df['label'].value_counts(normalize=True)['ham']*100:.1f}%")
    df['message_length'] = df['message'].apply(len)
    print(f"\nAvg message length (spam): {df[df['label']=='spam']['message_length'].mean():.0f} chars")
    print(f"Avg message length (ham) : {df[df['label']=='ham']['message_length'].mean():.0f} chars")


# ── 3. VISUALISE ──────────────────────────────────────────────────────────────

def plot_insights(df):
    """Generate and save insight charts."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Spam Email Classifier – Data Insights', fontsize=14, fontweight='bold')

    # Class distribution
    counts = df['label'].value_counts()
    axes[0].bar(counts.index, counts.values, color=['#4CAF50', '#F44336'], edgecolor='white')
    axes[0].set_title('Class Distribution')
    axes[0].set_xlabel('Label')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold')

    # Message length distribution
    df['message_length'] = df['message'].apply(len)
    for label, color in [('ham', '#4CAF50'), ('spam', '#F44336')]:
        subset = df[df['label'] == label]['message_length']
        axes[1].hist(subset, bins=30, alpha=0.6, label=label, color=color)
    axes[1].set_title('Message Length by Class')
    axes[1].set_xlabel('Character Count')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('data/insights.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("📈 Saved: data/insights.png")


# ── 4. PREPROCESS ─────────────────────────────────────────────────────────────

def preprocess(df):
    """Encode labels and vectorise text using TF-IDF."""
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label_num'],
        test_size=0.2, random_state=42, stratify=df['label_num']
    )

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tf = vectorizer.fit_transform(X_train)
    X_test_tf  = vectorizer.transform(X_test)

    print(f"\n✂️  Train size: {X_train_tf.shape[0]} | Test size: {X_test_tf.shape[0]}")
    return X_train_tf, X_test_tf, y_train, y_test, vectorizer


# ── 5. TRAIN & EVALUATE ───────────────────────────────────────────────────────

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train both models and print comparison."""
    models = {
        'Naive Bayes (MultinomialNB)': MultinomialNB(),
        'Logistic Regression':         LogisticRegression(max_iter=1000, random_state=42)
    }

    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Confusion Matrices', fontsize=13, fontweight='bold')

    print("\n🤖 Model Results")
    print("=" * 50)

    for idx, (name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {'model': model, 'accuracy': acc}

        print(f"\n📌 {name}")
        print(f"   Accuracy : {acc*100:.2f}%")
        print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=['Ham', 'Spam'])
        disp.plot(ax=axes[idx], colorbar=False, cmap='Blues')
        axes[idx].set_title(f'{name}\nAccuracy: {acc*100:.1f}%')

    plt.tight_layout()
    plt.savefig('data/confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n📈 Saved: data/confusion_matrices.png")
    return results


# ── 6. PREDICT NEW MESSAGES ───────────────────────────────────────────────────

def predict_message(message, model, vectorizer):
    """Predict if a single message is spam or ham."""
    vec = vectorizer.transform([message])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    label = "🚨 SPAM" if pred == 1 else "✅ HAM (Not Spam)"
    confidence = max(prob) * 100
    print(f"\nMessage  : {message}")
    print(f"Result   : {label}")
    print(f"Confidence: {confidence:.1f}%")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("       SPAM EMAIL CLASSIFIER — Ritika Ranjan")
    print("=" * 55)

    df          = load_data()
    explore_data(df)
    plot_insights(df)

    X_train, X_test, y_train, y_test, vectorizer = preprocess(df)
    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    # Pick best model
    best_name = max(results, key=lambda k: results[k]['accuracy'])
    best_model = results[best_name]['model']
    print(f"\n🏆 Best Model: {best_name}")

    # Demo predictions
    print("\n── Demo Predictions ──────────────────────────────")
    test_messages = [
        "Congratulations! You've won a $500 Amazon gift card. Click here to claim.",
        "Hey, are we still meeting for lunch today?",
        "URGENT: Your bank account has been suspended. Verify now to restore access.",
        "Can you send me the notes from yesterday's class?"
    ]
    for msg in test_messages:
        predict_message(msg, best_model, vectorizer)

    print("\n✅ Done! Check the data/ folder for saved charts.")
