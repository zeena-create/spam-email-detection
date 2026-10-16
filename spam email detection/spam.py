import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

try:
    df = pd.read_csv("spamhamdata.csv", encoding="latin-1", on_bad_lines="skip")
except Exception:
    df = pd.read_table("spamhamdata.csv", header=None, names=["label", "message"], encoding="latin-1")

df = df.dropna(how="all")

if len(df.columns) >= 2:
    label_col = df.columns[0]
    text_col = df.columns[1]

df[text_col] = df[text_col].fillna("").astype(str)
df = df[df[text_col].str.strip() != ""]

labels = df[label_col].astype(str).str.lower().str.strip()

if labels.str.contains("spam|ham").any():
    df["target"] = labels.map(lambda x: 1 if "spam" in x else 0)
else:
    df["target"] = pd.factorize(labels)[0]

X_train, X_test, y_train, y_test = train_test_split(
    df[text_col], df["target"], test_size=0.20, random_state=42
)

vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=300,
    random_state=42,
)

mlp.fit(X_train_tfidf, y_train)

y_pred = mlp.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

sample_messages = [
    "Congratulations! You won a $1,000 gift card. Click here to claim now.",
    "Hey, are we still meeting for dinner tonight at 7?",
]

sample_tfidf = vectorizer.transform(sample_messages)
predictions = mlp.predict(sample_tfidf)

print("\nCustom Predictions:\n")
for msg, pred in zip(sample_messages, predictions):
    label = "Spam" if pred == 1 else "Ham"
    print(f"Message: '{msg}'\nPrediction: {label}\n")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 1. Compute Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# 2. Plot Confusion Matrix & Metrics
plt.figure(figsize=(8, 6), dpi=300)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"],
    annot_kws={"size": 14, "weight": "bold"},
)

plt.title("MLP Spam Detection - Confusion Matrix", fontsize=14, fontweight="bold")
plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
plt.ylabel("Actual Label", fontsize=12, fontweight="bold")

# Save as PNG image
plt.tight_layout()
plt.savefig("results_plot.png", dpi=300)
plt.show()

import gradio as gr

def predict_spam(message):
    message_tfidf = vectorizer.transform([message])
    prediction = mlp.predict(message_tfidf)[0]
    return "🚨 Spam Email" if prediction == 1 else "✅ Normal Email (Ham)"

interface = gr.Interface(
    fn=predict_spam,
    inputs=gr.Textbox(lines=2, placeholder="Type your email or message here..."),
    outputs="text",
    title="Spam Email Detection System",
    description="Enter a message to check whether it's Spam or Ham using MLP Model.",
)

interface.launch(share=True) 

