"""
Fake News Detection System
==========================
Uses TF-IDF vectorization + Passive Aggressive Classifier
with a full pipeline: preprocessing → training → evaluation → prediction.
"""

import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_curve, auc
)
from sklearn.pipeline import Pipeline


FAKE_HEADLINES = [
    "SHOCKING: Scientists discover that vaccines cause mind control",
    "Breaking: President secretly replaced by an alien clone",
    "You won't believe what doctors don't want you to know about cancer",
    "URGENT: Government hiding cure for all diseases from the public",
    "Celebrity reveals secret society controls the entire world",
    "This one weird trick eliminates all debt instantly — banks hate it",
    "5G towers are transmitting mind-control signals to citizens",
    "Local man grows wings after eating this common household fruit",
    "Scientists ADMIT the moon landing was filmed in Hollywood",
    "New law will ban all cash, force citizens into digital slavery",
    "Politician caught on tape admitting to lizard shapeshifting",
    "Secret documents prove flat earth theory once and for all",
    "Doctors baffled as man cures diabetes by drinking bleach",
    "Hidden pyramid discovered beneath White House lawn",
    "Chemtrails confirmed to contain mind-altering chemicals by whistleblower",
    "Exclusive: CIA using household appliances to spy on every American",
    "World leaders meet secretly to plan global population reduction",
    "Ancient cure for cancer suppressed by pharmaceutical companies",
    "New study proves that the Earth is only 6,000 years old",
    "Billionaire reveals he has been living on the moon for 20 years",
]

REAL_HEADLINES = [
    "Federal Reserve raises interest rates by 0.25 percentage points",
    "New climate study shows global temperatures rising faster than expected",
    "Local government approves budget for infrastructure improvements",
    "Scientists develop promising new treatment for Alzheimer's disease",
    "Stock markets close lower amid concerns about global trade tensions",
    "UN report highlights growing water scarcity in developing nations",
    "City council votes to expand public transportation network",
    "Researchers publish findings on impact of diet on mental health",
    "Technology company announces quarterly earnings above expectations",
    "Health officials urge vaccinations as flu season approaches",
    "New legislation aims to improve data privacy protections for citizens",
    "Archaeological team uncovers ancient settlement in Middle East",
    "Electric vehicle sales reach record high in first quarter",
    "Scientists successfully test new renewable energy storage technology",
    "International summit addresses rising tensions in disputed region",
    "Local hospital receives grant to expand emergency care facilities",
    "University study links exercise to improved cognitive function in elderly",
    "Government announces new measures to reduce carbon emissions",
    "Central bank issues warning about rising household debt levels",
    "Medical researchers identify genetic markers linked to heart disease",
]


def build_dataset():
    fake_df = pd.DataFrame({"text": FAKE_HEADLINES, "label": 0})
    real_df = pd.DataFrame({"text": REAL_HEADLINES,  "label": 1})
    df = pd.concat([fake_df, real_df], ignore_index=True).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)
    return df


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    stopwords = {
        "the","a","an","and","or","but","in","on","at","to","for",
        "of","with","by","from","is","was","are","were","be","been",
        "being","have","has","had","do","does","did","will","would",
        "could","should","may","might","this","that","these","those",
        "it","its","i","you","he","she","we","they","as","up","out",
        "about","so","if","not","no","all","just","into","than","then",
    }
    tokens = [w for w in text.split() if w not in stopwords]
    return " ".join(tokens)


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            max_df=0.95,
        )),
        ("clf", PassiveAggressiveClassifier(
            C=0.5,
            max_iter=1000,
            random_state=42,
        )),
    ])


def train_and_evaluate(df):
    X = df["text"].apply(preprocess_text)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred  = pipeline.predict(X_test)
    y_score = pipeline.decision_function(X_test)

    acc    = accuracy_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["FAKE", "REAL"])

    print("=" * 55)
    print("         FAKE NEWS DETECTION — RESULTS")
    print("=" * 55)
    print(f"  Accuracy : {acc * 100:.1f}%")
    print(f"  Samples  : {len(df)} total  |  train={len(X_train)}  test={len(X_test)}")
    print("-" * 55)
    print(report)

    return pipeline, y_test, y_pred, y_score, cm, acc


def plot_results(y_test, y_pred, y_score, cm, acc, df):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Fake News Detection — Model Evaluation", fontsize=16,
                 fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    COLORS = {"fake": "#E74C3C", "real": "#2ECC71",
              "bg": "#F8F9FA",   "accent": "#3498DB"}

    # A. Class Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    counts = df["label"].value_counts()
    bars = ax1.bar(["FAKE", "REAL"],
                   [counts.get(0, 0), counts.get(1, 0)],
                   color=[COLORS["fake"], COLORS["real"]],
                   edgecolor="white", linewidth=1.5, width=0.5)
    for bar, cnt in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 str(cnt), ha="center", va="bottom", fontweight="bold")
    ax1.set_title("Dataset Distribution", fontweight="bold")
    ax1.set_ylabel("Number of Articles")
    ax1.set_facecolor(COLORS["bg"])
    ax1.spines[["top", "right"]].set_visible(False)

    # B. Confusion Matrix
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(cm, cmap="RdYlGn", aspect="auto")
    ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["FAKE", "REAL"])
    ax2.set_yticklabels(["FAKE", "REAL"])
    ax2.set_xlabel("Predicted Label")
    ax2.set_ylabel("True Label")
    ax2.set_title("Confusion Matrix", fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, str(cm[i, j]), ha="center", va="center",
                     fontsize=18, fontweight="bold",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    # C. ROC Curve
    ax3 = fig.add_subplot(gs[0, 2])
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    ax3.plot(fpr, tpr, color=COLORS["accent"], lw=2.5, label=f"AUC = {roc_auc:.3f}")
    ax3.fill_between(fpr, tpr, alpha=0.1, color=COLORS["accent"])
    ax3.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.set_title("ROC Curve", fontweight="bold")
    ax3.legend(loc="lower right")
    ax3.set_facecolor(COLORS["bg"])
    ax3.spines[["top", "right"]].set_visible(False)

    # D. Performance Metrics
    ax4 = fig.add_subplot(gs[1, 0])
    from sklearn.metrics import precision_score, recall_score, f1_score
    metrics_vals = {
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall":    recall_score(y_test, y_pred, average="weighted"),
        "F1-Score":  f1_score(y_test, y_pred, average="weighted"),
        "Accuracy":  acc,
    }
    colors_m = [COLORS["accent"], "#9B59B6", "#F39C12", COLORS["real"]]
    bars_m = ax4.barh(list(metrics_vals.keys()),
                      list(metrics_vals.values()),
                      color=colors_m, edgecolor="white", height=0.5)
    for bar in bars_m:
        w = bar.get_width()
        ax4.text(w - 0.03, bar.get_y() + bar.get_height() / 2,
                 f"{w * 100:.1f}%", va="center", ha="right",
                 color="white", fontweight="bold")
    ax4.set_xlim(0, 1.05)
    ax4.set_title("Performance Metrics", fontweight="bold")
    ax4.set_facecolor(COLORS["bg"])
    ax4.spines[["top", "right"]].set_visible(False)

    # E. Per-class Performance
    ax5 = fig.add_subplot(gs[1, 1])
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1])
    x_idx = np.arange(2)
    w = 0.25
    ax5.bar(x_idx - w, p, w, label="Precision", color=COLORS["accent"])
    ax5.bar(x_idx,     r, w, label="Recall",    color="#9B59B6")
    ax5.bar(x_idx + w, f, w, label="F1-Score",  color="#F39C12")
    ax5.set_xticks(x_idx)
    ax5.set_xticklabels(["FAKE", "REAL"])
    ax5.set_ylim(0, 1.15)
    ax5.set_title("Per-Class Performance", fontweight="bold")
    ax5.legend(fontsize=8)
    ax5.set_facecolor(COLORS["bg"])
    ax5.spines[["top", "right"]].set_visible(False)

    # F. Accuracy Summary Card
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_xlim(0, 1); ax6.set_ylim(0, 1)
    ax6.axis("off")
    ax6.set_facecolor(COLORS["bg"])
    pct = acc * 100
    color = COLORS["real"] if pct >= 80 else "#F39C12" if pct >= 60 else COLORS["fake"]
    ax6.text(0.5, 0.72, f"{pct:.1f}%",
             ha="center", va="center", fontsize=48,
             fontweight="bold", color=color)
    ax6.text(0.5, 0.45, "Overall Accuracy",
             ha="center", va="center", fontsize=14, color="#555")
    verdict = "Excellent" if pct >= 85 else "Good" if pct >= 70 else "Fair"
    ax6.text(0.5, 0.28, f"Model Rating: {verdict}",
             ha="center", va="center", fontsize=12,
             color="white", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.85))
    ax6.text(0.5, 0.12,
             f"TF-IDF + Passive Aggressive Classifier\n"
             f"Samples: {len(df)}  |  Features: 5000 ngrams",
             ha="center", va="center", fontsize=8, color="#777")

    plt.savefig("fake_news_results.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("\n  Chart saved → fake_news_results.png")
    plt.show()


def predict(pipeline, texts):
    print("\n" + "=" * 55)
    print("           LIVE PREDICTION DEMO")
    print("=" * 55)
    cleaned = [preprocess_text(t) for t in texts]
    preds   = pipeline.predict(cleaned)
    scores  = pipeline.decision_function(cleaned)

    for text, pred, score in zip(texts, preds, scores):
        label = "REAL ✅" if pred == 1 else "FAKE ❌"
        conf  = min(abs(score) / 3 * 100, 99.9)
        print(f"\n  Text    : {text[:70]}{'...' if len(text) > 70 else ''}")
        print(f"  Result  : {label}   (confidence ≈ {conf:.0f}%)")


if __name__ == "__main__":

    df = build_dataset()
    print(f"\n  Dataset loaded: {len(df)} articles  "
          f"(FAKE={sum(df.label==0)}, REAL={sum(df.label==1)})\n")

    pipeline, y_test, y_pred, y_score, cm, acc = train_and_evaluate(df)

    plot_results(y_test, y_pred, y_score, cm, acc, df)

    demo_texts = [
        "Government scientists reveal moon is made entirely of cheese",
        "Central bank raises benchmark interest rate by 25 basis points",
        "Secret pill that big pharma doesn't want you to know about cures everything",
        "City council approves new budget for road maintenance and public parks",
    ]
    predict(pipeline, demo_texts)

    print("\n" + "=" * 55)
    print("  Done! Model ready for production use.")
    print("=" * 55)