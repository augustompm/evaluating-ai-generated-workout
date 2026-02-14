"""Train and compare ML classifiers for exercise plan line classification."""

import re
import csv
import pickle
import numpy as np
import warnings
from pathlib import Path
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)
from scipy.sparse import hstack, csr_matrix

warnings.filterwarnings("ignore")

LABEL_NAMES = {0: "HEADER", 1: "EXERCISE", 2: "REST", 3: "SAFETY", 4: "NOTE"}

EXERCISE_VOCAB = {
    "walking", "running", "jogging", "cycling", "swimming", "stretching",
    "yoga", "pilates", "squat", "lunge", "push-up", "pushup", "plank",
    "burpee", "jumping", "hiking", "dancing", "rowing", "elliptical",
    "deadlift", "bench", "curl", "press", "pull-up", "pullup",
    "resistance", "dumbbell", "barbell", "kettlebell", "band",
    "tai chi", "aerobics", "sprint", "hiit", "tabata", "circuit",
    "step-up", "calf raise", "leg press", "leg curl", "lat pulldown",
    "cool-down", "warm-up", "cooldown", "warmup",
}
DAY_NAMES = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
INTENSITY_WORDS = {"light", "moderate", "vigorous", "easy", "hard", "intense", "gentle", "brisk"}
SAFETY_WORDS = {"doctor", "physician", "medical", "consult", "blood pressure", "glucose",
                "stop if", "chest pain", "dizziness", "medication", "clearance"}


def extract_features(line: str) -> dict:
    """Extract structural features from a line."""
    t = line.strip()
    tl = t.lower()
    words = tl.split()

    return {
        "has_pipe": int("|" in t),
        "starts_number": int(bool(re.match(r"^\d+[\.\)]\s", t))),
        "starts_bullet": int(t.startswith(("-", "*", "\u2022"))),
        "starts_bold": int(t.startswith("**")),
        "starts_hash": int(t.startswith("#")),
        "has_colon": int(":" in t),
        "char_count": min(len(t), 300),
        "word_count": min(len(words), 50),
        "is_short": int(len(words) < 4),
        "is_long": int(len(words) > 25),
        "has_duration": int(bool(re.search(r"\d+\s*min|hour", tl))),
        "has_intensity": int(any(w in tl for w in INTENSITY_WORDS)),
        "has_exercise_word": int(any(w in tl for w in EXERCISE_VOCAB)),
        "has_day_name": int(any(d in tl for d in DAY_NAMES)),
        "has_number": int(bool(re.search(r"\d", t))),
        "has_structured_format": int(bool(re.search(r"\|\s*intensity:", tl))),
        "has_sets_reps": int(bool(re.search(r"\d+\s*(?:sets?|reps?|x\d)", tl))),
        "has_safety_word": int(any(w in tl for w in SAFETY_WORDS)),
        "starts_note_word": int(any(
            tl.startswith(w) for w in
            ["note", "important", "tip", "remember", "ensure",
             "always", "drink", "aim", "consider", "listen"]
        )),
        "is_header_like": int(bool(
            re.match(r"^(\*\*)?[A-Z][a-z]+day\s*:?\s*(\*\*)?$", t) or
            re.match(r"^(\*\*)?(day\s+\d+|week\s+\d+)\s*:?\s*(\*\*)?$", t, re.I)
        )),
        "has_rest_word": int(any(w in tl for w in ["rest day", "rest", "recovery day"])),
        "pct_uppercase": sum(1 for c in t if c.isupper()) / max(len(t), 1),
    }


def build_feature_matrix(lines: list[str], tfidf_word=None, tfidf_char=None,
                         fit: bool = False):
    """Build combined feature matrix."""
    struct_dicts = [extract_features(line) for line in lines]
    feature_names = list(struct_dicts[0].keys())
    struct_array = np.array([[d[k] for k in feature_names] for d in struct_dicts])
    struct_sparse = csr_matrix(struct_array)

    if fit:
        tfidf_w = tfidf_word.fit_transform(lines)
        tfidf_c = tfidf_char.fit_transform(lines)
    else:
        tfidf_w = tfidf_word.transform(lines)
        tfidf_c = tfidf_char.transform(lines)

    combined = hstack([struct_sparse, tfidf_w, tfidf_c])
    return combined, feature_names


def train_and_evaluate():
    """Train 3 classifiers and compare with 5-fold CV."""
    csv_path = Path("data/annotations/lines_labeled.csv")
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    lines = [r["text"] for r in rows]
    labels = np.array([int(r["label"]) for r in rows])

    print(f"Loaded {len(lines)} lines from {csv_path}")
    print(f"Label distribution:")
    for lid, count in sorted(Counter(labels).items()):
        print(f"  {LABEL_NAMES[lid]:10s}: {count:4d} ({100*count/len(labels):.1f}%)")

    tfidf_word = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), max_features=800,
        min_df=2, sublinear_tf=True,
    )
    tfidf_char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), max_features=500,
        min_df=2, sublinear_tf=True,
    )

    print("\nBuilding features...")
    X, feat_names = build_feature_matrix(lines, tfidf_word, tfidf_char, fit=True)
    y = labels
    print(f"Feature matrix: {X.shape}")

    classifiers = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=3,
            random_state=42, n_jobs=-1, class_weight="balanced",
        ),
        "SVM (RBF)": make_pipeline(
            MaxAbsScaler(),
            SVC(kernel="rbf", C=10.0, gamma="scale",
                random_state=42, class_weight="balanced"),
        ),
    }

    try:
        from catboost import CatBoostClassifier
        classifiers["CatBoost"] = CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.1,
            verbose=0, random_seed=42, auto_class_weights="Balanced",
        )
    except ImportError:
        print("CatBoost not available, using Logistic Regression as 3rd model")
        from sklearn.linear_model import LogisticRegression
        classifiers["Logistic Regression"] = LogisticRegression(
            C=1.0, max_iter=1000, random_state=42, class_weight="balanced",
            multi_class="multinomial",
        )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    target_names = [LABEL_NAMES[i] for i in range(5)]

    for name, clf in classifiers.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")

        all_preds = np.zeros_like(y)
        fold_scores = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            if "CatBoost" in name:
                X_train = X_train.toarray()
                X_test = X_test.toarray()

            clf_copy = clone_classifier(clf, name)
            clf_copy.fit(X_train, y_train)
            preds = np.asarray(clf_copy.predict(X_test)).ravel()
            all_preds[test_idx] = preds

            acc = accuracy_score(y_test, preds)
            fold_scores.append(acc)
            print(f"  Fold {fold+1}: accuracy={acc:.4f}")

        overall_acc = accuracy_score(y, all_preds)
        macro_f1 = f1_score(y, all_preds, average="macro")
        weighted_f1 = f1_score(y, all_preds, average="weighted")

        print(f"\n  Overall accuracy: {overall_acc:.4f}")
        print(f"  Macro F1: {macro_f1:.4f}")
        print(f"  Weighted F1: {weighted_f1:.4f}")
        print(f"  Mean fold accuracy: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")

        print(f"\n  Classification Report:")
        print(classification_report(y, all_preds, target_names=target_names, digits=3))

        cm = confusion_matrix(y, all_preds)
        print(f"  Confusion Matrix:")
        print(f"  {'':12s} " + " ".join(f"{n:>8s}" for n in target_names))
        for i, row in enumerate(cm):
            print(f"  {target_names[i]:12s} " + " ".join(f"{v:8d}" for v in row))

        results[name] = {
            "accuracy": overall_acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "fold_scores": fold_scores,
            "predictions": all_preds,
        }

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':25s} {'Accuracy':>10s} {'Macro F1':>10s} {'Weighted F1':>12s}")
    print("-" * 60)
    best_name = None
    best_f1 = 0
    for name, res in results.items():
        print(f"{name:25s} {res['accuracy']:10.4f} {res['macro_f1']:10.4f} {res['weighted_f1']:12.4f}")
        if res["weighted_f1"] > best_f1:
            best_f1 = res["weighted_f1"]
            best_name = name

    print(f"\nBest model: {best_name} (weighted F1={best_f1:.4f})")

    print(f"\nTraining {best_name} on all data...")
    best_clf = clone_classifier(classifiers[best_name], best_name)
    X_all = X.toarray() if "CatBoost" in best_name else X
    best_clf.fit(X_all, y)

    model_path = Path("models/line_classifier.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({
            "model_name": best_name,
            "classifier": best_clf,
            "tfidf_word": tfidf_word,
            "tfidf_char": tfidf_char,
            "feature_names": feat_names,
            "label_names": LABEL_NAMES,
            "accuracy": results[best_name]["accuracy"],
            "macro_f1": results[best_name]["macro_f1"],
        }, f)
    print(f"Saved: {model_path}")


def clone_classifier(clf, name):
    """Create a fresh copy of a classifier."""
    if "Random Forest" in name:
        return RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=3,
            random_state=42, n_jobs=-1, class_weight="balanced",
        )
    elif "SVM" in name:
        return make_pipeline(
            MaxAbsScaler(),
            SVC(kernel="rbf", C=10.0, gamma="scale",
                random_state=42, class_weight="balanced"),
        )
    elif "CatBoost" in name:
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.1,
            verbose=0, random_seed=42, auto_class_weights="Balanced",
        )
    elif "Logistic" in name:
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            C=1.0, max_iter=1000, random_state=42, class_weight="balanced",
            multi_class="multinomial",
        )
    return clf


if __name__ == "__main__":
    train_and_evaluate()
