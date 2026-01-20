import joblib
import os
from sklearn.model_selection import train_test_split

from src.config import *
from src.data_loader import load_data
from src.preprocessing import clean_text
from src.feature_engineering import tfidf_features
from src.models.naive_bayes import train_naive_bayes
from src.models.deep_learning import build_dl_model
from src.evaluation import evaluate_model  

os.makedirs("models_saved", exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.makedirs(f"{BASE_RESULTS_DIR}/naive_bayes", exist_ok=True)
os.makedirs(f"{BASE_RESULTS_DIR}/deep_learning", exist_ok=True)

df = load_data("data/sentiment_analysis.csv")

df["clean_text"] = df["text"].apply(clean_text)

df["sentiment_encoded"] = df["sentiment"].map({
    "negative": 0,
    "neutral": 1,
    "positive": 2
})

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["sentiment_encoded"],
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["sentiment_encoded"]
)

X_train_tfidf, X_test_tfidf, vectorizer = tfidf_features(
    X_train, X_test, MAX_FEATURES
)

joblib.dump(vectorizer, "models_saved/tfidf_vectorizer.pkl")

nb_model = train_naive_bayes(X_train_tfidf, y_train)

joblib.dump(nb_model, "models_saved/naive_bayes.pkl")

evaluate_model(
    nb_model,
    X_test_tfidf,
    y_test,
    f"{BASE_RESULTS_DIR}/naive_bayes",
    "Naive Bayes"
)

X_train_dl = X_train_tfidf.toarray()
X_test_dl = X_test_tfidf.toarray()

dl_model = build_dl_model(X_train_dl.shape[1])
dl_model.fit(
    X_train_dl,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=0
)

dl_model.save("models_saved/deep_learning.h5")

evaluate_model(
    dl_model,
    X_test_dl,
    y_test,
    f"{BASE_RESULTS_DIR}/deep_learning",
    "Deep Learning"
)

print(" Models, vectorizer, and evaluation results saved successfully.")
