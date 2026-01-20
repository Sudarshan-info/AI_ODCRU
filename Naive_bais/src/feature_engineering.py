from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_features(train_text, test_text, max_features):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)
    return X_train, X_test, vectorizer