import pandas as pd
import numpy as np
from sklearn.base import is_classifier, is_regressor
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.base import clone


def reading_data():
    data = pd.read_excel('excel1.xlsx')

    X = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()

    return X, y


def materiality_testing(X, y):
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    selector = SelectKBest(score_func=f_classif, k=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    selected_features = X.columns[selector.get_support()]
    f_scores = selector.scores_[selector.get_support()]

    print(f"Selected Features: {selected_features}")
    print(f"F-Scores: {f_scores}")


# def feature_to_price(X, y):
#     for i in range(X.shape[1]):
#
#         fig, ax = plt.subplots()
#
#
#         feature_data = X[:, i]
#
#
#         ax.bar(feature_data, y)
#         ax.set_xlabel(f'Feature {i + 1}')
#         ax.set_ylabel('Price')
#         ax.set_title(f'Feature {i + 1} vs Price')
#         plt.tight_layout()
#         plt.savefig(f'({i})')


def features_select(X, y):
    n_features = X.shape[1]

    models = [
        SVC(),
        KNeighborsClassifier(),
        GaussianNB(),
        MLPClassifier(),
        DecisionTreeClassifier(),
    ]

    for model in models:
        result = np.zeros((n_features, 10))
        rkf = RepeatedKFold(n_splits=2, n_repeats=5)
        for j in range(1, n_features + 1):
            selector = SelectKBest(score_func=f_classif, k=j)
            X_new = selector.fit_transform(X, y)

            for i, (train_index, test_index) in enumerate(rkf.split(X_new, y)):
                X_train, X_test = X_new[train_index], X_new[test_index]
                y_train, y_test = y[train_index], y[test_index]

                clf = model
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                scores = accuracy_score(y_pred, y_test)
                result[j - 1, i] = scores

        z = list(range(1, n_features + 1))
        plt.figure(figsize=(8, 5))
        plt.plot(z, np.mean(result, axis=1), marker='o')
        plt.grid(True)
        plt.xticks(z)
        plt.savefig(f'{model}_v2.png')


def model_selection(X, y):
    models = [
        SVC(),
        KNeighborsClassifier(),
        GaussianNB(),
        MLPClassifier(max_iter=1000),
        SVR(),
        DecisionTreeClassifier(),
    ]

    results = {}

    rkf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=42)

    for model in models:
        scores = []

        for train_index, test_index in rkf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if is_classifier(model):
                score = accuracy_score(y_test, y_pred)
            elif is_regressor(model):
                score = r2_score(y_test, y_pred)
            else:
                continue

            scores.append(score)

        results[model.__class__.__name__] = np.mean(scores)

    for x, y in results.items():
        print(f'{x}: {y}')


def best_ratio(X, y):
    rkf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=42)
    model = GaussianNB()

    result = {}

    for train_index, test_index in rkf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        errors = np.where(y_pred != y_test)[0]
        global_error_indices = test_index[errors]

        max_error = 0

        for idx in global_error_indices:
            pred = model.predict(X[idx].reshape(1, -1))[0]
            true = y[idx]
            error = (true - pred)

            if error > max_error:
                result.clear()
                result[int(idx)] = (int(pred), int(true))
                max_error = error
            elif error == max_error:
                result[int(idx)] = (int(pred), int(true))

    for x, y in result.items():
        print(f'Wiersz {x}: Prawidłowe {y[0]}, Rozponane {y[1]}')
