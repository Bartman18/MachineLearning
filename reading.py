import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split


data = pd.read_excel('excel1.xlsx')  # już jako DataFrame

selector = SelectKBest(score_func=f_classif, k=2)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

selected_features = X.columns[selector.get_support()]
f_scores = selector.scores_[selector.get_support()]

print(f"Selected Features: {selected_features}")
print(f"F-Scores: {f_scores}")