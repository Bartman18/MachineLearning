from reading import reading_data, add_features, model_selection, best_ratio

X, y = reading_data()

X_new, y_v2 = add_features(X, y)


model_selection(X_new, y_v2)
model_selection(X, y)


best_ratio(X,y)