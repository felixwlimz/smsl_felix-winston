from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import mlflow
from modeling import read_dataset, split_dataset

def tune_logreg_model(X_train, y_train):
    params = {
        'C' : np.logspace(-4, 4, 20),
        'penalty' : ['l1', 'l2'],
        'solver' : ['liblinear', 'saga'],
        'max_iter' : [500, 1000, 2000, 5000],
    }
    model = LogisticRegression(max_iter=5000, random_state=42)
    search = RandomizedSearchCV(
        model, 
        param_distributions=params, 
        n_iter=20, 
        scoring='accuracy',
        cv=5, 
        random_state=42,
        verbose=1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


df = read_dataset('../preprocessing/cardio_train_scaled.csv')
X_train, X_test, y_train, y_test = split_dataset(df)
best_model = tune_logreg_model(X_train, y_train)
print(f"Best parameters: {best_model.get_params()}")
