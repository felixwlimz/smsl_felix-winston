import os
import pandas as pd 
import numpy as np 
import mlflow 
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Set up MLflow tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Cardio Health Prediction")

def read_dataset(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    df = pd.read_csv(file_path)
    return df 

def split_dataset(df: pd.DataFrame) -> tuple:
    X = df.drop(columns=['cardio'])
    y = df['cardio']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def logreg_model(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000, random_state=42)
    
    with mlflow.start_run(run_name="LogisticRegression"):
        mlflow.set_experiment("Cardio Health Prediction")
        mlflow.sklearn.autolog()
        
        model.fit(X_train, y_train)
        predict = model.predict(X_test)
        acc = accuracy_score(y_test, predict)
        
        mlflow.log_metric('accuracy', acc)
        mlflow.sklearn.log_model(model, "model", input_example=X_train.iloc[:5])
        print(f"[LogReg] Accuracy: {acc:.4f}")

def deep_learning_model(X_train, y_train, X_test, y_test):
    model = Sequential([
        Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

    with mlflow.start_run(run_name="DeepLearning"):
        mlflow.set_experiment("Cardio Health Prediction")
        mlflow.tensorflow.autolog()
        
        model.fit(X_train, y_train, 
                  epochs=50, 
                  batch_size=32, 
                  validation_split=0.2,
                  callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
                  verbose=0)

        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        mlflow.log_metric('accuracy', acc)
        print(f"[DeepLearning] Accuracy: {acc:.4f}")

def random_forest_model(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    with mlflow.start_run(run_name="RandomForest"):
        mlflow.set_experiment("Cardio Health Prediction")
        mlflow.sklearn.autolog()

        model.fit(X_train, y_train)
        predict = model.predict(X_test)
        acc = accuracy_score(y_test, predict)
        
        mlflow.log_metric('accuracy', acc)
        mlflow.sklearn.log_model(model, "rf_model", input_example=X_train.iloc[:5])
        print(f"[RandomForest] Accuracy: {acc:.4f}")

# === Main Execution ===
if __name__ == "__main__":
    dataset_path = "../preprocessing/cardio_train_scaled.csv"
    main_df = read_dataset(dataset_path)

    X_train, X_test, y_train, y_test = split_dataset(main_df)

    logreg_model(X_train, y_train, X_test, y_test)
    deep_learning_model(X_train, y_train, X_test, y_test)
    random_forest_model(X_train, y_train, X_test, y_test)
