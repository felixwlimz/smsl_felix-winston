import pandas as pd 
import numpy as np 
import mlflow 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Deep Learning Model 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Read the dataset 
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment('Cardio Health Prediction')

def read_dataset(file_path : str) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=',')
    return df 

def split_dataset(df : pd.DataFrame) -> tuple:
    X = df.drop(columns=['cardio'])
    y = df['cardio']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def logreg_model(X_train, y_train, X_test, y_test):
   
    model = LogisticRegression(max_iter=1000, random_state=42)
    with mlflow.start_run() :
        mlflow.autolog()
        model.fit(X_train, y_train)
        predict = model.predict(X_test)
        acc = accuracy_score(y_test, predict)
        mlflow.log_metric('accuracy', acc)
        mlflow.sklearn.log_model(model, "model", input_example=X_train.iloc[:5])
        print(f"Model accuracy: {acc}")
        

def deep_learning_model(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(len(X_train.columns),)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model 
    
def random_forest_model(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    with mlflow.start_run() :
        mlflow.autolog()
        model.fit(X_train, y_train)
        predict = model.predict(X_test)
        acc = accuracy_score(y_test, predict)
        mlflow.log_metric('accuracy', acc)
        mlflow.sklearn.log_model(model, "rf_model", input_example=X_train.iloc[:5])
        print(f"Random Forest Model accuracy: {acc}")
    

main_df = read_dataset('../preprocessing/cardio_train_scaled.csv')
X_train, X_test, y_train, y_test = split_dataset(main_df)
logreg_model(X_train, y_train, X_test, y_test)


deep_model = deep_learning_model(X_train, y_train, X_test, y_test)
with mlflow.start_run() :
    mlflow.autolog()
    deep_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
    loss, acc = deep_model.evaluate(X_test, y_test)
    mlflow.log_metric('accuracy', acc)
    mlflow.tensorflow.log_model(deep_model, "deep_model")
    print(f"Deep Learning Model accuracy: {acc}")
    
random_forest_model(X_train, y_train, X_test, y_test)

