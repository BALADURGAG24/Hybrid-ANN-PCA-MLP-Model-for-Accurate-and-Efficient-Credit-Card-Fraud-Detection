from preprocess import load_and_preprocess_data
from model import build_ann
from evaluate import evaluate_model
from utils import balance_with_smote

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
# print("Current working directory:", os.getcwd())

def run():
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data("data/creditcard.csv")

    print("Balancing training data with SMOTE...")
    X_train_bal, y_train_bal = balance_with_smote(X_train, y_train)

    print("Building ANN model...")
    model = build_ann(X_train_bal.shape[1])

    if not os.path.exists('models'):
        os.makedirs('models')

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, verbose=1),
        ModelCheckpoint('models/fraud_model.h5', save_best_only=True, verbose=1)
    ]

    print("Training model...")
    model.fit(X_train_bal, y_train_bal,
              epochs=50,
              batch_size=2048,
              validation_split=0.2,
              callbacks=callbacks)

    print("Evaluating model on test data...")
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    run()
