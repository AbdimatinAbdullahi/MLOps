import os
import boto3
import joblib
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

if __name__ == "__main__":
    print("[INFO] Extracting arguments")
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train-file", type=str, default="train-hr.csv")
    parser.add_argument("--test-file", type=str, default="test-hr.csv")

    args, _ = parser.parse_known_args()

    print("[INFO] Reading data")

    # Ensure the correct data source for test data
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    features = list(train_df.columns)
    label = features[0]
    print("Building Training and testing datasets")
    
    X_train = train_df.drop(label, axis=1)
    y_train = train_df[label]

    X_test = test_df.drop(label, axis=1)
    y_test = test_df[label]

    model = RandomForestClassifier(n_estimators=100)  # Increased n_estimators for better performance
    model.fit(X_train, y_train)
    
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print("Model persisted at " + model_path)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
