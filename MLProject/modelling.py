import pandas as pd
import mlflow
import mlflow.xgboost
import argparse
import dagshub
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)

parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators',  type=int,   default=200)
parser.add_argument('--max_depth',     type=int,   default=5)
parser.add_argument('--learning_rate', type=float, default=0.1)
args = parser.parse_args()

# Init DagsHub — otomatis setup tracking URI + artifact storage
dagshub.init(
    repo_owner='Yud1Pp',
    repo_name='Membangun_Model_YudiPratamaPutra',
    mlflow=True
)

X_train = pd.read_csv('telco_preprocessing/X_train.csv')
X_test  = pd.read_csv('telco_preprocessing/X_test.csv')
y_train = pd.read_csv('telco_preprocessing/y_train.csv').squeeze()
y_test  = pd.read_csv('telco_preprocessing/y_test.csv').squeeze()

print(f"[INFO] X_train: {X_train.shape} | X_test: {X_test.shape}")

with mlflow.start_run() as run:
    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    mlflow.log_param('n_estimators',  args.n_estimators)
    mlflow.log_param('max_depth',     args.max_depth)
    mlflow.log_param('learning_rate', args.learning_rate)

    mlflow.log_metric('accuracy',  acc)
    mlflow.log_metric('precision', prec)
    mlflow.log_metric('recall',    rec)
    mlflow.log_metric('f1_score',  f1)

    # Log model + register sekaligus
    mlflow.xgboost.log_model(
        xgb_model=model,
        artifact_path='model',
        registered_model_name='telco-churn-xgboost'
    )

    run_id = run.info.run_id
    with open('run_id.txt', 'w') as f:
        f.write(run_id)

    print(f"[INFO] Run ID: {run_id}")
    print(f"[RESULT] Accuracy : {acc:.4f}")
    print(f"[RESULT] Precision: {prec:.4f}")
    print(f"[RESULT] Recall   : {rec:.4f}")
    print(f"[RESULT] F1 Score : {f1:.4f}")
    print("✅ CI modelling selesai!")