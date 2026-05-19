FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir \
    pandas numpy scikit-learn imbalanced-learn \
    xgboost mlflow dagshub

CMD ["python", "modelling.py"]