import joblib
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference


df = pd.read_csv('data/census.csv')

model = joblib.load('model/rf.joblib')
encoder = joblib.load('model/encoder.joblib')
lb = joblib.load('model/lb.joblib')

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

sliced_perf = []
for feature in cat_features:
    for feature_value in df[feature].unique():
        # slicing dataframe with respect to feature and feature_value
        sliced_df = df[df[feature] == feature_value]
        X_slice, y_slice, _, _ = process_data(
            sliced_df,
            categorical_features=cat_features,
            label="salary", training=False,
            encoder=encoder, lb=lb)
        predictions_slice = inference(model, X_slice)
        precision, recall, f_beta = compute_model_metrics(y_slice, predictions_slice)
        slicing_perf_dict = {
            'feature' : feature,
            'feature_value' : feature_value,
            'precision' : precision,
            'recall' : recall,
            'f_beta' : f_beta,
        }
        sliced_perf.append(slicing_perf_dict)

sliced_df = pd.DataFrame(sliced_perf)
sliced_df.to_csv('sliced_output.txt')
