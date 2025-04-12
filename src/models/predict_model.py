import pickle

def predict_cluster(model, scaler, df):
    df_scaled = scaler.transform(df)
    return model.predict(df_scaled)