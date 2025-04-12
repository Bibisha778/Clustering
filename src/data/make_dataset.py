import pandas as pd

def load_and_preprocess_data(path):
    df = pd.read_csv(path)

    # Drop ID column if it exists
    df.drop(columns=["Customer_ID"], errors="ignore", inplace=True)

    # Encode Gender (map and fill)
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

    # Fill missing Gender with mode only if mode exists
    gender_mode = df["Gender"].mode()
    if not gender_mode.empty:
        df["Gender"].fillna(gender_mode[0], inplace=True)
    else:
        df["Gender"].fillna(0, inplace=True)  # fallback default

    # Fill any numeric NaNs
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Optional: sanity check
    assert df.isnull().sum().sum() == 0, "Data still contains NaNs!"

    return df
