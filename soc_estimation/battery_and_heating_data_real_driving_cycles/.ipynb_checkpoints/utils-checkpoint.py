def get_steady_cols(df):
    steady_cols = [col for col in df.columns if df[col].nunique() == 1]
    return steady_cols