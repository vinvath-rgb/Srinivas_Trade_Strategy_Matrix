# regime_matrix_app/data_utils.py
import pandas as pd
import io

def read_prices_upload(upload):
    if upload is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(upload)
    except Exception:
        upload.seek(0)
        df = pd.read_csv(upload, encoding_errors="ignore")
    return df