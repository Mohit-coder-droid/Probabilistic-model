import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def median_rank(n:int, i:int)->float:
    """Function to compute cumulative probability using Benard's median rank formula"""
    return (i - 0.3) / (n + 0.4)

def df_processor(df:pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df_1 = df.sort_values(by="Temperature", ascending=False).reset_index(drop=True)
    df_dict = {temp: df_1[df_1["Temperature"] == temp].reset_index(drop=True) for temp in df_1["Temperature"].unique()}

    df_1['Inverse_Temp'] = 11604.53 / (df_1['Temperature'] + 273.16)  # Convert to Kelvin
    df_1['Ln_Mpa'] = np.log(df_1['Mpa'])  # Log transformation

    return df_1, df_dict
