import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def median_rank(n:int, i:int)->int:
    """Function to compute cumulative probability using Benard's median rank formula"""
    return (i - 0.3) / (n + 0.4)

def df_processor(df):
    df_1 = df.sort_values(by="Temperature", ascending=True).reset_index(drop=True)
    df_dict = {temp: df_1[df_1["Temperature"] == temp].reset_index(drop=True) for temp in df_1["Temperature"].unique()}

    df['Inverse_Temp'] = 11604.53 / (df['Temperature'] + 273.16)  # Convert to Kelvin
    df['Ln_Mpa'] = np.log(df['Mpa'])  # Log transformation

    return df, df_dict


