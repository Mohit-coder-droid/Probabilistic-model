import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def median_rank(n:int, i:int)->float:
    """Function to compute cumulative probability using Benard's median rank formula"""
    return (i - 0.3) / (n + 0.4)

def df_processor(df:pd.DataFrame, temp_col, stress_col) -> tuple[pd.DataFrame, dict]:
    """Basic processing for streamlit app"""
    df_1 = df.sort_values(by=temp_col, ascending=False).reset_index(drop=True)
    df_dict = {temp: df_1[df_1[temp_col] == temp].reset_index(drop=True) for temp in df_1[temp_col].unique()}

    df_1['Inverse_Temp'] = 11604.53 / (df_1[temp_col] + 273.16)  # Convert to Kelvin
    df_1['Ln_Mpa'] = np.log(df_1[stress_col])  # Log transformation

    return df_1, df_dict

def fatigue_crack_preprocess_df(df, temp_col, c_col, m_col, krange_col, r_ratio_col):
    # Create a list to hold expanded rows
    expanded_rows = []

    # Loop through each row
    for _, row in df.iterrows():
        # Parse the ΔK range
        try:
            k_range = str(row[krange_col]).strip()
            k_min, k_max = map(float, k_range.split('-'))
        except:
            continue

        # Generate integer ΔK values within the range
        delta_k_values = list(range(int(k_min), int(k_max) + 1))

        # Create a new row for each ΔK value
        for delta_k in delta_k_values:
            expanded_rows.append({
                'Temperature, C': row[temp_col],
                'c': row[c_col],
                'm': row[m_col],
                'R- Ratio': row[r_ratio_col],
                'Delta K': delta_k
            })

    expanded_df = pd.DataFrame(expanded_rows)
    return expanded_df